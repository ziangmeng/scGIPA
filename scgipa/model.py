"""Expression and genetic-route model for four brain states."""
import torch
from torch import nn
from torch.nn import functional as F

class ScGIPA(nn.Module):
    """Predict state from expression, cell identity and a signed cell-by-gene prior.

    Labels are training targets, never forward-pass inputs. Layer layout follows
    the study implementation; dimensions are configurable for the CPU demo.
    """
    def __init__(self, n_genes, n_cell_types, prior, n_states=4, hidden_dim=64, latent_dim=12):
        super().__init__()
        prior = torch.as_tensor(prior, dtype=torch.float32).clone()
        if prior.shape != (n_cell_types, n_genes) or not torch.isfinite(prior).all():
            raise ValueError("prior must be a finite cell-type by gene matrix")
        self.register_buffer("prior", prior)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.mu, self.logvar = nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)
        self.cell_emb = nn.Embedding(n_cell_types, latent_dim)
        self.route_encoder = nn.Sequential(nn.Linear(n_genes, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                                           nn.GELU(), nn.Linear(hidden_dim // 2, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                     nn.Dropout(.1), nn.Linear(hidden_dim, n_genes), nn.Softplus())
        self.classifier = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(.1),
                                        nn.Linear(hidden_dim // 2, n_states))
        self.stage = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def compose(self, z, x, cell, masked_genes=()):
        route = x * self.prior[cell]
        if len(masked_genes):
            route = route.clone()
            route[:, list(masked_genes)] = 0
        return z + self.cell_emb(cell) + self.route_encoder(route)

    def forward(self, x, cell, stochastic=True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if self.training and stochastic else mu
        integrated = self.compose(z, x, cell)
        return self.decoder(integrated), self.classifier(integrated), self.stage(z).squeeze(1), mu, logvar

    def anchor_loss(self, targets):
        """Align route-supported reconstruction with training-donor contrasts."""
        terms = []
        for (cell, _contrast), delta in targets.items():
            supported = self.prior[cell].abs() > 0
            if int(supported.sum()) < 3:
                continue
            delta = torch.as_tensor(delta, dtype=self.prior.dtype, device=self.prior.device)
            base = self.cell_emb(torch.tensor([cell], device=self.prior.device))
            route_delta = self.route_encoder((delta * self.prior[cell]).unsqueeze(0))
            prediction = (self.decoder(base + route_delta) - self.decoder(base)).flatten()
            terms.append(1 - F.cosine_similarity(prediction[supported], delta[supported], dim=0))
        return torch.stack(terms).mean() if terms else self.prior.new_zeros(())

def scgipa_loss(model, output, x, state, anchor_targets, beta_kl=.001, stage_weight=.2, anchor_weight=.2):
    """Reconstruction, classification, auxiliary disease stage, KL and anchoring."""
    rec, logits, stage, mu, logvar = output
    pathology = state >= 1
    ordinal = F.mse_loss(stage[pathology], state[pathology].float() - 1) if pathology.any() else x.new_zeros(())
    kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return (F.mse_loss(rec, x) + F.cross_entropy(logits, state) + stage_weight * ordinal
            + beta_kl * kl + anchor_weight * model.anchor_loss(anchor_targets))

@torch.no_grad()
def intervention_effects(model, x, cell, reference, genes, previous_state, target_state):
    """Paired per-donor I and M for a gene or jointly perturbed gene set.

    I resets expression and recomputes both expression-dependent branches.
    M preserves expression and its encoding, masking only selected route inputs.
    Both use original minus intervened target-minus-previous logit margin.
    """
    if model.training:
        raise ValueError("call model.eval() before deterministic interventions")
    if previous_state == target_state:
        raise ValueError("the two states must differ")
    genes = list(genes)
    mu, _ = model.encode(x)
    baseline = model.classifier(model.compose(mu, x, cell))
    reset = x.clone()
    reset[:, genes] = reference[..., genes]
    reset_mu, _ = model.encode(reset)
    expression_logits = model.classifier(model.compose(reset_mu, reset, cell))
    route_logits = model.classifier(model.compose(mu, x, cell, genes))
    def margin(logits):
        return logits[:, target_state] - logits[:, previous_state]
    return margin(baseline) - margin(expression_logits), margin(baseline) - margin(route_logits)
