"""Conditional variational model used by the public toy workflow."""

import torch
from torch import nn
from torch.nn import functional as F


class GRSVAE(nn.Module):
    """VAE conditioned on cell identity, disease state, and a route-prior vector."""

    def __init__(self, n_genes, n_cell_types, n_states, latent_dim=12, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.cell_embedding = nn.Embedding(n_cell_types, latent_dim)
        self.state_embedding = nn.Embedding(n_states, latent_dim)
        self.prior_projection = nn.Linear(n_genes, latent_dim, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes)
        )
        self.state_head = nn.Linear(latent_dim, n_states)
        self.stage_head = nn.Linear(latent_dim, 1)

    def encode(self, expression):
        h = self.encoder(expression)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def condition(self, cell_type, state, prior):
        return self.cell_embedding(cell_type) + self.state_embedding(state) + self.prior_projection(prior)

    def decode(self, z, cell_type, state, prior):
        context = self.condition(cell_type, state, prior)
        return self.decoder(torch.cat([z, context], dim=1))

    def forward(self, expression, cell_type, state, prior):
        mu, logvar = self.encode(expression)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, cell_type, state, prior)
        return reconstruction, mu, logvar, self.state_head(mu), self.stage_head(mu).squeeze(1)


def grs_vae_loss(output, expression, state, stage, beta_kl=1e-3):
    """Reconstruction, variational, state-classification and ordered-stage loss."""
    reconstruction, mu, logvar, state_logits, stage_hat = output
    reconstruction_loss = F.mse_loss(reconstruction, expression)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    state_loss = F.cross_entropy(state_logits, state)
    stage_loss = F.mse_loss(stage_hat, stage)
    total = reconstruction_loss + beta_kl * kl_loss + 0.35 * state_loss + 0.15 * stage_loss
    return total, {
        "reconstruction": reconstruction_loss.detach(),
        "kl": kl_loss.detach(),
        "state": state_loss.detach(),
        "stage": stage_loss.detach(),
    }
