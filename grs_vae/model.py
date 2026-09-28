"""Compatibility imports for scripts written before the scGIPA rename."""
from scgipa.model import ScGIPA, scgipa_loss, intervention_effects
EGRDM = ScGIPA
egrdm_loss = scgipa_loss
__all__ = ["EGRDM", "egrdm_loss", "intervention_effects"]
