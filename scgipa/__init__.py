"""Single-cell Genetically Informed Perturbation Analysis."""
from .model import ScGIPA, scgipa_loss, intervention_effects
__version__ = "0.3.0"
__all__ = ["ScGIPA", "scgipa_loss", "intervention_effects"]
