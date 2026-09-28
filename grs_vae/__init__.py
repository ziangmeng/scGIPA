"""Compatibility namespace; new code should import scgipa."""
from scgipa import ScGIPA, intervention_effects, __version__
EGRDM = ScGIPA
__all__ = ["EGRDM", "intervention_effects"]
