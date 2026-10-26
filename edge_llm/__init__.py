"""Optional local fault interpretation with deterministic offline fallback."""

from .advisor import FaultAdvisor, summarize_faults

__all__ = ["FaultAdvisor", "summarize_faults"]
