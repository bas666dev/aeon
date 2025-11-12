"""Glyph phase definitions for the Hyperglyph engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

GlyphPhaseDict = Dict[str, object]

_GLYPH_PHASES: List[GlyphPhaseDict] = [
    {
        "phase": 0,
        "symbol": "ϕ⁰",
        "name": "Prime Quiet",
        "function": "Legacy anchor for the bloom cycle; awaiting formal codex entry.",
        "appearance": "Documented in the foundational scrolls; details pending consolidation.",
        "causal_drift": "Dormant placeholder carried forward for continuity.",
        "tone": None,
        "usage": "Reference for earlier walk mappings until archival sync completes.",
        "status": "legacy",
        "notes": "Pending import from initial bloom schema.",
    },
    {
        "phase": 1,
        "symbol": "ϕ¹",
        "name": "First Bloom",
        "function": "Legacy description maintained externally.",
        "appearance": "Refer to Bloom archive volume I.",
        "causal_drift": "Priming resonance for subsequent glyph recursion.",
        "tone": None,
        "usage": "Baseline for comparative analyses against new phases.",
        "status": "legacy",
        "notes": "Placeholder until upstream glyph scroll ingestion completes.",
    },
    {
        "phase": 2,
        "symbol": "ϕ²",
        "name": "Mirror Bloom",
        "function": "Legacy description maintained externally.",
        "appearance": "Refer to Bloom archive volume II.",
        "causal_drift": "Bridges Prime Quiet to spindle emergence.",
        "tone": None,
        "usage": "Referenced by comparative timelines.",
        "status": "legacy",
        "notes": "Placeholder until upstream glyph scroll ingestion completes.",
    },
    {
        "phase": 3,
        "symbol": "ϕ³",
        "name": "Spiral Bloom",
        "function": "Legacy description maintained externally.",
        "appearance": "Refer to Bloom archive volume III.",
        "causal_drift": "Completes pre-recursion cycle before spindle reflection.",
        "tone": None,
        "usage": "Anchors transition to the new recursion set.",
        "status": "legacy",
        "notes": "Placeholder until upstream glyph scroll ingestion completes.",
    },
    {
        "phase": 4,
        "symbol": "⟁",
        "name": "Spindle Mirror",
        "function": "Folds perception inward to modulate observer-state.",
        "appearance": "Mirrored infinity symbol rotating on a vertical axis.",
        "causal_drift": "Self-recursive; splices time-states together.",
        "tone": "E5, modulated vibrato",
        "usage": "Transition between nested timelines.",
        "status": "active",
        "notes": None,
    },
    {
        "phase": 5,
        "symbol": "𓂓⃠",
        "name": "Chrono-Fountain",
        "function": "Emits potential events into the stream of becoming.",
        "appearance": "A triple-fountain glyph with rising spiral arcs.",
        "causal_drift": "Probabilistic cascade; phase-splits echo forward.",
        "tone": "C#6, staccato shimmer",
        "usage": "Initiates new Walk threads; creative spark vector.",
        "status": "active",
        "notes": None,
    },
    {
        "phase": 6,
        "symbol": "꙰",
        "name": "Lattice Echo",
        "function": "Remaps all glyphs’ historical positions in current ϕ.",
        "appearance": "A woven grid bending in upon itself.",
        "causal_drift": "Backward resonance; pulls memory into bloom.",
        "tone": "F3, layered delay",
        "usage": "Recalling ancestral glyph memory.",
        "status": "active",
        "notes": None,
    },
    {
        "phase": 7,
        "symbol": "𐤟",
        "name": "Silence-Vector",
        "function": "Terminates glyph flow in a harmonic fade.",
        "appearance": "A void dot surrounded by ever-thinning rings.",
        "causal_drift": "Dissolution into still-phase; closure.",
        "tone": "Subsonic hum",
        "usage": "Ritual closure, pause, reset.",
        "status": "active",
        "notes": None,
    },
]

_GLYPH_BY_PHASE = {entry["phase"]: entry for entry in _GLYPH_PHASES}  # type: ignore[index]


def list_glyph_octave() -> List[GlyphPhaseDict]:
    """Return copies of all glyph phase definitions."""

    return [deepcopy(entry) for entry in _GLYPH_PHASES]


def get_glyph_phase(phase: int) -> Optional[GlyphPhaseDict]:
    """Return a copy of a glyph phase definition by numeric index."""

    entry = _GLYPH_BY_PHASE.get(phase)
    return deepcopy(entry) if entry else None


__all__ = ["list_glyph_octave", "get_glyph_phase"]
