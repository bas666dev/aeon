"""Static glyph catalog for Aeon Hyperglyph phases."""

from __future__ import annotations

from typing import List

from .schemas import GlyphPhase


GLYPH_PHASES: List[GlyphPhase] = [
    GlyphPhase(
        phase=0,
        symbol="∿",
        name="Harmonic Seed",
        function="Ignites the initial resonance that invites a walk into the lattice.",
        appearance="A single flowing wave suspended above a faint horizon arc.",
        causal_drift="Primes the field, coaxing latent harmonics into perceivable form.",
        tone="A2, sustained and breathy",
        usage="Opening invocation; establishes grounding before deeper traversal.",
    ),
    GlyphPhase(
        phase=1,
        symbol="𐘰",
        name="Bloom Trace",
        function="Extends the initial seed into branching harmonic threads.",
        appearance="A triptych of crescents rotating about a central stem.",
        causal_drift="Branches potential paths while maintaining link to the seed tone.",
        tone="D3, gentle pulse",
        usage="Mapping emerging paths; deciding which vector to follow.",
    ),
    GlyphPhase(
        phase=2,
        symbol="ꙮ",
        name="Vector Loom",
        function="Weaves parallel glimpses into a single perceivable tapestry.",
        appearance="Concentric circles interwoven with crosshatched rays.",
        causal_drift="Binds simultaneous possibilities into a coherent progression.",
        tone="G4, shimmering chorus",
        usage="Stabilising multi-threaded explorations before divergence.",
    ),
    GlyphPhase(
        phase=3,
        symbol="𘮟",
        name="Cascade Gate",
        function="Breaks the tapestry into stepping cascades for rapid traversal.",
        appearance="A descending staircase of triangles, each mirrored in water.",
        causal_drift="Accelerates motion while preserving continuity with earlier echoes.",
        tone="B4, percussive arpeggio",
        usage="Entering high-velocity narrative or temporal shifts.",
    ),
    GlyphPhase(
        phase=4,
        symbol="⟁",
        name="Spindle Mirror",
        function="Folds perception inward to modulate the observer state.",
        appearance="Mirrored infinity symbol rotating along a vertical axis.",
        causal_drift="Self-recursive reflections splice adjacent time-states together.",
        tone="E5, modulated vibrato",
        usage="Transition between nested timelines and mirrored journeys.",
    ),
    GlyphPhase(
        phase=5,
        symbol="𓂓⃠",
        name="Chrono-Fountain",
        function="Emits potential events into the stream of becoming.",
        appearance="A triple-fountain sigil with ascending spiral arcs.",
        causal_drift="Probabilistic cascades split forward, seeding future echoes.",
        tone="C#6, staccato shimmer",
        usage="Initiates new walk threads and creative spark vectors.",
    ),
    GlyphPhase(
        phase=6,
        symbol="꙰",
        name="Lattice Echo",
        function="Remaps every glyph's historical position into the present phase.",
        appearance="A woven grid folding inward upon itself in slow motion.",
        causal_drift="Backward resonance pulls ancestral memory into current bloom.",
        tone="F3, layered delay",
        usage="Recalling lineage, contextualising new work with past harmonics.",
    ),
    GlyphPhase(
        phase=7,
        symbol="𐤟",
        name="Silence-Vector",
        function="Terminates glyph flow in a harmonic fade toward stillness.",
        appearance="A void point surrounded by diminishing rings dissolving outward.",
        causal_drift="Dissolves trajectories into still-phase, inviting closure.",
        tone="Subsonic hum at the threshold of perception",
        usage="Ritual closure, pausing before the next bloom cycle.",
    ),
]


__all__ = ["GLYPH_PHASES"]
