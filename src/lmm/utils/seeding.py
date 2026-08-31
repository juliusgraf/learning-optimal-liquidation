"""Seeding (ruling D10, fully functional in Phase 2).

Legacy seeding is untrusted and rebuilt fresh: ONE ``numpy.random.Generator``
per named component, spawned from a master seed via
``np.random.SeedSequence.spawn``; torch is seeded from the same master seed
with determinism flags set. Global ``np.random.*`` / ``random.*`` are NEVER
touched. Same config + seed => bit-identical metrics across processes.

Stream identity depends on the master seed and on the SORTED list of
component names (spawn order is the sorted order, so it is independent of the
caller's argument order; adding a component changes the streams of components
that sort after it — by design, streams are a function of the full component
set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["SeedBundle", "seed_everything", "spawn_child"]


@dataclass(frozen=True)
class SeedBundle:
    """One independent ``np.random.Generator`` per component, plus the
    per-component ``SeedSequence`` used to spawn further children
    (e.g. per-episode environment seeds)."""

    master_seed: int
    generators: dict[str, np.random.Generator]
    sequences: dict[str, np.random.SeedSequence] = field(repr=False, default_factory=dict)


def seed_everything(
    master_seed: int,
    components: Sequence[str],
    *,
    seed_torch: bool = True,
) -> SeedBundle:
    """Create independent generators for ``components`` and (optionally) seed torch.

    Args:
        master_seed: the experiment's single master seed (written to seed.txt).
        components: distinct component names, e.g.
            ``("env", "exploration", "replay", "episode_seeds")``.
        seed_torch: also call ``torch.manual_seed`` with determinism flags.
            Kept optional so numpy-only callers do not import torch.

    Returns:
        A :class:`SeedBundle`; ``bundle.generators[name]`` is that component's
        private generator. No global RNG state is modified.
    """
    names = sorted(components)
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate component names in {list(components)!r}")
    master = np.random.SeedSequence(master_seed)
    children = master.spawn(len(names))
    sequences = dict(zip(names, children))
    generators = {name: np.random.default_rng(seq.spawn(1)[0]) for name, seq in sequences.items()}

    if seed_torch:
        import torch

        torch.manual_seed(master_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return SeedBundle(master_seed=master_seed, generators=generators, sequences=sequences)


def spawn_child(bundle: SeedBundle, component: str) -> np.random.Generator:
    """Spawn a fresh, reproducible child generator from a component's sequence.

    Successive calls for the same component yield a deterministic sequence of
    independent generators (used e.g. for per-episode env seeds; the paired
    estimator replays the same children for benchmark episodes — CRN).
    """
    if component not in bundle.sequences:
        raise KeyError(
            f"unknown component {component!r}; available: {sorted(bundle.sequences)}"
        )
    return np.random.default_rng(bundle.sequences[component].spawn(1)[0])
