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

import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["SeedBundle", "seed_everything", "spawn_child"]

_configured_torch_interop_threads: int | None = None


def _positive_thread_count_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _configure_torch_threads(torch) -> None:
    """Honor the multiseed launcher's explicit per-worker CPU budget."""

    global _configured_torch_interop_threads
    intra = _positive_thread_count_from_env("LMM_TORCH_INTRAOP_THREADS")
    inter = _positive_thread_count_from_env("LMM_TORCH_INTEROP_THREADS")
    if intra is not None:
        torch.set_num_threads(intra)
    if inter is None:
        return
    if _configured_torch_interop_threads is None:
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as exc:
            raise RuntimeError(
                "LMM_TORCH_INTEROP_THREADS must be applied before Torch starts "
                "parallel work"
            ) from exc
        _configured_torch_interop_threads = inter
    elif _configured_torch_interop_threads != inter:
        raise RuntimeError(
            "cannot change Torch inter-op threads within one process: "
            f"{_configured_torch_interop_threads} -> {inter}"
        )


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

        _configure_torch_threads(torch)
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
