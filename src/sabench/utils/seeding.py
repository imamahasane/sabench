"""Deterministic seed derivation.

Every random choice in this project traces back to ``config.seed``. Anything that needs its own
independent-looking stream (one per permutation, one per bootstrap resample, one per Monte Carlo
split) derives it with :func:`perm_seed` instead of incrementing a counter by hand, so re-running a
script from a cold start reproduces identical numbers regardless of execution order or parallelism.
"""
from __future__ import annotations

import random

import numpy as np

_MAX_UINT32 = 2**32 - 1


def perm_seed(base_seed: int, index: int) -> int:
    """Derive a reproducible, well-mixed seed for iteration ``index`` of some process seeded by
    ``base_seed`` (a permutation, a bootstrap resample, a CV repeat, ...).

    Uses NumPy's SeedSequence, which is designed exactly for "one base seed, many independent
    streams" and avoids the correlation artifacts of naive ``base_seed + index``.
    """
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    seq = np.random.SeedSequence([base_seed, index])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def set_global_seed(seed: int) -> None:
    """Seed the stdlib ``random`` and NumPy's legacy global RNG.

    Library code in this project should prefer local generators (``np.random.default_rng(perm_seed(...))``
    or an explicit ``random_state=``) over global state; this helper exists for third-party code
    (e.g. a plotting library's jitter) that only reads the global RNG.
    """
    random.seed(seed)
    np.random.seed(seed % _MAX_UINT32)
