# SPDX-License-Identifier: BSD-3-Clause
"""Python model for the fir kernel (fir_vspa): direct-form FIR, real taps
applied to complex data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_COMMON = Path(__file__).resolve().parents[2] / 'common' / 'python'
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from vspa.arith import r_half, r_smad


def r_fir_real(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """y[n] = sum_k taps[k] * x[n-k], zero-valued history before x[0].

    taps[k] pairs with x[n-k]: taps[0] is the most-recent-sample tap,
    taps[L-1] the oldest. Accumulates via a chain of r_smad calls (the same
    single-precision-truncating multiply-add r_mixer's oracle uses for its
    one cmac), staying in that precision across all L taps and only
    dropping to half_fixed on the final store -- mirrors an `rmac` chain
    feeding a single `wr.straight` at the end, not a round-per-tap.
    """
    x = np.asarray(x, dtype=np.complex128)
    h = np.asarray(taps, dtype=np.float64)
    n_taps = len(h)
    n_samples = len(x)

    xpad = np.concatenate([np.zeros(n_taps - 1, dtype=np.complex128), x])
    y = np.empty(n_samples, dtype=np.complex128)
    for n in range(n_samples):
        acc = 0.0 + 0.0j
        for k in range(n_taps):
            acc = r_smad(h[k], xpad[n + n_taps - 1 - k], acc)
        y[n] = acc

    return r_half(y)
