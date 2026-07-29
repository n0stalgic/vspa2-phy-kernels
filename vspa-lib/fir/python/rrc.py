# SPDX-License-Identifier: BSD-3-Clause
"""Root-raised-cosine tap generator.

Standard normalized (T=1 symbol) RRC impulse response:

    h(t) = [sin(pi(1-b)t) + 4bt*cos(pi(1+b)t)] / [pi*t*(1-(4bt)^2)]

with two removable (0/0) singularities that a naive evaluation turns into
NaN: t=0 and t=+-1/(4b). Both are handled below with their L'Hopital limits
rather than left to divide by zero -- this is *the* classic RRC generator
bug, so don't simplify this back down to the bare formula.

    h(0)         = 1 - b + 4b/pi
    h(+-1/(4b))  = (b/sqrt(2)) * [(1+2/pi)sin(pi/(4b)) + (1-2/pi)cos(pi/(4b))]

Verified (scratch, not re-checked here) by finite-difference convergence of
the general formula toward both closed-form limits, plus symmetry/no-NaN
checks, for beta=0.35 and for a forced case (beta=0.25, sps=4) where the
second singularity actually lands exactly on the sample grid.
"""

from __future__ import annotations

import numpy as np

_SING_EPS = 1e-9


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Real RRC taps, length L = span*sps + 1, unit-energy normalized.

    beta: rolloff factor, (0, 1].
    sps:  samples per symbol (oversampling factor).
    span: filter half-span in symbols on each side of the center tap.

    Unit-energy (sum(h**2) == 1) rather than peak-normalized: the standard
    convention for a pulse-shaping filter, and it lands the peak tap well
    inside Q15 range (~0.55 for beta=0.35/sps=4/span=8) with no need for an
    arbitrary headroom fudge factor.
    """
    length = span * sps + 1
    n = np.arange(length) - (length - 1) // 2
    t = n / sps

    at_zero = np.abs(t) < _SING_EPS
    at_sing = np.abs(np.abs(t) - 1.0 / (4 * beta)) < _SING_EPS
    general = ~(at_zero | at_sing)

    h = np.empty(length, dtype=np.float64)
    h[at_zero] = 1 - beta + 4 * beta / np.pi
    h[at_sing] = (beta / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
    )
    tg = t[general]
    h[general] = (
        np.sin(np.pi * (1 - beta) * tg) + 4 * beta * tg * np.cos(np.pi * (1 + beta) * tg)
    ) / (np.pi * tg * (1 - (4 * beta * tg) ** 2))

    return h / np.sqrt(np.sum(h ** 2))
