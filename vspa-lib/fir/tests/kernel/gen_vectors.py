#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""x2_interp vector generator.

Input:  N_SAMPLES complex float32 symbols.
Oracle: 2x zero-stuff upsample -> 32-tap float32 RRC FIR -> complex float32 output.

Accumulator stays in float32 across all taps (mirrors rmac chain + wr.straight).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_X2_DIR   = Path(__file__).resolve().parent
_FIR_DIR  = _X2_DIR.parent.parent          # vspa-lib/fir/
_COMM_PY  = _FIR_DIR.parent / 'common' / 'python'
_FIR_PY   = _FIR_DIR / 'python'

for p in (str(_COMM_PY), str(_FIR_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rrc import rrc_taps_n
from utils.hex_io import write_hex_u32

OUTDIR    = _X2_DIR / 'vectors'
N_SAMPLES = 64    # original symbols; must be multiple of 16
N_TAPS    = 32    # float32 taps; 16 taps per x2 polyphase branch
BETA      = 0.35
SPS       = 2


def _write_complex_f32(c: np.ndarray, path: str) -> None:
    """Write complex array as interleaved IEEE float32 (uint32) hex."""
    re = np.asarray(c).real.astype(np.float32).view(np.uint32)
    im = np.asarray(c).imag.astype(np.float32).view(np.uint32)
    buf = np.empty(2 * len(re), dtype=np.uint32)
    buf[0::2] = re
    buf[1::2] = im
    write_hex_u32(buf, path)


def _fir_f32(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """y[n] = sum_k h[k]*x[n-k], zero history, float32 accumulation per sample.

    Mirrors rmac chain (S0=single, S1=half, V=single) then wr.straight
    (St=single): accumulate and store in float32.
    """
    h = np.asarray(h, dtype=np.float32)
    x = np.asarray(x, dtype=np.complex64)
    N, L = len(x), len(h)
    y = np.empty(N, dtype=np.complex64)
    for n in range(N):
        acc_r = np.float32(0.0)
        acc_i = np.float32(0.0)
        for k in range(L):
            if n >= k:
                t = h[k]
                acc_r = np.float32(acc_r + t * np.float32(x[n - k].real))
                acc_i = np.float32(acc_i + t * np.float32(x[n - k].imag))
        # store=single: no float16 rounding.
        y[n] = acc_r + 1j * acc_i
    return y


def main() -> None:
    rng = np.random.default_rng(42)
    x_raw = (rng.uniform(-0.5, 0.5, N_SAMPLES)
             + 1j * rng.uniform(-0.5, 0.5, N_SAMPLES))
    x = x_raw.astype(np.complex64)

    taps_raw = rrc_taps_n(BETA, SPS, N_TAPS)
    assert len(taps_raw) == N_TAPS, f'expected {N_TAPS} taps, got {len(taps_raw)}'
    taps = taps_raw.astype(np.float32)

    # 2x upsample: even indices get input samples, odd indices are zero.
    x_up = np.zeros(2 * N_SAMPLES, dtype=np.complex64)
    x_up[0::2] = x

    y = _fir_f32(x_up, taps)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    _write_complex_f32(x,    str(OUTDIR / 'input.hex'))
    write_hex_u32(taps.view(np.uint32), str(OUTDIR / 'taps.hex'))
    _write_complex_f32(y,    str(OUTDIR / 'ref.hex'))

    print(f'Generated x2_interp vectors: '
          f'N_SAMPLES={N_SAMPLES}, N_UP={2*N_SAMPLES}, N_TAPS={N_TAPS}')


if __name__ == '__main__':
    main()
