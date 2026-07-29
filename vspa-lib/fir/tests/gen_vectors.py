#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""x2_interp vector generator: 2x upsample + 32-tap fp16-input RRC FIR.

Input:  N_SAMPLES complex fp16 symbols (from QAM modulator).
Oracle: direct polyphase E0/E1 accumulation (mirrors kernel rmad+rmac chain).
        Inputs quantized to fp16 (S1prec=half), taps remain float32 (S0prec=single).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

# fmaf: float32 fused multiply-add, matching VSPA2 rmac semantics.
# rmac: acc = fma(S0, S1, acc) — one rounding, no intermediate float32 product.
_libm = ctypes.CDLL('libm.so.6')
_libm.fmaf.restype  = ctypes.c_float
_libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]

def _fmaf(a: float, b: float, c: float) -> np.float32:
    return np.float32(_libm.fmaf(ctypes.c_float(a), ctypes.c_float(b), ctypes.c_float(c)))

_TESTS_DIR = Path(__file__).resolve().parent
_FIR_DIR   = _TESTS_DIR.parent
_COMM_PY   = _FIR_DIR.parent / 'common' / 'python'
_FIR_PY    = _FIR_DIR / 'python'

for p in (str(_COMM_PY), str(_FIR_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rrc import rrc_taps
from utils.hex_io import write_hex_u32

OUTDIR    = _TESTS_DIR / 'vectors'
N_SAMPLES = 64
N_TAPS    = 32
BETA      = 0.35
SPS       = 1
SPAN      = 31


def _write_complex_f16(c: np.ndarray, path: str) -> None:
    """Pack complex fp16 as one uint32 per sample: re in low 16, im in high 16."""
    c = np.asarray(c, dtype=np.complex64)
    re_u16 = c.real.astype(np.float16).view(np.uint16).astype(np.uint32)
    im_u16 = c.imag.astype(np.float16).view(np.uint16).astype(np.uint32)
    write_hex_u32(re_u16 | (im_u16 << 16), path)


def _to_hf16(v: np.float32) -> int:
    """float32 → VSPA half_fixed16 (sign-magnitude, truncation toward zero)."""
    v = float(v)
    sign = 1 if v < 0 else 0
    frac = min(int(abs(v) * 32768), 32767)
    return (sign << 15) | frac


def _write_complex_fixed16(c: np.ndarray, path: str) -> None:
    """half_fixed16 complex → packed uint32 (re in low 16 bits, im in high 16 bits)."""
    samples = np.asarray(c, dtype=np.complex64)
    buf = np.empty(len(samples), dtype=np.uint32)
    for i, s in enumerate(samples):
        re = _to_hf16(np.float32(s.real))
        im = _to_hf16(np.float32(s.imag))
        buf[i] = np.uint32(re) | (np.uint32(im) << 16)
    write_hex_u32(buf, path)


def _polyphase_f32(x: np.ndarray, E0: np.ndarray, E1: np.ndarray) -> np.ndarray:
    """Polyphase E0/E1 oracle matching the kernel's rmad+rmac chain.

    j=0  → rmad: acc  = float32(tap * x)          (plain multiply, no prior acc)
    j>=1 → rmac: acc  = fmaf(tap, x, acc)          (fused, one rounding)

    x must already be quantized to fp16 (caller's responsibility) since
    S1prec=half widens each delay-line element to float32 before the MAC.
    """
    E0 = np.asarray(E0, dtype=np.float32)
    E1 = np.asarray(E1, dtype=np.float32)
    x  = np.asarray(x, dtype=np.complex64)
    N, P = len(x), len(E0)
    y = np.empty(2 * N, dtype=np.complex64)
    for k in range(N):
        taps = min(k + 1, P)
        # E0 — rmad then rmac chain
        acc_re = np.float32(E0[0]) * np.float32(x[k].real)
        acc_im = np.float32(E0[0]) * np.float32(x[k].imag)
        for j in range(1, taps):
            acc_re = _fmaf(E0[j], x[k - j].real, acc_re)
            acc_im = _fmaf(E0[j], x[k - j].imag, acc_im)
        y[2 * k] = acc_re + 1j * acc_im
        # E1 — rmad then rmac chain
        acc_re = np.float32(E1[0]) * np.float32(x[k].real)
        acc_im = np.float32(E1[0]) * np.float32(x[k].imag)
        for j in range(1, taps):
            acc_re = _fmaf(E1[j], x[k - j].real, acc_re)
            acc_im = _fmaf(E1[j], x[k - j].imag, acc_im)
        y[2 * k + 1] = acc_re + 1j * acc_im
    return y


def main() -> None:
    rng = np.random.default_rng(42)
    x_raw = (rng.uniform(-0.5, 0.5, N_SAMPLES)
             + 1j * rng.uniform(-0.5, 0.5, N_SAMPLES)).astype(np.complex64)

    # Quantize to fp16: mirrors what S1prec=half loads from the delay line.
    x = (x_raw.real.astype(np.float16).astype(np.float32)
         + 1j * x_raw.imag.astype(np.float16).astype(np.float32)).astype(np.complex64)

    taps_raw = rrc_taps(BETA, SPS, SPAN)
    assert len(taps_raw) == N_TAPS, f'expected {N_TAPS} taps, got {len(taps_raw)}'
    taps = taps_raw.astype(np.float32)

    E0 = taps[0::2]  # h[0], h[2], ..., h[30]
    E1 = taps[1::2]  # h[1], h[3], ..., h[31]

    y = _polyphase_f32(x, E0, E1)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    _write_complex_f16(x_raw, str(OUTDIR / 'input.hex'))
    write_hex_u32(taps.view(np.uint32), str(OUTDIR / 'taps.hex'))
    _write_complex_fixed16(y, str(OUTDIR / 'ref.hex'))

    print(f'Generated fir vectors: N_SAMPLES={N_SAMPLES}, N_UP={2*N_SAMPLES}, N_TAPS={N_TAPS}')


if __name__ == '__main__':
    main()
