#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""TX chain vector generator and scalar oracle.

Pipeline:
    bits -> ld.qam model -> polyphase interpolation model -> mixer model

The generated ref.hex is the final half-fixed mixer output, packed as one
uint32 per complex sample: low 16 bits real, high 16 bits imag.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_VSPA_LIB = _TESTS_DIR.parents[1]
_COMMON_PY = _VSPA_LIB / 'common' / 'python'
_FIR_PY = _VSPA_LIB / 'fir' / 'python'
_PSK_HW_PY = _VSPA_LIB / 'psk_hw' / 'python'
_MIXER_PY = _VSPA_LIB / 'mixer' / 'python'

for p in (str(_COMMON_PY), str(_FIR_PY), str(_PSK_HW_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rrc import rrc_taps, rrc_taps_n
from utils.hex_io import write_hex_u32
from utils.packing import complex_to_u32_sm16, u32_to_complex_sm16
from vspa.arith import r_half

import model as qam_model

_mixer_spec = importlib.util.spec_from_file_location('tx_chain_mixer_model', _MIXER_PY / 'model.py')
if _mixer_spec is None or _mixer_spec.loader is None:
    raise RuntimeError('failed to load mixer model')
_mixer_model = importlib.util.module_from_spec(_mixer_spec)
_mixer_spec.loader.exec_module(_mixer_model)
r_mixer = _mixer_model.r_mixer

_libm = ctypes.CDLL('libm.so.6')
_libm.fmaf.restype = ctypes.c_float
_libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]


TOKEN_TO_MODE = {
    'BPSK': 'bpsk',
    'QPSK': 'qpsk',
    '16QAM': '16qam',
}

OUTDIR = _TESTS_DIR / 'vectors'
QAM_LINES = 2
N_SYMBOLS = QAM_LINES * 32
BETA = 0.35
X2_SPS = 2
X2_TAPS = 32
X4_SPS = 4
X4_SPAN = 8
X4_TAPS = X4_SPS * X4_SPAN + 1
PHASE_IN = 0x12345678
FREQ_IN = 0x00123456
SEED_BASE = 20260729


def _fmaf(a: float, b: float, c: float) -> np.float32:
    return np.float32(_libm.fmaf(ctypes.c_float(a), ctypes.c_float(b), ctypes.c_float(c)))


def _u32_to_complex_f16(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint32)
    re_u16 = (words & np.uint32(0xFFFF)).astype(np.uint16)
    im_u16 = (words >> np.uint32(16)).astype(np.uint16)
    re = re_u16.view(np.float16).astype(np.float32)
    im = im_u16.view(np.float16).astype(np.float32)
    return (re + 1j * im).astype(np.complex64)


def _unpack_bits_lsb_first(words: np.ndarray, n_bits: int) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint32).reshape(-1)
    bits = np.zeros(words.size * 32, dtype=np.uint8)
    for b in range(32):
        bits[b::32] = ((words >> np.uint32(b)) & np.uint32(1)).astype(np.uint8)
    return bits[:n_bits]


def _polyphase_x2(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    taps = np.asarray(taps, dtype=np.float32)
    e0 = taps[0::2]
    e1 = taps[1::2]
    y = np.empty(2 * len(x), dtype=np.complex64)

    for k in range(len(x)):
        n_taps = min(k + 1, len(e0))

        acc_re = np.float32(e0[0]) * np.float32(x[k].real)
        acc_im = np.float32(e0[0]) * np.float32(x[k].imag)
        for j in range(1, n_taps):
            acc_re = _fmaf(e0[j], x[k - j].real, acc_re)
            acc_im = _fmaf(e0[j], x[k - j].imag, acc_im)
        y[2 * k] = acc_re + 1j * acc_im

        acc_re = np.float32(e1[0]) * np.float32(x[k].real)
        acc_im = np.float32(e1[0]) * np.float32(x[k].imag)
        for j in range(1, n_taps):
            acc_re = _fmaf(e1[j], x[k - j].real, acc_re)
            acc_im = _fmaf(e1[j], x[k - j].imag, acc_im)
        y[2 * k + 1] = acc_re + 1j * acc_im

    return r_half(y).astype(np.complex128)


def _polyphase_x4(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    taps = np.asarray(taps, dtype=np.float32)
    phases = []

    for p in range(X4_SPS):
        phase = np.zeros(9, dtype=np.float32)
        raw = taps[p::X4_SPS]
        phase[:len(raw)] = raw
        phases.append(phase)

    y = np.empty(X4_SPS * len(x), dtype=np.complex64)
    for k in range(len(x)):
        for p, phase in enumerate(phases):
            n_taps = min(k + 1, len(phase))
            acc_re = np.float32(phase[0]) * np.float32(x[k].real)
            acc_im = np.float32(phase[0]) * np.float32(x[k].imag)
            for j in range(1, n_taps):
                acc_re = _fmaf(phase[j], x[k - j].real, acc_re)
                acc_im = _fmaf(phase[j], x[k - j].imag, acc_im)
            y[X4_SPS * k + p] = acc_re + 1j * acc_im

    return r_half(y).astype(np.complex128)


def _interp_fir(x: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex64)
    taps = np.asarray(taps, dtype=np.float32)
    x_up = np.zeros(len(x) * sps, dtype=np.complex64)
    x_up[0::sps] = x

    y = np.empty_like(x_up)
    for n in range(len(x_up)):
        acc_re = np.float32(0.0)
        acc_im = np.float32(0.0)
        for k in range(min(n + 1, len(taps))):
            acc_re = _fmaf(taps[k], x_up[n - k].real, acc_re)
            acc_im = _fmaf(taps[k], x_up[n - k].imag, acc_im)
        y[n] = acc_re + 1j * acc_im
    return y.astype(np.complex128)


def _write_matlab(path: Path, mode_token: str, bits: np.ndarray,
                  qam: np.ndarray, interp: np.ndarray, diagnostic4: np.ndarray,
                  mixed: np.ndarray, interp_mode: str, hw_sps: int) -> None:
    def row_complex(name: str, values: np.ndarray) -> str:
        vals = [
            f'{float(v.real): .9g}{float(v.imag):+.9g}i'
            for v in np.asarray(values).reshape(-1)
        ]
        return f'{name} = [{", ".join(vals)}];\n'

    m_bits = qam_model.QAM_MODES[TOKEN_TO_MODE[mode_token]]['M']
    bit_values = _unpack_bits_lsb_first(bits, len(qam) * m_bits)
    nrz_values = (2 * bit_values.astype(np.int16)) - 1

    with path.open('w') as handle:
        handle.write('% Auto-generated by gen_vectors.py. Do not edit by hand.\n')
        handle.write(f"mode = '{mode_token}';\n")
        handle.write(f"interp_mode = '{interp_mode}';\n")
        handle.write(f'bits_per_symbol = {m_bits};\n')
        handle.write(f'hw_sps = {hw_sps};\n')
        handle.write(f'plot_sps = {X4_SPS};\n')
        handle.write(f'plot_span = {X4_SPAN};\n')
        handle.write(f'rrc_beta = {BETA};\n')
        handle.write(f'phase_in = uint32({PHASE_IN});\n')
        handle.write(f'freq_in = int32({FREQ_IN});\n')
        handle.write('bits_u32 = uint32([')
        handle.write(', '.join(f"hex2dec('{int(v):08X}')" for v in bits))
        handle.write(']);\n')
        handle.write('bit_values = [')
        handle.write(', '.join(str(int(v)) for v in bit_values))
        handle.write('];\n')
        handle.write('nrz_values = [')
        handle.write(', '.join(str(int(v)) for v in nrz_values))
        handle.write('];\n')
        handle.write(row_complex('qam_symbols', qam))
        handle.write(row_complex('interp_out', interp))
        handle.write(row_complex('diagnostic4_out', diagnostic4))
        handle.write(row_complex('tx_out', mixed))
        handle.write('n_bits = 0:numel(nrz_values)-1;\n')
        handle.write('n_sym = 0:numel(qam_symbols)-1;\n')
        handle.write('n_interp = 0:numel(interp_out)-1;\n')
        handle.write('t_interp = n_interp / hw_sps;\n')
        handle.write('n_diag4 = 0:numel(diagnostic4_out)-1;\n')
        handle.write('t_diag4 = n_diag4 / plot_sps;\n')
        handle.write('n_tx = 0:numel(tx_out)-1;\n')
        handle.write('\n')
        handle.write("figure('Name', ['TX chain overview, ', mode]);\n")
        handle.write('tiledlayout(5, 1);\n')
        handle.write("nexttile; stairs(n_bits, nrz_values, 'LineWidth', 1); grid on;\n")
        handle.write("ylim([-1.25 1.25]); xlabel('bit index'); ylabel('NRZ'); title('Input NRZ bit stream');\n")
        handle.write("nexttile; stem(n_sym, real(qam_symbols), '.-'); hold on; stem(n_sym, imag(qam_symbols), '.-'); hold off; grid on;\n")
        handle.write("xlabel('symbol index'); ylabel('amplitude'); legend('I', 'Q'); title('Mapped symbols');\n")
        handle.write("nexttile; plot(t_interp, real(interp_out), '.-', t_interp, imag(interp_out), '.-'); grid on;\n")
        handle.write("xlabel('symbol index'); ylabel('amplitude'); legend('I', 'Q'); title(['Hardware ', interp_mode, ' pulse-shaped baseband output']);\n")
        handle.write("nexttile; plot(t_diag4, real(diagnostic4_out), '-', t_diag4, imag(diagnostic4_out), '-'); grid on;\n")
        handle.write("xlabel('symbol index'); ylabel('amplitude'); legend('I', 'Q'); title('Diagnostic 4-sps RRC output, span=8, beta=0.35');\n")
        handle.write("nexttile; plot(n_tx, real(tx_out), '.-', n_tx, imag(tx_out), '.-'); grid on;\n")
        handle.write("xlabel('sample index'); ylabel('amplitude'); legend('I', 'Q'); title('Mixed TX output');\n")
        handle.write('\n')
        handle.write("figure('Name', ['Constellation, ', mode]);\n")
        handle.write("subplot(1, 2, 1); plot(real(qam_symbols), imag(qam_symbols), 'o'); grid on; axis equal;\n")
        handle.write("xlabel('I'); ylabel('Q'); title('Symbol constellation');\n")
        handle.write("subplot(1, 2, 2); plot(real(interp_out), imag(interp_out), '.-'); grid on; axis equal;\n")
        handle.write("xlabel('I'); ylabel('Q'); title('Pulse-shaped trajectory');\n")
        handle.write('\n')
        handle.write("figure('Name', ['Pulse-shaping comparison, ', mode]);\n")
        handle.write("subplot(2, 1, 1); plot(t_interp, real(interp_out), '.-', t_interp, imag(interp_out), '.-'); grid on;\n")
        handle.write("legend('I', 'Q'); xlabel('symbol index'); ylabel('amplitude'); title(['Hardware ', interp_mode, ' output samples']);\n")
        handle.write("subplot(2, 1, 2); plot(t_diag4, real(diagnostic4_out), t_diag4, imag(diagnostic4_out)); grid on;\n")
        handle.write("legend('I', 'Q'); xlabel('symbol index'); ylabel('amplitude'); title('4-sps scalar RRC reference');\n")
        handle.write('\n')
        handle.write("figure('Name', ['Final TX IQ, ', mode]);\n")
        handle.write("subplot(2, 1, 1); plot(n_tx, real(tx_out), '.-', n_tx, imag(tx_out), '.-'); grid on;\n")
        handle.write("legend('real', 'imag'); title(['TX chain output, ', mode]); xlabel('sample index');\n")
        handle.write("subplot(2, 1, 2); plot(real(tx_out), imag(tx_out), '.-'); grid on;\n")
        handle.write("axis equal; xlabel('I'); ylabel('Q'); title('Mixed output IQ trajectory');\n")


def main() -> None:
    token = os.environ.get('QAM_MODE', 'QPSK').upper()
    if token not in TOKEN_TO_MODE:
        raise SystemExit(f'unknown QAM_MODE={token!r}, expected one of {sorted(TOKEN_TO_MODE)}')
    interp_mode = os.environ.get('INTERP', 'x2').lower()
    if interp_mode not in ('x2', 'x4'):
        raise SystemExit("unknown INTERP={!r}, expected 'x2' or 'x4'".format(interp_mode))

    mode = TOKEN_TO_MODE[token]
    m_bits = qam_model.QAM_MODES[mode]['M']
    n_input_words = (N_SYMBOLS * m_bits) // 32

    seed = SEED_BASE + sum(ord(c) for c in token)
    rng = np.random.default_rng(seed=seed)
    bits_u32 = rng.integers(0, 1 << 32, size=n_input_words, dtype=np.uint64).astype(np.uint32)

    qam_words = qam_model.r_qam_mod(bits_u32, mode)[:N_SYMBOLS]
    qam = _u32_to_complex_f16(qam_words)

    if interp_mode == 'x4':
        taps = rrc_taps(BETA, X4_SPS, X4_SPAN).astype(np.float32)
        expected_taps = X4_TAPS
        hw_sps = X4_SPS
        interp = _polyphase_x4(qam, taps)
    else:
        taps = rrc_taps_n(BETA, X2_SPS, X2_TAPS).astype(np.float32)
        expected_taps = X2_TAPS
        hw_sps = X2_SPS
        interp = _polyphase_x2(qam, taps)

    if len(taps) != expected_taps:
        raise SystemExit(f'expected {expected_taps} taps, got {len(taps)}')

    diagnostic4 = _interp_fir(qam, rrc_taps(BETA, X4_SPS, X4_SPAN), X4_SPS)
    mixed = r_mixer(u32_to_complex_sm16(complex_to_u32_sm16(interp)), PHASE_IN, FREQ_IN)
    ref_words = complex_to_u32_sm16(mixed)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_hex_u32(bits_u32, str(OUTDIR / 'input.hex'))
    write_hex_u32(taps.view(np.uint32), str(OUTDIR / 'taps.hex'))
    write_hex_u32(qam_words, str(OUTDIR / 'qam_ref.hex'))
    write_hex_u32(complex_to_u32_sm16(interp), str(OUTDIR / 'interp_ref.hex'))
    write_hex_u32(ref_words, str(OUTDIR / 'ref.hex'))
    _write_matlab(OUTDIR / 'plot_tx_chain.m', token, bits_u32, qam, interp, diagnostic4,
                  mixed, interp_mode, hw_sps)

    print(f'Generated tx_chain vectors: mode={token} qam_lines={QAM_LINES} '
          f'interp={interp_mode} symbols={N_SYMBOLS} out={hw_sps * N_SYMBOLS} '
          f'input_words={n_input_words}')


if __name__ == '__main__':
    main()
