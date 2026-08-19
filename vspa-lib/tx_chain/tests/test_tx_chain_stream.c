// SPDX-License-Identifier: BSD-3-Clause
// Steady-state TX chain benchmark: QAM -> interpolation [-> NCO mixer].

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <vspa/intrinsics.h>
#include "test_utils.h"
#include "x2_interp.h"
#include "x4_interp.h"

extern void mod_bpsk_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out,
                        unsigned int N);
extern void mod_qpsk_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out,
                        unsigned int N);
extern void mod_16qam_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out,
                         unsigned int N);

#ifndef TX_BYPASS_MIXER
#define TX_BYPASS_MIXER 0
#endif

#if !TX_BYPASS_MIXER
extern unsigned int mixer_vspa(cfixed16_t *mix_out, cfixed16_t *mix_in,
                               uint32_t phase_in, int32_t freq_in, uint32_t L);
#endif

#define QAM_MODE_BPSK     1
#define QAM_MODE_QPSK     2
#define QAM_MODE_16QAM    4
#define TX_INTERP_X2      2
#define TX_INTERP_X4      4

#ifndef QAM_MODE
#define QAM_MODE QAM_MODE_16QAM
#endif

#ifndef TX_INTERP
#define TX_INTERP TX_INTERP_X4
#endif

#ifndef STREAM_QAM_LINES
#define STREAM_QAM_LINES 8
#endif

#ifndef STREAM_BLOCKS
#define STREAM_BLOCKS 32
#endif

#define N_SYMBOLS (STREAM_QAM_LINES * 32)

#if TX_INTERP == TX_INTERP_X4
#  define N_TAPS        33
#  define INTERP_FACTOR 4
#  define INTERP_STATE  x4_interp_state_t
#  define INTERP_INIT   x4_interp_init
#  define INTERP_PROC   x4_interp_process
#elif TX_INTERP == TX_INTERP_X2
#  define N_TAPS        32
#  define INTERP_FACTOR 2
#  define INTERP_STATE  x2_interp_state_t
#  define INTERP_INIT   x2_interp_init
#  define INTERP_PROC   x2_interp_process
#else
#  error "Unknown TX_INTERP"
#endif

#define N_INTERP  (N_SYMBOLS * INTERP_FACTOR)
#define MIX_LINES (N_INTERP / (__AU_COUNT__ * 2))

#if QAM_MODE == QAM_MODE_BPSK
#  define N_INPUT_WORDS (STREAM_QAM_LINES * 1)
#  define QAM_MOD_FN    mod_bpsk_hw
#elif QAM_MODE == QAM_MODE_QPSK
#  define N_INPUT_WORDS (STREAM_QAM_LINES * 2)
#  define QAM_MOD_FN    mod_qpsk_hw
#elif QAM_MODE == QAM_MODE_16QAM
#  define N_INPUT_WORDS (STREAM_QAM_LINES * 4)
#  define QAM_MOD_FN    mod_16qam_hw
#else
#  error "Unknown QAM_MODE"
#endif

#define PHASE_IN 0x12345678u
#define FREQ_IN  0x00123456

static const unsigned int TAPS_DATA[N_TAPS] = {
#include "vectors/taps.hex"
};

_VSPA_VECTOR_ALIGN static unsigned int bit_in[N_INPUT_WORDS];
_VSPA_VECTOR_ALIGN static float32_t taps[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 qam_out[N_SYMBOLS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 interp_out[N_INTERP];
#if !TX_BYPASS_MIXER
_VSPA_VECTOR_ALIGN static cfixed16_t tx_out[N_INTERP];
#endif

void main(void)
{
    INTERP_STATE interp_state;
    uint32_t phase = PHASE_IN;
    uint32_t prng = 0x6d2b79f5u;
    uint32_t checksum = 0;
    long cycles;
    long total_output_samples;
    int i;
    int block;

    for (i = 0; i < N_INPUT_WORDS; i++) {
        prng ^= prng << 13;
        prng ^= prng >> 17;
        prng ^= prng << 5;
        bit_in[i] = prng;
    }

    for (i = 0; i < N_TAPS; i++)
        *(uint32_t *)&taps[i] = TAPS_DATA[i];

    INTERP_INIT(&interp_state, taps, N_SYMBOLS);

    KCYC_INIT();
    KCYC_START();
    for (block = 0; block < STREAM_BLOCKS; block++) {
        QAM_MOD_FN(bit_in, qam_out, (unsigned int)STREAM_QAM_LINES);
        INTERP_PROC(&interp_state, qam_out, interp_out);
#if !TX_BYPASS_MIXER
        phase = mixer_vspa(tx_out, (cfixed16_t *)interp_out,
                           phase, FREQ_IN, MIX_LINES);
#endif
    }
    KCYC_STOP_PRINT();

    cycles = (long)(_kcyc_end - _kcyc_start - _kcyc_overhead);
    total_output_samples = (long)STREAM_BLOCKS * N_INTERP;

    for (i = 0; i < N_INTERP; i++)
#if TX_BYPASS_MIXER
        checksum ^= ((const uint32_t *)interp_out)[i];
#else
        checksum ^= ((const uint32_t *)tx_out)[i];
#endif

    printf("TX_STREAM: blocks=%d qam_lines_per_block=%d symbols_per_block=%d mixer=%d\n",
           STREAM_BLOCKS, STREAM_QAM_LINES, N_SYMBOLS, !TX_BYPASS_MIXER);
    printf("TX_STREAM: input_bytes=%ld output_samples=%ld factor=%d\n",
           (long)STREAM_BLOCKS * N_INPUT_WORDS * 4,
           total_output_samples, INTERP_FACTOR);
    printf("TX_STREAM: cycles_per_sample_x1000=%ld final_phase=0x%08X checksum=0x%08X\n",
           (cycles * 1000 + total_output_samples / 2) / total_output_samples,
           phase, checksum);
    printf("PASS\n");
}
