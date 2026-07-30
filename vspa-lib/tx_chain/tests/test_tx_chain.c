// SPDX-License-Identifier: BSD-3-Clause
// TX chain harness: ld.qam hardware modulator -> polyphase interpolator
// -> NCO mixer. The Python generator owns the scalar oracle and MATLAB export.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <vspa/intrinsics.h>
#include "test_utils.h"
#include "x2_interp.h"
#include "x4_interp.h"

extern void mod_bpsk_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out, unsigned int N);
extern void mod_qpsk_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out, unsigned int N);
extern void mod_16qam_hw(unsigned int *bit_in, vspa_complex_float16 *qam_out, unsigned int N);

extern unsigned int mixer_vspa(cfixed16_t *mix_out, cfixed16_t *mix_in,
                               uint32_t phase_in, int32_t freq_in, uint32_t L);

#define QAM_MODE_BPSK     1
#define QAM_MODE_QPSK     2
#define QAM_MODE_16QAM    4
#define TX_INTERP_X2      2
#define TX_INTERP_X4      4

#ifndef QAM_MODE
#define QAM_MODE QAM_MODE_QPSK
#endif

#ifndef TX_INTERP
#define TX_INTERP TX_INTERP_X2
#endif

#define QAM_LINES       2
#define N_SYMBOLS       (QAM_LINES * 32)

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

#define N_INTERP        (N_SYMBOLS * INTERP_FACTOR)
#define MIX_LINES       (N_INTERP / (__AU_COUNT__ * 2))

#define PHASE_IN        0x12345678u
#define FREQ_IN         0x00123456

#if QAM_MODE == QAM_MODE_BPSK
#  define N_INPUT_WORDS (QAM_LINES * 1)
#  define QAM_MOD_FN    mod_bpsk_hw
#elif QAM_MODE == QAM_MODE_QPSK
#  define N_INPUT_WORDS (QAM_LINES * 2)
#  define QAM_MOD_FN    mod_qpsk_hw
#elif QAM_MODE == QAM_MODE_16QAM
#  define N_INPUT_WORDS (QAM_LINES * 4)
#  define QAM_MOD_FN    mod_16qam_hw
#else
#  error "Unknown QAM_MODE"
#endif

static const unsigned int INPUT_DATA[N_INPUT_WORDS] = {
#include "vectors/input.hex"
};

static const unsigned int TAPS_DATA[N_TAPS] = {
#include "vectors/taps.hex"
};

static const unsigned int REF_DATA[N_INTERP] = {
#include "vectors/ref.hex"
};

_VSPA_VECTOR_ALIGN static unsigned int bit_in[N_INPUT_WORDS];
_VSPA_VECTOR_ALIGN static float32_t taps[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 qam_out[N_SYMBOLS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 interp_out[N_INTERP];
_VSPA_VECTOR_ALIGN static cfixed16_t tx_out[N_INTERP];

void main(void)
{
    int i;
    INTERP_STATE interp_state;

    for (i = 0; i < N_INPUT_WORDS; i++)
        bit_in[i] = INPUT_DATA[i];

    for (i = 0; i < N_TAPS; i++)
        *(uint32_t *)&taps[i] = TAPS_DATA[i];

    INTERP_INIT(&interp_state, taps, N_SYMBOLS);

    KCYC_INIT();
    KCYC_START();
    QAM_MOD_FN(bit_in, qam_out, (unsigned int)QAM_LINES);
    INTERP_PROC(&interp_state, qam_out, interp_out);
    (void)mixer_vspa(tx_out, (cfixed16_t *)interp_out,
                     PHASE_IN, FREQ_IN, MIX_LINES);
    KCYC_STOP_PRINT();

    printf("TX_CHAIN: qam_lines=%d symbols=%d interp=%d factor=%d mix_lines=%d\n",
           QAM_LINES, N_SYMBOLS, N_INTERP, INTERP_FACTOR, MIX_LINES);

    vspa_array_cmp_hf16((const unsigned *)tx_out,
                        (const unsigned *)REF_DATA,
                        N_INTERP);
}
