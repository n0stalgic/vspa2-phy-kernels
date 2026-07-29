// SPDX-License-Identifier: BSD-3-Clause
// x2_interp kernel test harness: 2x polyphase interpolation FIR.
// fp16 complex input (vspa_complex_float16), half_fixed16 output (cfixed16_t).
// Vectors from gen_vectors.py.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vspa/intrinsics.h>
#include "test_utils.h"
#include "x2_interp.h"

#define N_SAMPLES   64
#define N_TAPS      32
#define N_UP        (N_SAMPLES * 2)
// __ld_vec at iter 1 reads 128 bytes from byte offset 128: 80-element pad is sufficient.
#define N_INPUT_PAD (N_SAMPLES + 16)

// input.hex: one uint32 per complex fp16 sample (re in low 16, im in high 16).
static const uint32_t INPUT_DATA[N_SAMPLES] = {
#include "vectors/input.hex"
};
static const uint32_t TAPS_DATA[N_TAPS] = {
#include "vectors/taps.hex"
};
// ref.hex: half_fixed16 complex, packed as uint32 (re in low 16, im in high 16).
static const uint32_t REF_DATA[N_UP] = {
#include "vectors/ref.hex"
};

_VSPA_VECTOR_ALIGN static vspa_complex_float16 INPUT_BUF[N_INPUT_PAD];
_VSPA_VECTOR_ALIGN static float32_t            TAPS_BUF[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 OUTPUT_BUF[N_UP];

// Impulse test buffers.
_VSPA_VECTOR_ALIGN static vspa_complex_float16 IMP_INPUT[N_INPUT_PAD];
_VSPA_VECTOR_ALIGN static float32_t            IMP_TAPS[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 IMP_OUTPUT[N_UP];

void main(void)
{
    int i;
    x2_interp_state_t state;

    // INPUT_DATA: packed fp16 pairs. Direct word copy; last 16 samples stay zero.
    for (i = 0; i < N_SAMPLES; i++)
        *(uint32_t *)&INPUT_BUF[i] = INPUT_DATA[i];

    for (i = 0; i < N_TAPS; i++)
        *(uint32_t *)&TAPS_BUF[i] = TAPS_DATA[i];

    x2_interp_init(&state, TAPS_BUF, N_SAMPLES);

    // --- IMPULSE TEST ---
    // E0[0]=1, E1[0]=0. Input: fp16 complex (1.0, 0.0) at [0].
    // fp16 1.0 = 0x3C00. 1.0f saturates to 0x7FFF in half_fixed16.
    // Expected: out[0]=0x00007FFF (re=sat(1.0), im=0), out[1]=0x00000000.
    {
        x2_interp_state_t imp_state;

        memset(IMP_INPUT,  0, sizeof(IMP_INPUT));
        memset(IMP_TAPS,   0, sizeof(IMP_TAPS));
        memset(IMP_OUTPUT, 0, sizeof(IMP_OUTPUT));

        // fp16 complex (1.0 + 0j): re=0x3C00, im=0x0000 → packed uint32 = 0x00003C00
        *(uint32_t *)&IMP_INPUT[0] = 0x00003C00U;
        IMP_TAPS[0] = 1.0f;
        IMP_TAPS[1] = 0.0f;

        x2_interp_init(&imp_state, IMP_TAPS, N_SAMPLES);
        x2_interp_process(&imp_state, IMP_INPUT, IMP_OUTPUT);

        printf("IMPULSE TEST (tap[0]=1, input[0]=(fp16 1.0, 0)):\n");
        printf("  out[0]: 0x%08x  (expect 0x00007FFF) %s\n",
               *(uint32_t *)&IMP_OUTPUT[0],
               *(uint32_t *)&IMP_OUTPUT[0] == 0x00007FFFU ? "PASS" : "FAIL");
        printf("  out[1]: 0x%08x  (expect 0x00000000) %s\n",
               *(uint32_t *)&IMP_OUTPUT[1],
               *(uint32_t *)&IMP_OUTPUT[1] == 0x00000000U ? "PASS" : "FAIL");
    }

    KCYC_INIT();
    KCYC_START();
    x2_interp_process(&state, INPUT_BUF, OUTPUT_BUF);
    KCYC_STOP_PRINT();

    // --- DEBUG: first 8 output samples vs reference ---
    {
        const uint32_t *out = (const uint32_t *)OUTPUT_BUF;
        printf("DBG output[0..7] vs ref (half_fixed16 re/im packed as uint32):\n");
        for (i = 0; i < 8; i++)
            printf("  out[%2d]: 0x%08x  ref: 0x%08x %s\n",
                   i, out[i], REF_DATA[i],
                   out[i] == REF_DATA[i] ? "OK" : "MISMATCH");
    }

    vspa_array_cmp_hf16((const unsigned *)OUTPUT_BUF,
                        (const unsigned *)REF_DATA,
                        N_UP);
}
