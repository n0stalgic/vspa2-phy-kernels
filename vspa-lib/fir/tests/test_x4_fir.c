// SPDX-License-Identifier: BSD-3-Clause
// x4_interp kernel test harness: 4x polyphase interpolation FIR.
// fp16 complex input (vspa_complex_float16), half_fixed16 output (cfixed16_t).
// Vectors from gen_vectors_x4.py.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vspa/intrinsics.h>
#include "test_utils.h"
#include "x4_interp.h"

#define N_SAMPLES   64
#define N_TAPS      33
#define N_UP        (N_SAMPLES * 4)
#define N_INPUT_PAD (N_SAMPLES + 16)

static const uint32_t INPUT_DATA[N_SAMPLES] = {
#include "vectors/input.hex"
};
static const uint32_t TAPS_DATA[N_TAPS] = {
#include "vectors/taps.hex"
};
static const uint32_t REF_DATA[N_UP] = {
#include "vectors/ref.hex"
};

_VSPA_VECTOR_ALIGN static vspa_complex_float16 INPUT_BUF[N_INPUT_PAD];
_VSPA_VECTOR_ALIGN static float32_t            TAPS_BUF[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 OUTPUT_BUF[N_UP];

_VSPA_VECTOR_ALIGN static vspa_complex_float16 IMP_INPUT[N_INPUT_PAD];
_VSPA_VECTOR_ALIGN static float32_t            IMP_TAPS[N_TAPS];
_VSPA_VECTOR_ALIGN static vspa_complex_float16 IMP_OUTPUT[N_UP];

void main(void)
{
    int i;
    x4_interp_state_t state;

    for (i = 0; i < N_SAMPLES; i++)
        *(uint32_t *)&INPUT_BUF[i] = INPUT_DATA[i];

    for (i = 0; i < N_TAPS; i++)
        *(uint32_t *)&TAPS_BUF[i] = TAPS_DATA[i];

    x4_interp_init(&state, TAPS_BUF, N_SAMPLES);

    // --- IMPULSE TEST ---
    // E0[0]=1 and E1/E2/E3[0]=0. Input: fp16 complex (1.0, 0.0) at [0].
    // Expected: out[0]=1.0 saturated to half_fixed16, out[1..3]=0.
    {
        x4_interp_state_t imp_state;

        memset(IMP_INPUT,  0, sizeof(IMP_INPUT));
        memset(IMP_TAPS,   0, sizeof(IMP_TAPS));
        memset(IMP_OUTPUT, 0, sizeof(IMP_OUTPUT));

        *(uint32_t *)&IMP_INPUT[0] = 0x00003C00U;
        IMP_TAPS[0] = 1.0f;

        x4_interp_init(&imp_state, IMP_TAPS, N_SAMPLES);
        x4_interp_process(&imp_state, IMP_INPUT, IMP_OUTPUT);

        printf("IMPULSE TEST (tap[0]=1, input[0]=(fp16 1.0, 0)):\n");
        printf("  out[0]: 0x%08x  (expect 0x00007FFF) %s\n",
               *(uint32_t *)&IMP_OUTPUT[0],
               *(uint32_t *)&IMP_OUTPUT[0] == 0x00007FFFU ? "PASS" : "FAIL");
        printf("  out[1]: 0x%08x  (expect 0x00000000) %s\n",
               *(uint32_t *)&IMP_OUTPUT[1],
               *(uint32_t *)&IMP_OUTPUT[1] == 0x00000000U ? "PASS" : "FAIL");
        printf("  out[2]: 0x%08x  (expect 0x00000000) %s\n",
               *(uint32_t *)&IMP_OUTPUT[2],
               *(uint32_t *)&IMP_OUTPUT[2] == 0x00000000U ? "PASS" : "FAIL");
        printf("  out[3]: 0x%08x  (expect 0x00000000) %s\n",
               *(uint32_t *)&IMP_OUTPUT[3],
               *(uint32_t *)&IMP_OUTPUT[3] == 0x00000000U ? "PASS" : "FAIL");
    }

    KCYC_INIT();
    KCYC_START();
    x4_interp_process(&state, INPUT_BUF, OUTPUT_BUF);
    KCYC_STOP_PRINT();

    {
        const uint32_t *out = (const uint32_t *)OUTPUT_BUF;
        printf("DBG output[0..15] vs ref (half_fixed16 re/im packed as uint32):\n");
        for (i = 0; i < 16; i++)
            printf("  out[%2d]: 0x%08x  ref: 0x%08x %s\n",
                   i, out[i], REF_DATA[i],
                   out[i] == REF_DATA[i] ? "OK" : "MISMATCH");
    }

    vspa_array_cmp_hf16((const unsigned *)OUTPUT_BUF,
                        (const unsigned *)REF_DATA,
                        N_UP);
}
