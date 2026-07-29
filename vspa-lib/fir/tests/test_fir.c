// SPDX-License-Identifier: BSD-3-Clause
// x2_interp kernel test harness: 2x polyphase interpolation FIR.
// float32 input, half_fixed16 output (cfixed16_t). Vectors from gen_vectors.py.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vspa/intrinsics.h>
#include "test_utils.h"
#include "x2_interp.h"

#define N_SAMPLES  64
#define N_TAPS     32
#define N_UP       (N_SAMPLES * 2)

static const uint32_t INPUT_DATA[N_SAMPLES * 2] = {
#include "vectors/input.hex"
};
static const uint32_t TAPS_DATA[N_TAPS] = {
#include "vectors/taps.hex"
};
// ref.hex: half_fixed16 complex, packed as uint32 (re in low 16, im in high 16).
static const uint32_t REF_DATA[N_UP] = {
#include "vectors/ref.hex"
};

_VSPA_VECTOR_ALIGN static vspa_complex_float32 INPUT_BUF[N_SAMPLES];
_VSPA_VECTOR_ALIGN static float32_t            TAPS_BUF[N_TAPS];
_VSPA_VECTOR_ALIGN static cfixed16_t           OUTPUT_BUF[N_UP];

// Impulse test buffers.
_VSPA_VECTOR_ALIGN static vspa_complex_float32 IMP_INPUT[N_SAMPLES];
_VSPA_VECTOR_ALIGN static float32_t            IMP_TAPS[N_TAPS];
_VSPA_VECTOR_ALIGN static cfixed16_t           IMP_OUTPUT[N_UP];

void main(void)
{
    int i;
    x2_interp_state_t state;

    for (i = 0; i < N_SAMPLES; i++) {
        memcpy(&INPUT_BUF[i].real, &INPUT_DATA[2 * i],     sizeof(float));
        memcpy(&INPUT_BUF[i].imag, &INPUT_DATA[2 * i + 1], sizeof(float));
    }
    for (i = 0; i < N_TAPS; i++)
        *(uint32_t *)&TAPS_BUF[i] = TAPS_DATA[i];

    x2_interp_init(&state, TAPS_BUF, N_TAPS, N_SAMPLES);

    // --- IMPULSE TEST ---
    // h=[1,0]: E0[0]=1, E1[0]=0. Input impulse at [0]=(1,0).
    // float32 1.0 saturates to 0x7fff in half_fixed16.
    // Expected: out[0]=0x00007fff (re=sat(1.0), im=0), out[1]=0x00000000.
    {
        x2_interp_state_t imp_state;

        memset(IMP_INPUT,  0, sizeof(IMP_INPUT));
        memset(IMP_TAPS,   0, sizeof(IMP_TAPS));
        memset(IMP_OUTPUT, 0, sizeof(IMP_OUTPUT));

        IMP_INPUT[0].real = 1.0f;
        IMP_INPUT[0].imag = 0.0f;
        IMP_TAPS[0] = 1.0f;
        IMP_TAPS[1] = 0.0f;

        x2_interp_init(&imp_state, IMP_TAPS, 2, N_SAMPLES);
        x2_interp_process(&imp_state, IMP_INPUT, IMP_OUTPUT);

        printf("IMPULSE TEST (tap[0]=1, input[0]=(1,0)):\n");
        printf("  out[0]: 0x%08x  (expect 0x00007fff) %s\n",
               *(uint32_t *)&IMP_OUTPUT[0],
               *(uint32_t *)&IMP_OUTPUT[0] == 0x00007fffU ? "PASS" : "FAIL");
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
