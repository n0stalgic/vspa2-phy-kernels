#include <cstdint>
#include <stdio.h>
#include "vspa/intrinsics.h"
#include "vcpu.h"
#include "vspa.h"
#include "x4_interp.h"

#pragma optimization_level 3

#define X4_SECOND_PASS_POSITION 48

#define X4_RUN_PHASE(coeff_reg, conv_buff, taps)                                \
	do {                                                                    \
		/* Reload pristine pass-one and pass-two windows for this phase. */\
		__ld_Rx_mem(4, state->window_stage);                               \
		__ld_Rx_mem(5, state->window_stage + 32);                          \
		__ld_Rx_mem(6, state->window_stage + 64);                          \
		__ld_Rx_mem(7, state->window_stage + 96);                          \
		__set_rot(R4R5l2);                                                \
		__set_VRAptr_rS1(_VR4);                                           \
		__set_VRAptr_rS0(coeff_reg);                                      \
		__rd_S0(); __rd_S1(); __rmad(); __rol();                          \
		for (j = 1; j < (taps); j++) {                                   \
			__rd_S0(); __rd_S1(); __rmac(); __rol();                  \
		}                                                                 \
		__wr(straight);                                                   \
		__st_vec(conv_buff);                                              \
		__set_rot(R6R7l2);                                                \
		__set_VRAptr_rS1(_VR6);                                           \
		__set_VRAptr_rS0(coeff_reg);                                      \
		__rd_S0(); __rd_S1(); __rmad(); __rol();                          \
		for (j = 1; j < (taps); j++) {                                   \
			__rd_S0(); __rd_S1(); __rmac(); __rol();                  \
		}                                                                 \
		__wr(straight);                                                   \
		__st_vec((conv_buff) + 32);                                       \
	} while (0)


void x4_interp_init(x4_interp_state_t * __restrict state, float * __restrict coeffs,
                    int32_t num_samples)
{
	int32_t i;

	for (i = 0; i < 16; i++) {
		state->E0_coeffs[i] = 0.0f;
		state->E1_coeffs[i] = 0.0f;
		state->E2_coeffs[i] = 0.0f;
		state->E3_coeffs[i] = 0.0f;
	}

	for (i = 0; i < X4_PHASE_TAPS; i++) {
		int32_t idx = 4 * i;
		if (idx < X4_NUM_COEFFICIENTS)
			state->E0_coeffs[i] = coeffs[idx];
		if (idx + 1 < X4_NUM_COEFFICIENTS)
			state->E1_coeffs[i] = coeffs[idx + 1];
		if (idx + 2 < X4_NUM_COEFFICIENTS)
			state->E2_coeffs[i] = coeffs[idx + 2];
		if (idx + 3 < X4_NUM_COEFFICIENTS)
			state->E3_coeffs[i] = coeffs[idx + 3];
	}

	for (i = 0; i < 32; i++)
		*(uint32_t *)&state->delay_hist[i] = 0;

	state->num_samples = num_samples;
}


static void _fir(x4_interp_state_t * __restrict state,
		vspa_complex_float16 * __restrict input,
		vspa_complex_float16 * __restrict output)
__attribute__ ((noinline))
{
	int32_t i;
	int32_t j;
	int32_t k;

	__clr_VRA();

	//--------------------------------------------------------
	// VCPU data plane set up:
	//
	// hardware precision:
	// S0 : single
	// S1 : half
	// S2 : single
	// AU : single
	// V  : half_fixed
	//
	// rS0 -> VR0 : E0 coefficients, padded to 9 active taps
	//     -> VR1 : E1 coefficients, padded to 9 active taps
	//     -> VR2 : E2 coefficients, padded to 9 active taps
	//     -> VR3 : E3 coefficients, padded to 9 active taps
	// rS1 -> VR4 : input delay line
	//        VR5 : previous input block / delay history
	// rV  -> VR4 : phase filter output scratch after pass-one input is used
	//
	// A 33-tap RRC at sps=4, span=8 splits as 9/8/8/8 taps.
	// The short phases carry an explicit zero in the ninth coefficient slot.
	//
	// Vector Rotation Unit (VRU):
	// R4:R5 left rotation, 2 HW (1 complex fp16) per tap.
	// R4 holds the current 32-sample block and R5 holds the previous block,
	// so rotations bring causal history samples into R4 instead of zeros.
	// Full R4:R5 wrap = 64 rols.
	//
	// The pass-two window is formed once at rotation position 48 and saved
	// alongside the pristine pass-one window. Both are reloaded per phase.
	//--------------------------------------------------------

	__set_prec(single, half, single, single, half_fixed);
	__set_Smode(S0hword, S1straight, S2zeros);
	__set_VRAptr_rS0(_VR0);
	__set_VRAptr_rS1(_VR4);
	// Pass one no longer needs R4 once its MAC completes, so reuse VR4 for
	// the accumulator and stores. This preserves the pass-two R6:R7 window.
	__set_VRAptr_rV(_VR4);
	__set_VRAptr_rSt(4);
	__set_VRAincr_rS0(2);
	__set_VRAincr_rS1(0);
	__set_rot(R4R5l2);

	__ld_Rx_mem(0, state->E0_coeffs);
	__ld_Rx_mem(1, state->E1_coeffs);
	__ld_Rx_mem(2, state->E2_coeffs);
	__ld_Rx_mem(3, state->E3_coeffs);

	// Load delay line history into VR5.
	__ld_vec(state->delay_hist);
	__ld_Rx(normal, 5);

	for (i = 0; i < (state->num_samples / 32); i++)
	{
		__ld_vec(input + i * 32);
		__ld_Rx(normal, 4);

		// Save the pristine pass-one pair.
		__set_VRAptr_rSt(4);
		__st_vec(state->window_stage);
		__set_VRAptr_rSt(5);
		__st_vec(state->window_stage + 32);

		// Form the pass-two pair once, then reuse it for all four phases.
		__set_rot(R4R5l2);
		for (j = 0; j < X4_SECOND_PASS_POSITION; j++)
			__rol();
		__set_VRAptr_rSt(4);
		__st_vec(state->window_stage + 64);
		__set_VRAptr_rSt(5);
		__st_vec(state->window_stage + 96);
		__set_VRAptr_rSt(4);

		X4_RUN_PHASE(_VR0, state->E0_conv_buff, 9);
		X4_RUN_PHASE(_VR1, state->E1_conv_buff, 8);
		X4_RUN_PHASE(_VR2, state->E2_conv_buff, 8);
		X4_RUN_PHASE(_VR3, state->E3_conv_buff, 8);

		// Update VR5 (aka history) with the current block for the next block.
		__ld_vec(input + i * 32);
		__ld_Rx(normal, 5);

		// Interleave the four polyphase outputs:
		// y[4*k + phase] = Ephase[k].
		#pragma loop_count (16, 16, 1, 0)
		for (k = 0; k < 16; k++) {
			output[i * 128 + 4*k]     = state->E0_conv_buff[k];
			output[i * 128 + 4*k + 1] = state->E1_conv_buff[k];
			output[i * 128 + 4*k + 2] = state->E2_conv_buff[k];
			output[i * 128 + 4*k + 3] = state->E3_conv_buff[k];
		}
		#pragma loop_count (16, 16, 1, 0)
		for (k = 0; k < 16; k++) {
			output[i * 128 + 64 + 4*k]     = state->E0_conv_buff[32 + k];
			output[i * 128 + 64 + 4*k + 1] = state->E1_conv_buff[32 + k];
			output[i * 128 + 64 + 4*k + 2] = state->E2_conv_buff[32 + k];
			output[i * 128 + 64 + 4*k + 3] = state->E3_conv_buff[32 + k];
		}
	}

	// VR5 holds the last input block. Store it directly to delay_hist for
	// the next call.
	__set_VRAptr_rSt(5);
	__st_vec(state->delay_hist);
}


void x4_interp_process(x4_interp_state_t *state,
                       vspa_complex_float16 *input,
                       vspa_complex_float16 *output)
{
	_fir(state, input, output);
}

#pragma optimization_level reset
