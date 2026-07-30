#include <cstdint>
#include <stdio.h>
#include "vspa/intrinsics.h"
#include "vcpu.h"
#include "vspa.h"
#include "x2_interp.h"

// Debug dump buffers (used by x2_interp_get_vra_* diagnostic APIs).
_VSPA_VECTOR_ALIGN static uint32_t _dbg_vr4[32];
_VSPA_VECTOR_ALIGN static uint32_t _dbg_vr0[16];
_VSPA_VECTOR_ALIGN static uint32_t _dbg_vr6[32];

#pragma optimization_level 3


void x2_interp_init(x2_interp_state_t * __restrict state, float * __restrict coeffs,
                    int32_t num_samples)
{
	for (int32_t i = 0; i < NUM_COEFFICIENTS / 2; i++) {
		state->E0_coeffs[i] = coeffs[2 * i];
		state->E1_coeffs[i] = coeffs[2 * i + 1];
	}

	for (int32_t i = 0; i < 32; i++)
		*(uint32_t *)&state->delay_hist[i] = 0;

	state->num_samples = num_samples;

}


static void _fir(x2_interp_state_t * __restrict state,
		vspa_complex_float16 * __restrict input,
		vspa_complex_float16 * __restrict output)
__attribute__ ((noinline))
{
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
	// FP16 from QAM hardware symbol mapper
	// Vector writeback to DMEM is __fx16 for AXIQ to DAC
	//
	// rS0 -> VR0 : E0 coefficients  , incr 2 HW
	//     -> VR1 : E1 coefficients  , incr 2 HW
	// rS1 -> VR4 : input delay line , incr 0 half-words
	// rV  -> VR6 : E0 filter output
	//        VR7 : E1 filter output
	//
	// S0mode hword   : broadcast one tap scalar to all AUs
	// S1mode straight: distribute delay line samples across AUs
	//
	// Vector Rotation Unit (VRU):
	// R4:R5 left rotation, 2 HW (1 complex fp16) per tap.
	// R4 holds the current 32-sample block and R5 holds the previous block,
	// so rotations bring causal history samples into R4 instead of zeros.
	// Full R4:R5 wrap = 64 rols.
	// Each polyphase bank runs two 16-tap passes per 32-sample block:
	//   pass 1 (k=0..15) : 16 rols -> store -> advance 32 rols
	//   pass 2 (k=16..31): 16 rols -> store
	//   total: 64 rols = full wrap; delay line returns to 0, no restore needed
	//--------------------------------------------------------

	__set_prec(single, half, single, single, half_fixed);
	__set_Smode(S0hword, S1straight, S2zeros);
	__set_VRAptr_rS0(_VR0);
	__set_VRAptr_rS1(_VR4);
	__set_VRAptr_rV(_VR6);
	__set_VRAptr_rSt(6);
	__set_VRAincr_rS0(2);
	__set_VRAincr_rS1(0);
	__set_rot(R4R5l2);

	__ld_Rx_mem(0, state->E0_coeffs);
	__ld_Rx_mem(1, state->E1_coeffs);

	// Load delay line history into VR5
	__ld_vec(state->delay_hist);
	__ld_Rx(normal, 5);

	for (int32_t i = 0; i < (state->num_samples / 32); i++)
	{
		__ld_vec(input + i * 32);
		__ld_Rx(normal, 4);

		// E0 pass 1: k=0..15
		__rd_S0(); __rd_S1(); __rmad(); __rol();
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t j = 1; j < NUM_COEFFICIENTS / 2; j++) {
			__rd_S0(); __rd_S1(); __rmac(); __rol();
		}
		__wr(straight);
		__st_vec(state->E0_conv_buff);

		// advance delay line to x[k+16] window (32 rols)
		#pragma loop_count (32, 32, 1, 0)
		for (int32_t j = 0; j < 32; j++) __rol();

		// E0 pass 2: k=16..31
		__set_VRAptr_rS0(_VR0);
		__rd_S0(); __rd_S1(); __rmad(); __rol();
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t j = 1; j < NUM_COEFFICIENTS / 2; j++) {
			__rd_S0(); __rd_S1(); __rmac(); __rol();
		}
		__wr(straight);
		__st_vec(state->E0_conv_buff + 32);
		// delay line at position 0 (16 + 32 + 16 = 64 = full wrap)

		// switch to E1
		__set_VRAptr_rS0(_VR1);
		__set_VRAptr_rV(_VR7);
		__set_VRAptr_rSt(7);

		// E1 pass 1: k=0..15
		__rd_S0(); __rd_S1(); __rmad(); __rol();
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t j = 1; j < NUM_COEFFICIENTS / 2; j++) {
			__rd_S0(); __rd_S1(); __rmac(); __rol();
		}
		__wr(straight);
		__st_vec(state->E1_conv_buff);

		// advance delay line to x[k+16] window (32 rols)
		#pragma loop_count (32, 32, 1, 0)
		for (int32_t j = 0; j < 32; j++) __rol();

		// E1 pass 2: k=16..31
		__set_VRAptr_rS0(_VR1);
		__rd_S0(); __rd_S1(); __rmad(); __rol();
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t j = 1; j < NUM_COEFFICIENTS / 2; j++) {
			__rd_S0(); __rd_S1(); __rmac(); __rol();
		}
		__wr(straight);
		__st_vec(state->E1_conv_buff + 32);
		// delay line at position 0 (64 + 64 = 128 = 2 × full wrap)

		// reset for E0 next block
		__set_VRAptr_rS0(_VR0);
		__set_VRAptr_rV(_VR6);
		__set_VRAptr_rSt(6);

		// Update VR5 (aka history) with current block 
		__ld_vec(input + i * 32);
		__ld_Rx(normal, 5);

		// would eventually like to use the IPPU hardware for
		// vector interleaving, but this will do for now.

		// interleave pass 1 outputs (k=0..15)
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t k = 0; k < 16; k++) {
			output[i * 64 + 2*k]     = state->E0_conv_buff[k];
			output[i * 64 + 2*k + 1] = state->E1_conv_buff[k];
		}
		// interleave pass 2 outputs (k=16..31)
		#pragma loop_count (16, 16, 1, 0)
		for (int32_t k = 0; k < 16; k++) {
			output[i * 64 + 32 + 2*k]     = state->E0_conv_buff[32 + k];
			output[i * 64 + 32 + 2*k + 1] = state->E1_conv_buff[32 + k];
		}
	}

	// VR5 holds the last input block
	// Store it directly to delay_hist for the next call.
	__set_VRAptr_rSt(5);
	__st_vec(state->delay_hist);
	__set_VRAptr_rSt(6);
}

void x2_interp_process(x2_interp_state_t *state,
                       vspa_complex_float16 *input,
                       vspa_complex_float16 *output)
{
	_fir(state, input, output);
}

const uint32_t *x2_interp_get_vra_taps(uint32_t *buf)
{
	int i;
	for (i = 0; i < 32; i++)
		buf[i] = _dbg_vr4[i];
	return buf;
}

const uint16_t *x2_interp_get_vra_r0(uint16_t *buf)
{
	int i;
	for (i = 0; i < 16; i++)
		buf[i] = _dbg_vr0[i];
	return buf;
}

const uint32_t *x2_interp_get_vra_r6(uint32_t *buf)
{
	int i;
	for (i = 0; i < 32; i++)
		buf[i] = _dbg_vr6[i];
	return buf;
}

#pragma optimization_level reset
