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
                    int32_t num_coeffs, int32_t num_samples)
{
	ASSERT((num_coeffs % 2) == 0);
	
	for (int32_t i = 0; i < num_coeffs / 2; i++) 
	{
	    state->E0_coeffs[i] = coeffs[2 * i];
	    state->E1_coeffs[i] = coeffs[2 * i + 1];
	}	

	state->num_coeffs     = num_coeffs;
	state->num_samples  = num_samples;

}


static void _fir(x2_interp_state_t * __restrict state,
		vspa_complex_float32 * __restrict input,
		cfixed16_t * __restrict output)
__attribute__ ((noinline))
{
	
	__clr_VRA();
	
	//--------------------------------------------------------
	// VCPU data plane set up:
	// 
	// hardware precision:
	// S0 : single
	// S1 : single
	// S2 : single
	// AU : single
	// V  : half_fixed
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
	// R4:R5  left rotation, 4 HW (1 complex float32) per tap
	//
	//--------------------------------------------------------

	__set_prec(single, single, single, single, half_fixed);
	__set_Smode(S0hword, S1straight, S2zeros);
	__set_VRAptr_rS0(_VR0);
	__set_VRAptr_rS1(_VR4);
	__set_VRAptr_rV(_VR6);
	__set_VRAptr_rSt(6);
	__set_VRAincr_rS0(2);
	__set_VRAincr_rS1(0);
	__set_rot(R4R5l4);

	__ld_Rx_mem(0, state->E0_coeffs);
	__ld_Rx_mem(1, state->E1_coeffs);

	for (int32_t i = 0; i < (state->num_samples / 16); i++)
	{
		// load new input block into VR4 (positions 0..15 of R2R3 delay line)
		__ld_vec(input + i * 16);
		__ld_Rx(normal, 4);

		// E0 pass: accumulate into VR6
		__rd_S0();
		__rd_S1();
		__rmad();
		__rol();

		for (int32_t j = 1; j < state->num_coeffs / 2; j++)
		{
			__rd_S0();
			__rd_S1();
			__rmac();
			__rol();
		}

		__wr(straight);
		__st_vec(state->E0_conv_buff);

		// switch to E1 coefficients and accumulator
		__set_VRAptr_rS0(_VR1);
		__set_VRAptr_rV(_VR7);
		__set_VRAptr_rSt(7);

		// advance 16 to restore current block to VR4 for E1 pass 
		for (int32_t j = 0; j < state->num_coeffs / 2; j++)
			__rol();

		// E1 pass: accumulate into VR7
		__rd_S0();
		__rd_S1();
		__rmad();
		__rol();

		for (int32_t j = 1; j < state->num_coeffs / 2; j++)
		{
			__rd_S0();
			__rd_S1();
			__rmac();
			__rol();
		}

		__wr(straight);
		__st_vec(state->E1_conv_buff);

		// reset for next block: S0 back to E0 coefficients, rV back to VR6
		// S1 stays at VR4 (delay line never moves)
		__set_VRAptr_rS0(_VR0);
		__set_VRAptr_rV(_VR6);
		__set_VRAptr_rSt(6);
		
		// interleave E0/E1 for this block
		for (int32_t k = 0; k < 16; k++)
		{
			output[i * 32 + 2*k]     = state->E0_conv_buff[k];
			output[i * 32 + 2*k + 1] = state->E1_conv_buff[k];
		}
	}
}

void x2_interp_process(x2_interp_state_t *state,
                       vspa_complex_float32 *input,
                       cfixed16_t *output)
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
