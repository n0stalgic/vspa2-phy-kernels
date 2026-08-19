#include <cstdint>
#include "vspa/intrinsics.h"
#include "vcpu.h"
#include "vspa.h"

#define X4_NUM_COEFFICIENTS 33
#define X4_PHASE_TAPS 9
#define X4_NUM_COMPLEX_SAMPLES 64

typedef struct
{
	// Four polyphase banks. Each bank is padded to one 16-lane coefficient
	// vector even though only X4_PHASE_TAPS entries are consumed.
	_VSPA_VECTOR_ALIGN float E0_coeffs[16];
	_VSPA_VECTOR_ALIGN float E1_coeffs[16];
	_VSPA_VECTOR_ALIGN float E2_coeffs[16];
	_VSPA_VECTOR_ALIGN float E3_coeffs[16];

	_VSPA_VECTOR_ALIGN vspa_complex_float16 E0_conv_buff[X4_NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 E1_conv_buff[X4_NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 E2_conv_buff[X4_NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 E3_conv_buff[X4_NUM_COMPLEX_SAMPLES];

	_VSPA_VECTOR_ALIGN vspa_complex_float16 delay_hist[32];
	// Pristine pass-one and pass-two register pairs. Pass two is formed by
	// rotating once per block, then reused by all four polyphase branches.
	_VSPA_VECTOR_ALIGN vspa_complex_float16 window_stage[128];
	int32_t num_samples;

} x4_interp_state_t;

void x4_interp_init(x4_interp_state_t * __restrict state, float * __restrict coeffs,
                    int32_t num_samples);

void x4_interp_process(x4_interp_state_t * __restrict state,
                       vspa_complex_float16 * __restrict input,
                       vspa_complex_float16 * __restrict output);
