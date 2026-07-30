#include <cstdint>
#include "vspa/intrinsics.h"
#include "vcpu.h"
#include "vspa.h"

#define NUM_COEFFICIENTS 32
#define NUM_COMPLEX_SAMPLES 64


typedef struct
{
	_VSPA_VECTOR_ALIGN float E0_coeffs[NUM_COEFFICIENTS];
	_VSPA_VECTOR_ALIGN float E1_coeffs[NUM_COEFFICIENTS];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 E0_conv_buff[NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 E1_conv_buff[NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN vspa_complex_float16 delay_hist[32];
	int32_t num_samples;

} x2_interp_state_t;

void x2_interp_init(x2_interp_state_t * __restrict state, float * __restrict coeffs, int32_t num_samples);

void x2_interp_process(x2_interp_state_t * __restrict state,
                       vspa_complex_float16 * __restrict input,
                       vspa_complex_float16 * __restrict output);

const uint32_t *x2_interp_get_vra_taps(uint32_t *buf);
const uint16_t *x2_interp_get_vra_r0(uint16_t *buf);
const uint32_t *x2_interp_get_vra_r6(uint32_t *buf);
