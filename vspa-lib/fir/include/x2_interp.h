#include <cstdint>
#include "vspa/intrinsics.h"
#include "vcpu.h"
#include "vspa.h"

#define NUM_TAPS 32
#define NUM_COMPLEX_SAMPLES 32


typedef struct
{
	_VSPA_VECTOR_ALIGN float E0_coeffs[NUM_TAPS];
	_VSPA_VECTOR_ALIGN float E1_coeffs[NUM_TAPS];
	_VSPA_VECTOR_ALIGN cfixed16_t E0_conv_buff[NUM_COMPLEX_SAMPLES];
	_VSPA_VECTOR_ALIGN cfixed16_t E1_conv_buff[NUM_COMPLEX_SAMPLES];
	int32_t num_samples;
	int32_t num_coeffs;

} x2_interp_state_t;

void x2_interp_init(x2_interp_state_t * __restrict state, float * __restrict coeffs, int32_t num_coeffs, int32_t num_samples);

void x2_interp_process(x2_interp_state_t * __restrict state,
                       vspa_complex_float32 * __restrict input,
                       cfixed16_t * __restrict output);

const uint32_t *x2_interp_get_vra_taps(uint32_t *buf);
const uint16_t *x2_interp_get_vra_r0(uint16_t *buf);
const uint32_t *x2_interp_get_vra_r6(uint32_t *buf);
