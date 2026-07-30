# SPDX-License-Identifier: BSD-3-Clause
# Polyphase interpolation FIR tests.
#
# Default:
#   make test
#
# x4:
#   make INTERP=x4 test

INTERP ?= x2

ifeq ($(INTERP),x4)
KERNEL      = x4_interp
SRCS        = test_x4_fir.c kernel/x4_interp.c
GEN_VECTORS = gen_vectors_x4.py
else
KERNEL      = x2_interp
SRCS        = test_fir.c kernel/x2_interp.c
endif

include ../../common/kernel.mk

.PHONY: x2 x4
x2:
	$(MAKE) INTERP=x2 test

x4:
	$(MAKE) INTERP=x4 test
