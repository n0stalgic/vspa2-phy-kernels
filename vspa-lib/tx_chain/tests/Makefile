# SPDX-License-Identifier: BSD-3-Clause
# TX chain test: hardware QAM modulator -> polyphase interpolator -> mixer.

QAM_MODE ?= QPSK
INTERP   ?= x2
STREAM   ?= 0
export QAM_MODE
export INTERP

KERNEL = tx_chain_$(QAM_MODE)_$(INTERP)

ifeq ($(STREAM),1)
KERNEL := $(KERNEL)_stream
HARNESS = test_tx_chain_stream.c
else
HARNESS = test_tx_chain.c
endif

ifeq ($(INTERP),x4)
INTERP_SRC = ../../fir/tests/kernel/x4_interp.c
INTERP_DEF = -DTX_INTERP=TX_INTERP_X4
else
INTERP_SRC = ../../fir/tests/kernel/x2_interp.c
INTERP_DEF = -DTX_INTERP=TX_INTERP_X2
endif

SRCS   = $(HARNESS) \
         ../../psk_hw/tests/kernel/psk_hw.c \
         $(INTERP_SRC) \
         ../../mixer/tests/kernel/mixer.c

DEFS = -DQAM_MODE=QAM_MODE_$(QAM_MODE) \
       $(INTERP_DEF) \
       -I$(abspath ../../fir/include)

include ../../common/kernel.mk

.PHONY: stream stream-test

stream:
	$(MAKE) STREAM=1 all

stream-test:
	$(MAKE) STREAM=1 test
