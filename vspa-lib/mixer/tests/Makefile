# SPDX-License-Identifier: BSD-3-Clause
# mixer kernel test (NCO + cmad frequency shift) -- see
# ../../common/kernel.mk for the shared build/test/clean rules.
#
# Two implementations share this test harness and vectors, both exporting
# the same C-callable symbol (mixer_vspa):
#   make test          - kernel/mixer.c, the correctness-proven C+intrinsics baseline
#   make test IMPL=asm - kernel/mix.sx,  the hand-written VLIW asm version

KERNEL = mixer
IMPL   ?= c

ifeq ($(IMPL),asm)
KERNEL := mixer_asm
SRCS    = test_mixer.c kernel/mix.sx
else
SRCS    = test_mixer.c kernel/mixer.c
endif

include ../../common/kernel.mk
