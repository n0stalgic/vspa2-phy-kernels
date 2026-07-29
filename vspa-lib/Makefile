# SPDX-License-Identifier: BSD-3-Clause
# Standalone build for upsample.c
#
#   make            - build

PROJ_ROOT := $(abspath ..)

VSPA_TOOL ?= /opt/VSPA_Tools_vbeta_14_00_781
AU        ?= 16
ARCH       = vspa2
CORE       = sp
KEEP      ?=

VSPA_CC      = $(VSPA_TOOL)/bin/fsvspacc
VSPA_SDK_INC = $(PROJ_ROOT)/vspa-sdk/inc

BUILD_DIR ?= build
TARGET     = $(BUILD_DIR)/upsample_$(ARCH)_$(AU)au.eld

VCFLAGS = \
    -arch $(ARCH) \
    -au_count $(AU) \
    -core_type $(CORE) \
    -ansi off \
    -O3  \
    --keep \
    -opt=noalias_by_type \
    -mvcpu \
    -msgstyle gcc \
    -env "$(VSPA_TOOL)" \
    -isystem $(VSPA_TOOL)/include \
    -I$(VSPA_SDK_INC) \
    -Xlnk "-Map $(BUILD_DIR)/upsample.map"


.PHONY: all clean

all: $(TARGET)

$(TARGET): upsample.c
	@mkdir -p $(BUILD_DIR)
	$(VSPA_CC) $(VCFLAGS) \
	    -mem $(VSPA_TOOL)/etc/$(ARCH)_$(AU)au/lcf/$(ARCH)_$(AU)au_$(CORE).lcf \
	    $(VSPA_TOOL)/lib/$(ARCH)/startup_au$(AU)_$(CORE).eln \
	    upsample.c \
	    $(VSPA_TOOL)/lib/$(ARCH)/libc_au$(AU)_$(CORE).elb \
	    $(VSPA_TOOL)/lib/$(ARCH)/librt_au$(AU)_$(CORE).elb \
	    -o $@
	@echo "Built: $@"

clean:
	rm -rf $(BUILD_DIR)
