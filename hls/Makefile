#!****************************************************************************************!*
#! [UAV-NAS Project] CONFIDENTIAL & PROPRIETARY                                           !*
#! Copyright (c) 2026 Doh-Yon Kim (First Author), All Rights Reserved.                 !*
#!                                                                                        !*
#! NOTICE: This source code is associated with the following academic publication:        !*
#! "UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search"                  !*
#! Accepted for publication in IEEE Transactions on Industrial Informatics (TII).         !*
#!                                                                                        !*
#! All information contained herein is, and remains the property of the authors.          !*
#! The intellectual and technical concepts contained herein are protected by copyright     !*
#! law and international treaties. Unauthorized copying, modification, or distribution    !*
#! of this file, via any medium, is strictly prohibited.                                  !*
#!                                                                                        !*
#!****************************************************************************************!*
#
#* [ PAPER METADATA ]
#  - Title   : UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search
#  - Journal : IEEE Transactions on Industrial Informatics (TII)
#
#* [ AUTHOR INFORMATION ]
#  - First Author : Kim, Doh-Yon
#
#* [ FILE DESCRIPTION ]
#  - Module Name : Makefile
#  - Function    : Vitis v++ build system for HLS kernel synthesis and linking
#  - Platform    : Kria KV260 (xck26-sfvc784-2lv-c)
#

# ===== User Configuration =====
TARGET       ?= hw
PLATFORM     ?= xilinx_kv260_base_202510_1
PLATFORM_REPO_PATH ?= Xilinx/2025.1/Vitis/base_platforms/xilinx_kv260_base_202510_1
PFM          := $(PLATFORM_REPO_PATH)/$(PLATFORM).xpfm

KERNEL_NAME  := cell0
FREQ_HZ      := 100000000

# ===== Build Targets =====
.PHONY: all clean

all: build/$(KERNEL_NAME).xclbin

# Compile kernel source to .xo (HLS synthesis)
build/$(KERNEL_NAME).xo: kernel/$(KERNEL_NAME).cpp kernel/$(KERNEL_NAME)_weights.cpp
	@mkdir -p build
	v++ -c -k $(KERNEL_NAME) \
		-o $@ \
		--platform $(PFM) \
		--target $(TARGET) \
		--hls.clock $(FREQ_HZ):$(KERNEL_NAME) \
		--save-temps \
		--temp_dir build/temp \
		--report_dir build/report \
		--connectivity.nk $(KERNEL_NAME):1:$(KERNEL_NAME)_1 \
		-Ikernel \
		-I. \
		-DUSE_DSP_WGEN \
		$^

# Link .xo into .xclbin (bitstream generation)
build/$(KERNEL_NAME).xclbin: build/$(KERNEL_NAME).xo
	v++ -l \
		-o $@ \
		$< \
		--platform $(PFM) \
		--target $(TARGET) \
		--clock.freqHz $(FREQ_HZ):$(KERNEL_NAME)_1 \
		--save-temps \
		--temp_dir build/temp_link \
		--report_dir build/report_link \
		--connectivity.nk $(KERNEL_NAME):1:$(KERNEL_NAME)_1

# ===== Clean =====
clean:
	rm -rf build tmp* *~ .Xil *.log *.jou *.xml *.json *.html
