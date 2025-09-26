# ==== User Config ====
TARGET ?= hw
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
PLATFORM_REPO_PATH ?= /data/isplab/andrew
PFM := $(PLATFORM_REPO_PATH)/$(PLATFORM)/$(PLATFORM).xpfm

KERNEL_NAME := cell0
KERNELS := 1
# 100 MHz
FREQ_HZ_LIST := 100000000

# ==== Targets ====
.PHONY: all clean
all: $(foreach K,$(KERNELS),$(foreach F,$(FREQ_HZ_LIST),build_k$(K)_f$(F)))

define MAKE_KERNEL_RULE
build_k$(1)_f$(2):
	@echo "========================================================"
	@echo "==> Building Kernels=$(1), FREQ=$(2)Hz"
	@echo "========================================================"
	@mkdir -p build/k$(1)_f$(2)
	@# Compile to .xo
	v++ -c -k $(KERNEL_NAME) \
		-o build/k$(1)_f$(2)/$(KERNEL_NAME).xo \
		--platform $(PFM) \
		--target $(TARGET) \
		--hls.clock $(2):$(KERNEL_NAME) \
		--save-temps \
		--temp_dir build/k$(1)_f$(2)/temp \
		--report_dir build/k$(1)_f$(2)/report \
		--config config/multi_$(1).ini \
		-Ikernel \
		-I. \
		-DUSE_DSP_WGEN \
		$(KERNEL_NAME).cpp \
		$(KERNEL_NAME)_weights.cpp

	@# Link to .xclbin (Fixed CU name for instance)
	v++ -l \
		-o build/k$(1)_f$(2)/$(KERNEL_NAME).xclbin \
		build/k$(1)_f$(2)/$(KERNEL_NAME).xo \
		--platform $(PFM) \
		--target $(TARGET) \
		--clock.freqHz $(2):$(KERNEL_NAME)_1 \
		--save-temps \
		--temp_dir build/k$(1)_f$(2)/temp_link \
		--report_dir build/k$(1)_f$(2)/report_link \
		--config config/multi_$(1).ini
endef

# Expand rules for each kernel/frequency combination
$(foreach K,$(KERNELS),$(foreach F,$(FREQ_HZ_LIST),$(eval $(call MAKE_KERNEL_RULE,$(K),$(F)))))

# ==== Clean ====
clean:
	rm -rf build tmp* *~ .Xil *.log *.jou *.xml *.json *.html
