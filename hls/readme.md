# UAV-NAS HLS Inference Accelerator

FPGA-accelerated neural network inference for UAV (drone) identification using IQ signals.
Target board: **Kria KV260** (FPGA part: `xck26-sfvc784-2lv-c`).

## Prerequisites

- Xilinx Vitis 2025.1
- Xilinx Runtime (XRT)
- KV260 base platform (`xilinx_kv260_base_202510_1`)

## Build Kernel

Edit `PLATFORM_REPO_PATH` in the Makefile to match your local Vitis installation, then:

```bash
make
```

This synthesizes `cell0.cpp` into `build/cell0.xclbin`.

## Build Host Application

```bash
cd host
g++ -std=c++17 host.cpp ../host/xcl2.cpp \
    -o host \
    -I../kernel \
    -I$(XILINX_VITIS)/include \
    -I$(XILINX_XRT)/include \
    -L$(XILINX_XRT)/lib \
    -lOpenCL -lxrt_core -lxrt_coreutil -pthread
```

## Run

```bash
cd host
./host
```

The host application loads `cell0.xclbin`, executes Cell 0 inference on the FPGA,
runs Cell 1 on the CPU, and prints the classification result.
