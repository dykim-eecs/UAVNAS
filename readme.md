# UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search

Reference implementation for the paper:

> **UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search**
> Doh-Yon Kim *(First Author)* et al.
> Accepted at *IEEE Transactions on Industrial Informatics (TII)*.

UAV-NAS is an end-to-end pipeline that discovers a compact convolutional architecture
for drone (UAV) identification from raw I/Q radio samples and deploys it on a
Xilinx Kria KV260 FPGA.

## Repository Layout

```
.
├── code/   # PyTorch DARTS search and fixed-architecture training (Python)
└── hls/    # Vitis HLS C++ kernel and OpenCL host application (FPGA inference)
```

| Folder | Contents | Stage |
|---|---|---|
| [`code/`](code) | DARTS search, fixed training, ONNX export | Software / GPU |
| [`hls/`](hls) | HLS kernel (`cell0`), CPU stem + cell1, OpenCL host | FPGA / KV260 |

## End-to-End Workflow

```
        +--------------------+        +----------------------+        +--------------------+
        | code/train_search  | -----> | code/infer.py        | -----> | hls/ (Vitis 2025.1)|
        | (DARTS search)     |  geno  | (fixed train + ONNX) | model  | (KV260 inference)  |
        +--------------------+        +----------------------+        +--------------------+
```

1. **Search** the architecture with `code/train_search.py`. The discovered genotype is
   saved to `best_genotype.pkl` (and printed to stdout).
2. **Train and export** the fixed network with `code/infer.py`, producing
   `best_model.pth` and `model.onnx`.
3. **Convert** the trained weights into the C arrays under `hls/kernel/` (the
   `*_weights.cpp` files are auto-generated from the ONNX model).
4. **Build and run** the FPGA kernel and host on a Kria KV260 with Vitis 2025.1.
   See [`hls/readme.md`](hls/readme.md) for build and run instructions.

## Target Hardware

- **Board:** Xilinx Kria KV260
- **FPGA part:** `xck26-sfvc784-2lv-c`
- **Toolchain:** Xilinx Vitis 2025.1, XRT, base platform `xilinx_kv260_base_202510_1`

## Citation

If you use this code, please cite:

```bibtex
@article{kim2026uavnas,
  title   = {UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search},
  author  = {Kim, Doh-Yon and others},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2026}
}
```

## License

Copyright (c) 2026 Doh-Yon Kim (First Author). All rights reserved.
This code is released as the reference implementation associated with the IEEE TII
paper above; see the per-file headers for the full notice.

The files `hls/host/xcl2.cpp` and `hls/host/xcl2.hpp` are taken from the Xilinx Vitis
Accel Examples and remain under their original BSD-3-Clause license.
