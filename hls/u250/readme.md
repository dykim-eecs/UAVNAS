g++ -std=c++17 host.cpp /home/rbln/26sep/u250/xcl2.cpp -o host \
-I/home/rbln/26sep/u250 \
-I/tools/Xilinx/Vitis_HLS/2022.2/include \
-I/opt/xilinx/xrt/include \
-L/opt/xilinx/xrt/lib \
-lOpenCL -lxrt_core -lxrt_coreutil -pthread
