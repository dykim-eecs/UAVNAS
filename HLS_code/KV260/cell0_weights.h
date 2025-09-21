#pragma once
#include "model.h"

// 모든 가중치 배열의 요소 수 계산 (data_t 기준, float 또는 ap_fixed)
constexpr int SIZE_NODE_0_0_OP1_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_316 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_317 = 8;                   // 8
constexpr int SIZE_NODE_0_1_OP1_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_319 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_320 = 8;                   // 8
constexpr int SIZE_NODE_0_1_OP5_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_322 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_323 = 8;                   // 8
constexpr int SIZE_NODE_1_0_CONV1_WEIGHT = 4 * 8 * 1 * 1; // 32
constexpr int SIZE_NODE_1_0_CONV2_WEIGHT = 4 * 8 * 1 * 1; // 32
constexpr int SIZE_NODE_1_0_BN_WEIGHT = 8;              // 8
constexpr int SIZE_NODE_1_0_BN_BIAS = 8;                // 8
constexpr int SIZE_NODE_1_0_BN_MEAN = 8;                // 8
constexpr int SIZE_NODE_1_0_BN_VAR = 8;                 // 8
constexpr int SIZE_NODE_1_1_OP1_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_325 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_326 = 8;                   // 8
constexpr int SIZE_NODE_1_1_OP5_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_328 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_329 = 8;                   // 8
constexpr int SIZE_NODE_2_0_OP1_WEIGHT = 8 * 1 * 3 * 3;  // 72
constexpr int SIZE_ONNX_CONV_331 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_332 = 8;                   // 8
constexpr int SIZE_NODE_2_1_OP1_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_334 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_335 = 8;                   // 8
constexpr int SIZE_NODE_2_1_OP5_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_337 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_338 = 8;                   // 8
constexpr int SIZE_NODE_3_0_OP1_WEIGHT = 8 * 1 * 5 * 5;  // 200
constexpr int SIZE_ONNX_CONV_340 = 8 * 8 * 1 * 1;       // 64
constexpr int SIZE_ONNX_CONV_341 = 8;                   // 8
constexpr int SIZE_NODE_3_1_CONV1_WEIGHT = 4 * 8 * 1 * 1; // 32
constexpr int SIZE_NODE_3_1_CONV2_WEIGHT = 4 * 8 * 1 * 1; // 32
constexpr int SIZE_NODE_3_1_BN_WEIGHT = 8;              // 8
constexpr int SIZE_NODE_3_1_BN_BIAS = 8;                // 8
constexpr int SIZE_NODE_3_1_BN_MEAN = 8;                // 8
constexpr int SIZE_NODE_3_1_BN_VAR = 8;                 // 8

// 오프셋 정의: 각 배열의 시작 위치
constexpr int OFFSET_NODE_0_0_OP1_WEIGHT = 0;
constexpr int OFFSET_ONNX_CONV_316 = OFFSET_NODE_0_0_OP1_WEIGHT + SIZE_NODE_0_0_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_317 = OFFSET_ONNX_CONV_316 + SIZE_ONNX_CONV_316;
constexpr int OFFSET_NODE_0_1_OP1_WEIGHT = OFFSET_ONNX_CONV_317 + SIZE_ONNX_CONV_317;
constexpr int OFFSET_ONNX_CONV_319 = OFFSET_NODE_0_1_OP1_WEIGHT + SIZE_NODE_0_1_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_320 = OFFSET_ONNX_CONV_319 + SIZE_ONNX_CONV_319;
constexpr int OFFSET_NODE_0_1_OP5_WEIGHT = OFFSET_ONNX_CONV_320 + SIZE_ONNX_CONV_320;
constexpr int OFFSET_ONNX_CONV_322 = OFFSET_NODE_0_1_OP5_WEIGHT + SIZE_NODE_0_1_OP5_WEIGHT;
constexpr int OFFSET_ONNX_CONV_323 = OFFSET_ONNX_CONV_322 + SIZE_ONNX_CONV_322;
constexpr int OFFSET_NODE_1_0_CONV1_WEIGHT = OFFSET_ONNX_CONV_323 + SIZE_ONNX_CONV_323;
constexpr int OFFSET_NODE_1_0_CONV2_WEIGHT = OFFSET_NODE_1_0_CONV1_WEIGHT + SIZE_NODE_1_0_CONV1_WEIGHT;
constexpr int OFFSET_NODE_1_0_BN_WEIGHT = OFFSET_NODE_1_0_CONV2_WEIGHT + SIZE_NODE_1_0_CONV2_WEIGHT;
constexpr int OFFSET_NODE_1_0_BN_BIAS = OFFSET_NODE_1_0_BN_WEIGHT + SIZE_NODE_1_0_BN_WEIGHT;
constexpr int OFFSET_NODE_1_0_BN_MEAN = OFFSET_NODE_1_0_BN_BIAS + SIZE_NODE_1_0_BN_BIAS;
constexpr int OFFSET_NODE_1_0_BN_VAR = OFFSET_NODE_1_0_BN_MEAN + SIZE_NODE_1_0_BN_MEAN;
constexpr int OFFSET_NODE_1_1_OP1_WEIGHT = OFFSET_NODE_1_0_BN_VAR + SIZE_NODE_1_0_BN_VAR;
constexpr int OFFSET_ONNX_CONV_325 = OFFSET_NODE_1_1_OP1_WEIGHT + SIZE_NODE_1_1_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_326 = OFFSET_ONNX_CONV_325 + SIZE_ONNX_CONV_325;
constexpr int OFFSET_NODE_1_1_OP5_WEIGHT = OFFSET_ONNX_CONV_326 + SIZE_ONNX_CONV_326;
constexpr int OFFSET_ONNX_CONV_328 = OFFSET_NODE_1_1_OP5_WEIGHT + SIZE_NODE_1_1_OP5_WEIGHT;
constexpr int OFFSET_ONNX_CONV_329 = OFFSET_ONNX_CONV_328 + SIZE_ONNX_CONV_328;
constexpr int OFFSET_NODE_2_0_OP1_WEIGHT = OFFSET_ONNX_CONV_329 + SIZE_ONNX_CONV_329;
constexpr int OFFSET_ONNX_CONV_331 = OFFSET_NODE_2_0_OP1_WEIGHT + SIZE_NODE_2_0_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_332 = OFFSET_ONNX_CONV_331 + SIZE_ONNX_CONV_331;
constexpr int OFFSET_NODE_2_1_OP1_WEIGHT = OFFSET_ONNX_CONV_332 + SIZE_ONNX_CONV_332;
constexpr int OFFSET_ONNX_CONV_334 = OFFSET_NODE_2_1_OP1_WEIGHT + SIZE_NODE_2_1_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_335 = OFFSET_ONNX_CONV_334 + SIZE_ONNX_CONV_334;
constexpr int OFFSET_NODE_2_1_OP5_WEIGHT = OFFSET_ONNX_CONV_335 + SIZE_ONNX_CONV_335;
constexpr int OFFSET_ONNX_CONV_337 = OFFSET_NODE_2_1_OP5_WEIGHT + SIZE_NODE_2_1_OP5_WEIGHT;
constexpr int OFFSET_ONNX_CONV_338 = OFFSET_ONNX_CONV_337 + SIZE_ONNX_CONV_337;
constexpr int OFFSET_NODE_3_0_OP1_WEIGHT = OFFSET_ONNX_CONV_338 + SIZE_ONNX_CONV_338;
constexpr int OFFSET_ONNX_CONV_340 = OFFSET_NODE_3_0_OP1_WEIGHT + SIZE_NODE_3_0_OP1_WEIGHT;
constexpr int OFFSET_ONNX_CONV_341 = OFFSET_ONNX_CONV_340 + SIZE_ONNX_CONV_340;
constexpr int OFFSET_NODE_3_1_CONV1_WEIGHT = OFFSET_ONNX_CONV_341 + SIZE_ONNX_CONV_341;
constexpr int OFFSET_NODE_3_1_CONV2_WEIGHT = OFFSET_NODE_3_1_CONV1_WEIGHT + SIZE_NODE_3_1_CONV1_WEIGHT;
constexpr int OFFSET_NODE_3_1_BN_WEIGHT = OFFSET_NODE_3_1_CONV2_WEIGHT + SIZE_NODE_3_1_CONV2_WEIGHT;
constexpr int OFFSET_NODE_3_1_BN_BIAS = OFFSET_NODE_3_1_BN_WEIGHT + SIZE_NODE_3_1_BN_WEIGHT;
constexpr int OFFSET_NODE_3_1_BN_MEAN = OFFSET_NODE_3_1_BN_BIAS + SIZE_NODE_3_1_BN_BIAS;
constexpr int OFFSET_NODE_3_1_BN_VAR = OFFSET_NODE_3_1_BN_MEAN + SIZE_NODE_3_1_BN_MEAN;

constexpr int TOTAL_WEIGHT_SIZE = OFFSET_NODE_3_1_BN_VAR + SIZE_NODE_3_1_BN_VAR;

// Function declaration
void pack_cell0_weights(data_t* all_weights);

