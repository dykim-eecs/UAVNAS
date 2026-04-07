//!****************************************************************************************!*
//! [UAV-NAS Project] CONFIDENTIAL & PROPRIETARY                                           !*
//! Copyright (c) 2026 Doh-Yon Kim (First Author), All Rights Reserved.                 !*
//!                                                                                        !*
//! NOTICE: This source code is associated with the following academic publication:        !*
//! "UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search"                  !*
//! Accepted for publication in IEEE Transactions on Industrial Informatics (TII).         !*
//!                                                                                        !*
//! All information contained herein is, and remains the property of the authors.          !*
//! The intellectual and technical concepts contained herein are protected by copyright     !*
//! law and international treaties. Unauthorized copying, modification, or distribution    !*
//! of this file, via any medium, is strictly prohibited.                                  !*
//!                                                                                        !*
//!****************************************************************************************!*
//
//* [ PAPER METADATA ]
//  - Title   : UAV-NAS: UAV Identification on FPGAs via Neural Architecture Search
//  - Journal : IEEE Transactions on Industrial Informatics (TII)
//
//* [ AUTHOR INFORMATION ]
//  - First Author : Kim, Doh-Yon
//
//* [ FILE DESCRIPTION ]
//  - Module Name : cell1_weights.h
//  - Function    : Weight declarations for Cell 1 (auto-generated)
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef CELL1_WEIGHTS_H
#define CELL1_WEIGHTS_H

// Auto-generated weight declarations
#include "model.h"

// ===== Cell 1 =====
// -- Node 0.0 & 0.1 --
extern data_t cells_1_ops_by_node_0_0_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_0_1_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_0_1_op_5_weight[16][1][5][5];

// -- Node 1.0 --
extern data_t cells_1_ops_by_node_1_0_bn_bias[16];
extern data_t cells_1_ops_by_node_1_0_bn_running_mean[16];
extern data_t cells_1_ops_by_node_1_0_bn_running_var[16];
extern data_t cells_1_ops_by_node_1_0_bn_weight[16];
extern data_t cells_1_ops_by_node_1_0_conv1_weight[8][16][1][1];
extern data_t cells_1_ops_by_node_1_0_conv2_weight[8][16][1][1];

// -- Node 1.1 --
extern data_t cells_1_ops_by_node_1_1_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_1_1_op_5_weight[16][1][5][5];

// -- Node 2.x, 3.x --
extern data_t cells_1_ops_by_node_2_0_op_1_weight[16][1][3][3];
extern data_t cells_1_ops_by_node_2_1_op_1_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_2_1_op_5_weight[16][1][5][5];
extern data_t cells_1_ops_by_node_3_0_op_1_weight[16][1][5][5];

extern data_t cells_1_ops_by_node_3_1_bn_bias[16];
extern data_t cells_1_ops_by_node_3_1_bn_running_mean[16];
extern data_t cells_1_ops_by_node_3_1_bn_running_var[16];
extern data_t cells_1_ops_by_node_3_1_bn_weight[16];
extern data_t cells_1_ops_by_node_3_1_conv1_weight[8][16][1][1];
extern data_t cells_1_ops_by_node_3_1_conv2_weight[8][16][1][1];

//
// ===== ONNX Imported Conv Layers =====
extern data_t onnx_Conv_346[16][16][1][1];
extern data_t onnx_Conv_347[16];
extern data_t onnx_Conv_349[16][16][1][1];
extern data_t onnx_Conv_350[16];
extern data_t onnx_Conv_352[16][16][1][1];
extern data_t onnx_Conv_353[16];
extern data_t onnx_Conv_355[16][16][1][1];
extern data_t onnx_Conv_356[16];
extern data_t onnx_Conv_358[16][16][1][1];
extern data_t onnx_Conv_359[16];
extern data_t onnx_Conv_361[16][16][1][1];
extern data_t onnx_Conv_362[16];
extern data_t onnx_Conv_364[16][16][1][1];
extern data_t onnx_Conv_365[16];
extern data_t onnx_Conv_367[16][16][1][1];
extern data_t onnx_Conv_368[16];
extern data_t onnx_Conv_370[16][16][1][1];
extern data_t onnx_Conv_371[16];

#endif // CELL1_WEIGHTS_H
