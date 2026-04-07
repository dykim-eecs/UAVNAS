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
//  - Module Name : add_layer.h
//  - Function    : Element-wise tensor addition layer for HLS
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef ADD_LAYER_H
#define ADD_LAYER_H

#include "model.h"

//
// Element-wise addition layer
//
// This function performs element-wise addition of two input tensors.
// Inputs:
//   - input1: first input tensor with shape [CH][H][W]
//   - input2: second input tensor with shape [CH][H][W]
// Output:
//   - output: result tensor with shape [CH][H][W], where each element is the sum of the corresponding elements in input1 and input2
//
template<int CH, int H, int W>
void add_layer(
    data_t input1[CH][H][W],
    data_t input2[CH][H][W],
    data_t output[CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = input1[c][h][w] + input2[c][h][w];
            }
        }
    }
}

#endif // ADD_LAYER_H
