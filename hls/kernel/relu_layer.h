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
//  - Module Name : relu_layer.h
//  - Function    : ReLU activation layer for HLS
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "model.h"


//
// ReLU Activation Layer
//
// Applies the Rectified Linear Unit activation function:
//   output = max(0, input)
//
// Template Parameters:
//   - CH: number of channels
//   - H:  height of feature map
//   - W:  width of feature map
//
// Input:
//   - input:  input tensor of shape [CH][H][W]
//
// Output:
//   - output: output tensor of shape [CH][H][W], where each value is max(0, input)
//
template<int CH, int H, int W>
void relu_layer(
    data_t input[CH][H][W],
    data_t output[CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = (input[c][h][w] > data_t(0)) ? input[c][h][w] : data_t(0);
            }
        }
    }
}

#endif // RELU_LAYER_H
