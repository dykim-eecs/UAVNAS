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
//  - Module Name : concat_layer.h
//  - Function    : Channel-wise concatenation layers (2-way and 4-way) for HLS
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef CONCAT_LAYER_H
#define CONCAT_LAYER_H

#include "model.h"

//
// Concatenation Layer (4-way)
// Concatenates four input feature maps along the channel dimension.
//
// Inputs:
//   - in0, in1, in2, in3: input feature maps with shape [CH][H][W]
// Output:
//   - output: concatenated feature map with shape [4*CH][H][W]
//
template<int CH, int H, int W>
void concat_layer(
    data_t in0[CH][H][W],
    data_t in1[CH][H][W],
    data_t in2[CH][H][W],
    data_t in3[CH][H][W],
    data_t output[4*CH][H][W]
) {
    for (int c = 0; c < CH; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w]         = in0[c][h][w];
                output[CH + c][h][w]    = in1[c][h][w];
                output[2*CH + c][h][w]  = in2[c][h][w];
                output[3*CH + c][h][w]  = in3[c][h][w];
            }
        }
    }
}

//
// Concatenation Layer (2-way)
// Concatenates two input feature maps along the channel dimension.
//
// Inputs:
//   - in0, in1: input feature maps with shape [CH/2][H][W]
// Output:
//   - output: concatenated feature map with shape [CH][H][W]
//
template<int OUT_CH, int H, int W>
void concat2_layer(
    data_t in0[OUT_CH / 2][H][W],
    data_t in1[OUT_CH / 2][H][W],
    data_t output[OUT_CH][H][W]
) {
    for (int c = 0; c < OUT_CH / 2; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w]             = in0[c][h][w];
                output[OUT_CH / 2 + c][h][w] = in1[c][h][w];
            }
        }
    }
}


#endif // CONCAT_LAYER_H
