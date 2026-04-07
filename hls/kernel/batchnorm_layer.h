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
//  - Module Name : batchnorm_layer.h
//  - Function    : Batch normalization layer for HLS
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include <cmath>
#include "model.h"

//
// Batch Normalization Layer
//
template<int CH, int H, int W>
void batchnorm_layer(
    data_t input[CH][H][W],
    data_t gamma[CH],
    data_t beta[CH],
    data_t mean[CH],
    data_t var[CH],
    data_t output[CH][H][W]
) {
    // Set eps large enough to prevent ap_fixed<16,8> from rounding to zero
    data_t eps = 0.001;

    // Precompute scale and bias terms for each channel
    data_t scale[CH];
    data_t bias[CH];

    for (int c = 0; c < CH; ++c) {
        // Ensure (var[c] + eps) does not evaluate to zero
        data_t inv_std = data_t(1.0) / sqrt(var[c] + eps);

        scale[c] = gamma[c] * inv_std;
        bias[c]  = beta[c] - scale[c] * mean[c];
    }

    // Apply normalization using only multiplication and addition
    for (int c = 0; c < CH; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h][w] = scale[c] * input[c][h][w] + bias[c];
            }
        }
    }
}

#endif // BATCHNORM_LAYER_H
