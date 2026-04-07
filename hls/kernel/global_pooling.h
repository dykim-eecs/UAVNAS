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
//  - Module Name : global_pooling.h
//  - Function    : Global average pooling layer for HLS
//  - Platform    : Optimized for FPGA Implementation (xck26-sfvc784-2lv-c)
//
//

#ifndef GLOBAL_POOLING_H
#define GLOBAL_POOLING_H

#include "model.h"


//
// Global Average Pooling Layer
//
// Computes the average of each channel across spatial dimensions.
//
// Template Parameters:
//   - CH: number of channels
//   - H:  input height
//   - W:  input width
//
// Input:
//   - input:  feature map of shape [CH][H][W]
//
// Output:
//   - output: vector of shape [CH], where each element is the mean of the corresponding input channel
//
template<int CH, int H, int W>
void global_avg_pool(data_t input[CH][H][W], data_t output[CH]) {
    for (int c = 0; c < CH; c++) {
        data_t sum = 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[c][h][w];
            }
        }
        output[c] = sum / (H * W);
    }
}

#endif // GLOBAL_POOLING_H
