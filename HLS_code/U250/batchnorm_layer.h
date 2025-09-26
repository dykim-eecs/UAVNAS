#ifndef BATCHNORM_LAYER_H
#define BATCHNORM_LAYER_H

#include <hls_math.h>
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
    // eps 값을 ap_fixed<16, 8> 타입이 0으로 변환하지 않도록 충분히 크게 설정
    data_t eps = 0.004;

    // Precompute scale and bias terms for each channel
    data_t scale[CH];
    data_t bias[CH];

    for (int c = 0; c < CH; ++c) {
        // var[c] + eps 연산 결과가 0이 되지 않도록 보장
        data_t inv_std = data_t(1.0) / hls::sqrt(var[c] + eps);

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
