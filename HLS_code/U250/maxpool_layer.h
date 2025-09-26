#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "model.h"

#include <cfloat>

//
// Max Pooling Layer
//
// Applies 2D max pooling over an input feature map with configurable
// kernel size, stride, and padding.
//
// Template Parameters:
//   - CH:     number of channels
//   - K:      kernel size (assumed square, K x K)
//   - H, W:   height and width of the input
//   - STRIDE: stride (used for output shape calculation)
//
// Inputs:
//   - input:  input tensor of shape [CH][H][W]
//   - stride: stride size for pooling
//   - padding: zero-padding size (applied symmetrically)
//
// Output:
//   - output: result tensor of shape [CH][OH][OW], where:
//       OH = (H + 2 * padding - K) / stride + 1
//       OW = (W + 2 * padding - K) / stride + 1
//
template<int CH, int K, int H, int W, int STRIDE>
void maxpool_layer(
    data_t input[CH][H][W],
    data_t output[CH][(H + 2 - K)/STRIDE + 1][(W + 2 - K)/STRIDE + 1], // assumes default padding=1
    int stride,
    int padding
) {
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;
    const int PH = H + 2 * padding;
    const int PW = W + 2 * padding;

    data_t padded[CH][PH][PW];

    // Initialize padded input
    for (int c = 0; c < CH; c++) {
        for (int i = 0; i < PH; i++) {
            for (int j = 0; j < PW; j++) {
                if (i >= padding && i < H + padding && j >= padding && j < W + padding)
                    padded[c][i][j] = input[c][i - padding][j - padding];
                else
                    padded[c][i][j] = -FLT_MAX; // Use negative infinity for max pooling
            }
        }
    }

    // Perform max pooling
    for (int c = 0; c < CH; c++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                data_t max_val = -FLT_MAX;
                for (int kh = 0; kh < K; kh++) {
                    for (int kw = 0; kw < K; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        max_val = (padded[c][ih][iw] > max_val) ? padded[c][ih][iw] : max_val;
                    }
                }
                output[c][oh][ow] = max_val;
            }
        }
    }
}

#endif // MAXPOOL_LAYER_H
