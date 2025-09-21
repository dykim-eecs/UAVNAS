#include "model.h"
#include "cell0_weights.h"
#include <hls_math.h>
#include <iostream>
#include <cfloat>

// relu
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

// add
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

// concat
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
                output[c][h][w] = in0[c][h][w];
                output[CH + c][h][w] = in1[c][h][w];
                output[2*CH + c][h][w] = in2[c][h][w];
                output[3*CH + c][h][w] = in3[c][h][w];
            }
        }
    }
}

template<int OUT_CH, int H, int W>
void concat2_layer(
    data_t in0[OUT_CH / 2][H][W],
    data_t in1[OUT_CH / 2][H][W],
    data_t output[OUT_CH][H][W]
) {
    for (int c = 0; c < OUT_CH / 2; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                output[c][h][w] = in0[c][h][w];
                output[OUT_CH / 2 + c][h][w] = in1[c][h][w];
            }
        }
    }
}

// maxpool
template<int CH, int K, int H, int W, int STRIDE>
void maxpool_layer(
    data_t input[CH][H][W],
    data_t output[CH][(H + 2 - K)/STRIDE + 1][(W + 2 - K)/STRIDE + 1],
    int stride,
    int padding
) {
    const int OH = (H + 2 * padding - K) / stride + 1;
    const int OW = (W + 2 * padding - K) / stride + 1;
    const int PH = H + 2 * padding;
    const int PW = W + 2 * padding;
    data_t padded[CH][PH][PW];
    for (int c = 0; c < CH; c++) {
        for (int i = 0; i < PH; i++) {
            for (int j = 0; j < PW; j++) {
                if (i >= padding && i < H + padding && j >= padding && j < W + padding)
                    padded[c][i][j] = input[c][i - padding][j - padding];
                else
                    padded[c][i][j] = -FLT_MAX;
            }
        }
    }
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

// batchnorm
template<int CH, int H, int W>
void batchnorm_layer(
    data_t input[CH][H][W],
    const data_t gamma[CH],
    const data_t beta[CH],
    const data_t mean[CH],
    const data_t var[CH],
    data_t output[CH][H][W],
    float eps = 1e-5
) {
    data_t scale[CH];
    data_t bias[CH];
    for (int c = 0; c < CH; ++c) {
#pragma HLS UNROLL
        data_t inv_std = data_t(1.0) / hls::sqrt(var[c] + data_t(eps));
        scale[c] = gamma[c] * inv_std;
        bias[c] = beta[c] - scale[c] * mean[c];
    }
    for (int c = 0; c < CH; ++c) {
#pragma HLS UNROLL
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                output[c][h][w] = scale[c] * input[c][h][w] + bias[c];
            }
        }
    }
}

// conv
template<int IN_CH, int OUT_CH, int IN_H, int IN_W, int OUT_H, int OUT_W, int K, int GROUP>
void conv_layer(
    data_t input[IN_CH][IN_H][IN_W],
    const data_t weight[OUT_CH][IN_CH / GROUP][K][K],
    const data_t* bias,
    data_t output[OUT_CH][OUT_H][OUT_W],
    int stride,
    int padding,
    int group,
    int dilation
) {
    const int in_ch_per_group = IN_CH / group;
    const int out_ch_per_group = OUT_CH / group;
    for (int g = 0; g < group; g++) {
        for (int oc = 0; oc < out_ch_per_group; oc++) {
            const int oc_global = g * out_ch_per_group + oc;
            for (int oh = 0; oh < OUT_H; oh++) {
                for (int ow = 0; ow < OUT_W; ow++) {
                    data_t sum = (bias != nullptr) ? bias[oc_global] : data_t(0);
                    for (int ic = 0; ic < in_ch_per_group; ic++) {
                        const int ic_global = g * in_ch_per_group + ic;
                        for (int kh = 0; kh < K; kh++) {
                            for (int kw = 0; kw < K; kw++) {
                                int ih = oh * stride - padding + kh * dilation;
                                int iw = ow * stride - padding + kw * dilation;
                                if (ih >= 0 && ih < IN_H && iw >= 0 && iw < IN_W) {
                                    sum += input[ic_global][ih][iw] * weight[oc_global][ic][kh][kw];
                                }
                            }
                        }
                    }
                    output[oc_global][oh][ow] = sum;
                }
            }
        }
    }
}

extern "C" void cell0(
    data_t cells_0_preprocess0_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
    data_t cells_0_preprocess1_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
    data_t cells_0_Concat_output_0[4 * CH0][IN0_H / 2][IN0_W / 2],
    const data_t *all_weights
) {
#pragma HLS INTERFACE m_axi port=cells_0_preprocess0_op_op_1_Conv_output_0 offset=slave bundle=input0 depth=CH0*IN0_H*IN0_W
#pragma HLS INTERFACE m_axi port=cells_0_preprocess1_op_op_1_Conv_output_0 offset=slave bundle=input1 depth=CH0*IN0_H*IN0_W
#pragma HLS INTERFACE m_axi port=cells_0_Concat_output_0 offset=slave bundle=output depth=4*CH0*(IN0_H/2)*(IN0_W/2)
#pragma HLS INTERFACE m_axi port=all_weights offset=slave bundle=weights depth=TOTAL_WEIGHT_SIZE num_read_outstanding=16 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=return

    static data_t relu_buf[2][CH0][IN0_H][IN0_W];
    static data_t relu_half_buf[2][CH0][IN0_H / 2][IN0_W / 2];
    static data_t conv_buf[4][CH0][IN0_H / 2][IN0_W / 2];
    static data_t add_buf[4][CH0][IN0_H / 2][IN0_W / 2];
    static data_t concat_buf[2][CH0 / 2][IN0_H / 2][IN0_W / 2];
    static data_t concat_out[CH0][IN0_H / 2][IN0_W / 2];
#pragma HLS ARRAY_PARTITION variable=relu_buf dim=1 factor=2 cyclic
#pragma HLS ARRAY_PARTITION variable=relu_half_buf dim=1 factor=2 cyclic
#pragma HLS ARRAY_PARTITION variable=conv_buf dim=1 factor=2 cyclic
#pragma HLS ARRAY_PARTITION variable=add_buf dim=1 factor=2 cyclic
#pragma HLS ARRAY_PARTITION variable=concat_buf dim=1 factor=2 cyclic
#pragma HLS ARRAY_PARTITION variable=concat_out dim=1 factor=2 cyclic

    // 0.0
    relu_layer<CH0, IN0_H, IN0_W>(cells_0_preprocess1_op_op_1_Conv_output_0, relu_buf[0]);
    conv_layer<CH0, CH0, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_buf[0], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_0_0_OP1_WEIGHT], nullptr, conv_buf[0], 2, 4, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_316], &all_weights[OFFSET_ONNX_CONV_317], conv_buf[1], 1, 0, 1, 1);
    // 0.1
    relu_layer<CH0, IN0_H, IN0_W>(cells_0_preprocess0_op_op_1_Conv_output_0, relu_buf[1]);
    conv_layer<CH0, CH0, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_buf[1], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_0_1_OP1_WEIGHT], nullptr, conv_buf[2], 2, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_319], &all_weights[OFFSET_ONNX_CONV_320], conv_buf[3], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[3], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_0_1_OP5_WEIGHT], nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_322], &all_weights[OFFSET_ONNX_CONV_323], conv_buf[3], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[3], add_buf[0]);
    // 1.0
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_NODE_1_0_CONV1_WEIGHT], nullptr, concat_buf[0], 2, 0, 1, 1);
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_NODE_1_0_CONV2_WEIGHT], nullptr, concat_buf[1], 2, 0, 1, 1);
    concat2_layer<CH0, IN0_H / 2, IN0_W / 2>(concat_buf[0], concat_buf[1], concat_out);
    batchnorm_layer<CH0, IN0_H / 2, IN0_W / 2>(
        concat_out, &all_weights[OFFSET_NODE_1_0_BN_WEIGHT], &all_weights[OFFSET_NODE_1_0_BN_BIAS],
        &all_weights[OFFSET_NODE_1_0_BN_MEAN], &all_weights[OFFSET_NODE_1_0_BN_VAR], conv_buf[0]);
    // 1.1
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[0], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_1_1_OP1_WEIGHT], nullptr, conv_buf[1], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[1], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_325], &all_weights[OFFSET_ONNX_CONV_326], conv_buf[2], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[2], relu_half_buf[1]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[1], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_1_1_OP5_WEIGHT], nullptr, conv_buf[1], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[1], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_328], &all_weights[OFFSET_ONNX_CONV_329], conv_buf[2], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[0], conv_buf[2], add_buf[1]);
    // 2.0
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 3, 8>(
        relu_half_buf[0], (data_t (*)[1][3][3]) &all_weights[OFFSET_NODE_2_0_OP1_WEIGHT], nullptr, conv_buf[0], 1, 2, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_331], &all_weights[OFFSET_ONNX_CONV_332], conv_buf[1], 1, 0, 1, 1);
    // 2.1
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[1], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_2_1_OP1_WEIGHT], nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_334], &all_weights[OFFSET_ONNX_CONV_335], conv_buf[3], 1, 0, 1, 1);
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[3], relu_half_buf[1]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[1], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_2_1_OP5_WEIGHT], nullptr, conv_buf[2], 1, 2, 8, 1);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[2], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_337], &all_weights[OFFSET_ONNX_CONV_338], conv_buf[3], 1, 0, 1, 1);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[3], add_buf[2]);
    // 3.0
    relu_layer<CH0, IN0_H / 2, IN0_W / 2>(add_buf[2], relu_half_buf[0]);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 5, 8>(
        relu_half_buf[0], (data_t (*)[1][5][5]) &all_weights[OFFSET_NODE_3_0_OP1_WEIGHT], nullptr, conv_buf[0], 1, 4, 8, 2);
    conv_layer<CH0, CH0, IN0_H / 2, IN0_W / 2, IN0_H / 2, IN0_W / 2, 1, 1>(
        conv_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_ONNX_CONV_340], &all_weights[OFFSET_ONNX_CONV_341], conv_buf[1], 1, 0, 1, 1);
    // 3.1
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_NODE_3_1_CONV1_WEIGHT], nullptr, concat_buf[0], 2, 0, 1, 1);
    conv_layer<CH0, CH0 / 2, IN0_H, IN0_W, IN0_H / 2, IN0_W / 2, 1, 1>(
        relu_buf[0], (data_t (*)[8][1][1]) &all_weights[OFFSET_NODE_3_1_CONV2_WEIGHT], nullptr, concat_buf[1], 2, 0, 1, 1);
    concat2_layer<CH0, IN0_H / 2, IN0_W / 2>(concat_buf[0], concat_buf[1], concat_out);
    batchnorm_layer<CH0, IN0_H / 2, IN0_W / 2>(
        concat_out, &all_weights[OFFSET_NODE_3_1_BN_WEIGHT], &all_weights[OFFSET_NODE_3_1_BN_BIAS],
        &all_weights[OFFSET_NODE_3_1_BN_MEAN], &all_weights[OFFSET_NODE_3_1_BN_VAR], conv_buf[2]);
    add_layer<CH0, IN0_H / 2, IN0_W / 2>(conv_buf[1], conv_buf[2], add_buf[3]);
    concat_layer<CH0, IN0_H / 2, IN0_W / 2>(
        add_buf[0], add_buf[1], add_buf[2], add_buf[3], cells_0_Concat_output_0);
}
