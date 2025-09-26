#ifndef MODEL_H
#define MODEL_H
#include <ap_fixed.h>

//typedef float data_t;          // Floating-point data type
typedef ap_fixed<16,8> data_t;  // total 16 bits, 6 integer bits

typedef unsigned char bit_t;   // Bit type (used for ReLU masking or flags)

constexpr int CH0 = 8;
constexpr int IN0_H = 32;
constexpr int IN0_W = 3072;

constexpr int CH1 = 16;
constexpr int IN1_H = 16;
constexpr int IN1_W = 1536;

// Top-level model function declaration
extern "C" void cell0(
	    data_t cells_0_preprocess0_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
	    data_t cells_0_preprocess1_op_op_1_Conv_output_0[CH0][IN0_H][IN0_W],
	    data_t cells_0_Concat_output_0[4 * CH0][IN0_H / 2][IN0_W / 2],
		data_t relu_buf[2][CH0][IN0_H][IN0_W],
		data_t relu_half_buf[2][CH0][IN0_H / 2][IN0_W / 2],
		data_t conv_buf[4][CH0][IN0_H / 2][IN0_W / 2],
		data_t add_buf[4][CH0][IN0_H / 2][IN0_W / 2],
		data_t concat_buf[2][CH0 / 2][IN0_H / 2][IN0_W / 2],
		data_t concat_out[CH0][IN0_H / 2][IN0_W / 2]
);

// Top-level model function declaration
void cell1(
	    data_t cells_1_preprocess0_bn_BatchNormalization_output_0[CH1][IN1_H][IN1_W],
	    data_t cells_1_preprocess1_op_op_1_Conv_output_0[CH1][IN1_H][IN1_W],
	    data_t cells_1_Concat_output_0[4 * CH1][IN1_H / 2][IN1_W / 2]
);

#endif // MODEL_H
