#include <stdio.h>
#include <stdlib.h>

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

void matmul_10x784(const float *W, const float *X, float *Y) {
    for (int i = 0; i < 10; i++) {
        float sum = 0.0f;
        for (int j = 0; j < IMAGE_SIZE; j++) {
            sum += W[i * IMAGE_SIZE + j] * X[j];
        }
        Y[i] = sum;
    }
}