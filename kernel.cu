
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace std;


double randomNumber(int min, int max) {
    return rand() % (max - min) + min;
}

void initializeMatrix(double*& matrix, int w, int h) {
    matrix = (double*) calloc(w * h, sizeof(double));
    for (int i = 0; i < w * h; i++) {
       matrix[i] = randomNumber(1, 10);
    }
}

void printMatrix(double* matrix, int w, int h) {
    cout << "\n";
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout << matrix[i * w + j] << "\t";
        }
        cout << "\n";
    }
}

__global__ void multiplyMatrices(
    double* a,
    int w1,
    int h1,
    double* b,
    int w2,
    int h2,
    double* c
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < h1 && col < w2) {
        double sum = 0;
        for (int k = 0; k < w1; k++) {
            sum += a[row * w1 + k] * b[k * w2 + col];
        }
        c[row * w2 + col] = sum;
    }
}

int main() {
    srand(time(NULL));
    int w1 = randomNumber(4, 5), h1 = randomNumber(3, 7), w2 = randomNumber(4, 6), h2 = w1;
    double* x, * y, * z;
    initializeMatrix(x, w1, h1);
    initializeMatrix(y, w2, h2);
    z = (double*)calloc(h1 * w2, sizeof(double));

    double* d_x, * d_y, * d_z;
    cudaMalloc(&d_x, w1 * h1 * sizeof(double));
    cudaMemcpy(d_x, x, w1 * h1 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, w2 * h2 * sizeof(double));
    cudaMemcpy(d_y, y, w2 * h2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_z, w2 * h1 * sizeof(double));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((w2 + 15) / 16, (h1 + 15) / 16);
    multiplyMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_x, w1, h1, d_y, w2, h2, d_z);
    cudaMemcpy(z, d_z, w2 * h1 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    printMatrix(x, w1, h1);
    printMatrix(y, w2, h2);
    printMatrix(z, w2, h1);
    free(x);
    free(y);
    free(z);
    return 0;
}

