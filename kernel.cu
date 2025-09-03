
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>
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

void multiplyMatricesCPU(
    double* a,
    int w1,
    int h1,
    double* b,
    int w2,
    int h2,
    double*& c
) {
    c = (double*)calloc(h1 * w2, sizeof(double));
    for (int i = 0; i < h1; i++) {
        for (int j = 0; j < w2; j++) {
            double sum = 0;
            for (int k = 0; k < w1; k++) {
                sum += a[i * w1 + k] * b[k * w2 + j];
            }
            c[i * w2 + j] = sum;
        }
    }
}

__global__ void multiplyMatricesGPU(
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

int main(int argc, char** args) {
    srand(time(NULL));
    int w1 = 250, h1 = 400, w2 = 300, h2 = w1;
    double* x, * y, * z;
    initializeMatrix(x, w1, h1);
    initializeMatrix(y, w2, h2);

    float millis;
    
    if (argc < 2 || string(args[1]) != "--parallel") {
        auto start = chrono::high_resolution_clock::now();
        multiplyMatricesCPU(x, w1, h1, y, w2, h2, z);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        millis = duration.count();
    } else {
        z = (double*)calloc(h1 * w2, sizeof(double));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double* d_x, * d_y, * d_z;
        cudaMalloc(&d_x, w1 * h1 * sizeof(double));
        cudaMemcpy(d_x, x, w1 * h1 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_y, w2 * h2 * sizeof(double));
        cudaMemcpy(d_y, y, w2 * h2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_z, w2 * h1 * sizeof(double));

        cudaEventRecord(start);
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((w2 + 15) / 16, (h1 + 15) / 16);
        multiplyMatricesGPU << <blocksPerGrid, threadsPerBlock >> > (d_x, w1, h1, d_y, w2, h2, d_z);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millis, start, stop);

        cudaMemcpy(z, d_z, w2 * h1 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
    }

    printMatrix(x, w1, h1);
    printMatrix(y, w2, h2);
    printMatrix(z, w2, h1);
    free(x);
    free(y);
    free(z);
    cout << "\n" << "Executed for " << millis << " ms" << "\n";
    return 0;
}

