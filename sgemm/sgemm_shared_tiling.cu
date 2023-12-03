#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "generator.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

const int BM = 32;
const int BN = 32;
const int BK = 32;
const int M = 1024;
const int N = 1024;
const int K = 1024;
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    return;
};

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__ void naiveSgemm(int M, int N, int K, float *A, float *B,  float *C) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * blockDim.y * K];
    B = &B[bx * blockDim.x];
    C = &C[by * blockDim.y * N + bx * blockDim.x];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = tmp;
}

float testMaxError(
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a = new float[M * K];
    float *h_b = new float[K * N];
    float *h_c = new float[M * N];
    float *h_d_c = new float[M * N];
    float *d_a, *d_b, *d_c;

    generate_array(h_a, M * K);
    generate_array(h_b, K * N);
    generate_array(h_c, M * N);
    generate_array(h_d_c, M * N);

    cudaCheck(cudaMalloc(&d_a, size_a));
    cudaCheck(cudaMalloc(&d_b, size_b));
    cudaCheck(cudaMalloc(&d_c, size_c));

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    
    cudaCheck(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    naiveSgemm<<<gridDim, blockDim>>>( M, N, K, d_a, d_b, d_c);
    cudaCheck(cudaDeviceSynchronize());

    cudaCheck(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

int main() {

    printf("\nKernal = naiveSgemm\n");

    {

        dim3 blockDim(BM, BN);
        dim3 gridDim((M + BM - 1) / BM, (N + BN - 1) / BN);
        float max_error = testMaxError(gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    return 0;
}