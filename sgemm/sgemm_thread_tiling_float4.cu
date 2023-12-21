#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "generator.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

const int M = 1024;
const int N = 1024;
const int K = 1024;
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

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

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  __shared__ float sharedA[BM * BK];
  __shared__ float sharedB[BK * BN];

  float resultPerThread[TM * TN] = {0.0};
  float vectorOuterA[TM] = {0.0};
  float vectorOuterB[TN] = {0.0};

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BM;

  const int sharedARow = threadIdx.x / (BK / 4);
  const int sharedACol = threadIdx.x % (BK / 4);
  const int sharedBRow = threadIdx.x / (BN / 4);
  const int sharedBCol = threadIdx.x % (BN / 4);

  for (int dotOrder = 0; dotOrder < K; dotOrder += BK) {
    reinterpret_cast<float4 *>(&sharedA[sharedARow * BK + sharedACol * 4])[0] = reinterpret_cast<float4 *>(&A[sharedARow * K + sharedACol * 4])[0];
    reinterpret_cast<float4 *>(&sharedB[sharedBRow * BN + sharedBCol * 4])[0] = reinterpret_cast<float4 *>(&B[sharedBRow * N + sharedBCol * 4])[0];

    __syncthreads();

    A += BK;
    B += BK * N;

    for (int innerOuterProdOrder = 0; innerOuterProdOrder < BK;
         innerOuterProdOrder++) {
        for (int i = 0; i < TM; i++) {
            vectorOuterA[i] =
            sharedA[(threadRow * TM + i) * BK + innerOuterProdOrder];
        }
        for (int i = 0; i < TN; i++) {
            vectorOuterB[i] =
            sharedB[innerOuterProdOrder * BN + (threadCol * TN + i)];
        }

        for (int resultRow = 0; resultRow < TM; resultRow++) {
            for (int resultCol = 0; resultCol < TN; resultCol++) {
                resultPerThread[resultRow * TN + resultCol] +=
              vectorOuterA[resultRow] * vectorOuterB[resultCol];
            }
        }
        

    }

    __syncthreads();

  }

  for (int resultRow = 0; resultRow < TM; resultRow += 1) {
    for (int resultCol = 0; resultCol < TN; resultCol += 4) { 
        float4 tmp;

        tmp.x = resultPerThread[resultRow * TN + resultCol + 0];
      tmp.y = resultPerThread[resultRow * TN + resultCol + 1];
      tmp.z = resultPerThread[resultRow * TN + resultCol + 2];
      tmp.w = resultPerThread[resultRow * TN + resultCol + 3];

      reinterpret_cast<float4 *>(&C[(threadRow * TM + resultRow) * N +
                                    (threadCol * TN + resultCol)])[0] = tmp;
    }
  }

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
    printf("Before kernel launch\n");
    naiveSgemm<<<gridDim, blockDim>>>( M, N, K, d_a, d_b, d_c);
    cudaCheck(cudaDeviceSynchronize());
    printf("After kernel launch\n");

    cudaCheck(cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost));

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        // if (this_error > 0) {
        //     printf("index %d, host calculate %f, device %f, error %f\n", i, h_c[i], h_d_c[i], this_error);
        // }
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

    printf("\nKernal = sgemm thread tailing\n");

    {
        int thread_num = BM * BN / TM / TN;
        dim3 blockDim(thread_num);
        dim3 gridDim((M - 1) / BM + 1, (N - 1) / BN + 1);
        float max_error = testMaxError(gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    return 0;
}