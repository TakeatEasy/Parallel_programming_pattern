#include "generator.h"


#define BLOCK_SIZE 1024

int ceil(int a, int b){
    return int((a + b - 1) / b);
}

__global__ void reduce_warp_divergence(float * in, int len) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int did = threadIdx.x + blockDim.x * blockIdx.x;

    if (did < len) {
        sdata[tid] = in[did];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1 ; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        in[blockIdx.x] = sdata[tid];
    }
}

int main(int argc, char ** argv) {

    int inputLength = 1024;
    float * hostInput;
    float * hostOutput;
    float * deviceInput;

    hostInput = new float[inputLength];
    hostOutput = new float[inputLength];

    generate_array(hostInput, inputLength);

    float res = 0.0;
    for (int i = 0; i < inputLength; i++) {
        res += hostInput[i];
    }
    std::cout << "The result for cpu is:" << res << std::endl;

    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceInput, sizeof(float) * inputLength);

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(inputLength, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    
    //@@ Launch the GPU Kernel here
    reduce_warp_divergence<<<DimGrid, DimBlock>>>(deviceInput, inputLength);

    cudaDeviceSynchronize();
    
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceInput, sizeof(float) * inputLength, cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(deviceInput);

    std::cout << "The result for gpu is:" << hostOutput[0] << std::endl;
    std::cout << std::endl;

    free(hostInput);
    free(hostOutput);

    return 0;
}