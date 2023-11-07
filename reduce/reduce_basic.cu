#include "generator.h"


#define BLOCK_SIZE 256

int ceil(int a, int b){
    return int((a + b - 1) / b);
}

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len){
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int argc, char ** argv) {

    int inputLength = 10;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    hostInput1 = new float[inputLength];
    hostInput2 = new float[inputLength];
    hostOutput = new float[inputLength];

    

    generate_array(hostInput1, inputLength);
    generate_array(hostInput2, inputLength);

    for (int i = 0; i < inputLength; i++) {
        std::cout << hostInput1[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < inputLength; i++) {
        std::cout << hostInput2[i] << " ";
    }
    std::cout << std::endl;

    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceInput1, sizeof(float) * inputLength);
    cudaMalloc((void**) &deviceInput2, sizeof(float) * inputLength);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * inputLength);

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(float) * inputLength, cudaMemcpyHostToDevice);
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(inputLength, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    
    //@@ Launch the GPU Kernel here
    vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();
    
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * inputLength, cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    for (int i = 0; i < inputLength; i++) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < inputLength; i++) {
        std::cout << hostInput1[i] + hostInput2[i] << " ";
    }
    std::cout << std::endl;

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}