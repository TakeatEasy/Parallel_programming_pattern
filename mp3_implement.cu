#include    <wb.h>

#define BLOCK_SIZE 8

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

int ceil(int a, int b){
    return (a + b - 1) / b;
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    
    __shared__ float subTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subTileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // printf("row is %d, value is %f\n", row, A[row * numAColumns + 0]);
    int step = (numAColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float value = 0.0;
    for (int i = 0; i < step; i++) {
        if ((i * BLOCK_SIZE + tx) < numAColumns) {
            subTileA[ty][tx] = A[row * numAColumns + i * BLOCK_SIZE + tx];
        } else {
            subTileA[ty][tx] = 0.0;
        }
        if ((i * BLOCK_SIZE + ty) > numBRows) {
            subTileB[ty][tx] = 0.0;
        } else {
            subTileB[ty][tx] = B[(i * BLOCK_SIZE + ty) * numBColumns + col];
        }
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++) {
            value += subTileA[ty][j] * subTileB[j][tx];
        }
        __syncthreads();

    }
    

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = value;
        printf("row is %d, col is %d, value is %f\n", row, col, value);
    }

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(sizeof(float) * numCColumns * numCRows);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceA, sizeof(float) * numAColumns * numARows);
    cudaMalloc((void**) &deviceB, sizeof(float) * numBColumns * numBRows);
    cudaMalloc((void**) &deviceC, sizeof(float) * numCColumns * numCRows);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float) * numAColumns * numARows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBColumns * numBRows, cudaMemcpyHostToDevice);
    

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(numCColumns, BLOCK_SIZE), ceil(numCRows, BLOCK_SIZE), 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCColumns * numCRows, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}