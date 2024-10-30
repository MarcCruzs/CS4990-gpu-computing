#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16
#define TILE_DIM 16

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d, ",__FILE__, __LINE__); \
        printf("code: %d, reason %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    for (int i = 0; i < m; ++i) {       
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            
            for (int h = 0; h < k; ++h) {
                sum += A_h[i * k + h] * B_h[h * n + j];
            }

            C_h[i * n + j] = sum;      
        }
    }
}

__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;

        for (int i = 0; i < k; ++i) {
            sum += A_d[row*k + i] * B_d[i*n + col];
        }

        C_d[row*n + col] = sum;
    }
}

__global__ void matrixMulKernel_tiled(int m, int k, int n, const float *A_d, const float *B_d, float *C_d, unsigned Adz_sz, unsigned Bdz_sz) {
    extern __shared__ float As_Bs[];

    float* A_s = (float*) As_Bs;
    float* B_s = (float*) As_Bs + Adz_sz + Bdz_sz;

    unsigned int row = blockIdx.y * TILE_DIM + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    for (unsigned int tile = 0; tile < (k + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Load elements of the current tile of A into shared memory
        if (row < m && (tile * TILE_DIM + threadIdx.x) < k) {
            A_s[threadIdx.y * TILE_DIM + threadIdx.x] = A_d[row * k + tile * TILE_DIM + threadIdx.x];
        } else {
            A_s[threadIdx.y * TILE_DIM + threadIdx.x] = 0.0f;
        }

        // Load elements of the current tile of B into shared memory
        if (col < n && (tile * TILE_DIM + threadIdx.y) < k) {
            B_s[threadIdx.y * TILE_DIM + threadIdx.x] = B_d[(tile * TILE_DIM + threadIdx.y) * n + col];
        } else {
            B_s[threadIdx.y * TILE_DIM + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate the results
        for (unsigned int i = 0; i < TILE_DIM; ++i) {
            sum += A_s[threadIdx.y * TILE_DIM + i] * B_s[i * TILE_DIM + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C_d[row * n + col] = sum;
    }
}


void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    double startTime, endTime;
    
    // (1) Allocate device memory
    float *A_d, *B_d, *C_d;
    
    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**) &A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void**) &B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void**) &C_d, sizeof(float) * m * n));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMalloc:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (2) Copy data to device memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMemcpy:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (3) Call kernel to launch a grid of threads to perform the computation on GPU
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y); 

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("    matrixMulKernel_1thread1element<<<(%d, %d, %d),(%d, %d, %d)>>>:                   %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
    fflush(stdout);

    // (4) Copy the result data from the device memory to the host memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();

    printf("    cudaMemcpy:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

void basicSgemm_d_tiled (int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    double startTime, endTime;

    // (1) Allocate device memory
    float *A_d, *B_d, *C_d;
    cudaDeviceProp devProp;

    // calculating appropriate SM usage
    CHECK(cudaGetDeviceProperties(&devProp, 0));
    size_t sharedMemPerBlock = devProp.sharedMemPerBlock;
    size_t size = 2 * TILE_DIM * TILE_DIM * sizeof(float);

    if (size > sharedMemPerBlock) {
        std::cerr << "Error: Required shared memory (" << size
                << " bytes) exceeds the device limit (" << sharedMemPerBlock << " bytes)." << std::endl;
        return; 
    }

    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**)&A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float) * m * n));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMalloc:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (2) Copy data to device memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMemcpy:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (3) Call kernel to launch a grid of threads to perform the computation on GPU
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y); 

    startTime = myCPUTimer();
    matrixMulKernel_tiled<<<gridDim, blockDim, size>>>(m, k, n, A_d, B_d, C_d, TILE_DIM * TILE_DIM, TILE_DIM * TILE_DIM);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("    matrixMulKernel_tiled<<<(%d, %d, %d),(%d, %d, %d),(%d)>>>:                                   %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, size, endTime - startTime);
    fflush(stdout);

    // (4) Copy the result data from the device memory to the host memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();

    printf("    cudaMemcpy:                                                                      %f s\n", endTime - startTime);
    fflush(stdout);

    // (5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}


bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    unsigned int total = nRows * nCols; 
    const float tolerance = 1e-3;

    for (unsigned int i = 0; i < total; ++i) {
        if (fabs(CPU_Answer[i] - GPU_Answer[i]) > tolerance) {
            printf("%f != %f on %d \n", CPU_Answer[i], GPU_Answer[i], i);
            return false;
        }
    }
    return true;
}

int main( int argc, char** argv){
    // setting dimensions (m * k * n)
    unsigned int m = atoi(argv[1]);
    unsigned int k = atoi(argv[2]);
    unsigned int n = atoi(argv[3]);

    bool isVerified;
    double startTime, endTime;

    // CPU setup
    float* A_h = (float*) malloc(sizeof(float) * m * k);  
    float* B_h = (float*) malloc(sizeof(float) * k * n);  
    float* C_h = (float*) calloc(m * n, sizeof(float));  
    float* CPU_Answer = (float*) calloc(m * n, sizeof(float)); 

    // generating random float-point values
    for(unsigned int i = 0; i < m * k; i++){ 
        A_h[i] = rand()%100 / 100.0;
    }
    for(unsigned int i = 0; i < k * n; i++){  
        B_h[i] = rand()%100 / 100.0;
    }
    
    // CPU
    startTime = myCPUTimer();
    basicSgemm_h(m, k, n, A_h, B_h, C_h);
    endTime = myCPUTimer();

    printf("basicSgemm_h (CPU):                                                                  %f s\n\n", endTime -startTime);
    fflush(stdout);

    memcpy(CPU_Answer, C_h, sizeof(float) * m * n);

    // basicSgemm_d_1thread1element (GPU)
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h);
    endTime = myCPUTimer();

    isVerified = verify(CPU_Answer, C_h, m, n);
    printf("    Verification Results:                                                            %s\n", isVerified ? "SUCCESS" : "FAILED");
    fflush(stdout);

    printf("basicSgemm_d_1thread1element (GPU):                                                  %f s\n\n", endTime -startTime);
    fflush(stdout);

    // basicSgemm_d_1thread1element (GPU)
    startTime = myCPUTimer();
    basicSgemm_d_tiled(m, k, n, A_h, B_h, C_h);
    endTime = myCPUTimer();

    isVerified = verify(CPU_Answer, C_h, m, n);
    printf("    Verification Results:                                                            %s\n", isVerified ? "SUCCESS" : "FAILED");
    fflush(stdout);

    printf("basicSgemm_d_tiled (GPU):                                                            %f s\n\n", endTime -startTime);
    fflush(stdout);

    free(A_h);
    free(B_h);
    free(C_h);

    free(CPU_Answer);

    return 0;
}