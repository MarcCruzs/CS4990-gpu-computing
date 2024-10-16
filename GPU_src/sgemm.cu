#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

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

__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float *A_d, const float *B_d, float* C_d){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;

            for (int i = 0; i < k; ++i) {
                sum += A_d[row*k + i] * B_d[i*n + col];
            }

            C_d[row * n + col] = sum;
        }
    }
}

__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float *A_d, const float *B_d, float* C_d){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;

            for (int i = 0; i < k; ++i) {
                sum += A_d[row*k + i] * B_d[i*n + col];
            }

            C_d[row*n + col] = sum;
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

    printf("    cudaMalloc: %f s\n", endTime - startTime);
    fflush(stdout);

    // (2) Copy data to device memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (3) Call kernel to launch a grid of threads to perform the computation on GPU
    dim3 gridDim(16, 16, 1);
    dim3 blockDim(ceil((float)m / blockDim.x), ceil((float)n / blockDim.y));

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("    matrixMulKernel_1thread1element<<<(%d, %d, %d),(%d, %d, %d)>>>: %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
    fflush(stdout);

    // (4) Copy the result data from the device memory to the host memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));

    return 0;
}

void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    double startTime, endTime;
    
    // (1) Allocate device memory
    float *A_d, *B_d, *C_d;
    
    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**) &A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void**) &B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void**) &C_d, sizeof(float) * m * n));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMalloc: %f s\n", endTime - startTime);
    fflush(stdout);

    // (2) Copy data to device memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (3) Call kernel to launch a grid of threads to perform the computation on GPU
    dim3 gridDim(16, 1);
    dim3 blockDim(ceil((float)m / blockDim.x), ceil((float)n / blockDim.y));

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("    basicSgemm_d_1thread1row<<<(%d, %d, %d),(%d, %d, %d)>>>: %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
    fflush(stdout);

    // (4) Copy the result data from the device memory to the host memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));

    return 0;
}

void basicSgemm_d_1thread1column(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    double startTime, endTime;
    
    // (1) Allocate device memory
    float *A_d, *B_d, *C_d;
    
    startTime = myCPUTimer();
    CHECK(cudaMalloc((void**) &A_d, sizeof(float) * m * k));
    CHECK(cudaMalloc((void**) &B_d, sizeof(float) * k * n));
    CHECK(cudaMalloc((void**) &C_d, sizeof(float) * m * n));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMalloc: %f s\n", endTime - startTime);
    fflush(stdout);

    // (2) Copy data to device memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * k * n, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (3) Call kernel to launch a grid of threads to perform the computation on GPU
    dim3 gridDim(16, 1);
    dim3 blockDim(ceil((float)m / blockDim.x), ceil((float)n / blockDim.y));

    startTime = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("    basicSgemm_d_1thread1column<<<(%d, %d, %d),(%d, %d, %d)>>>: %f s\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, endTime - startTime);
    fflush(stdout);

    // (4) Copy the result data from the device memory to the host memory
    startTime = myCPUTimer();
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    endTime = myCPUTimer();

    printf("    cudaMemcpy: %f s\n", endTime - startTime);
    fflush(stdout);

    // (5) Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));

    return 0;
}

bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    unsigned int total = nRows * nCols; 
    for (unsigned int i = 0; i < total; ++i) {
        if (CPU_Answer[i] != GPU_Answer[i]) {
            return false;
        }
    }
    return true;
}


int main( int argc, char** argv){
    // setting dimensions (m * k * n)
    unsigned int m = argv[0];
    unsigned int k = argv[1];
    unsigned int n = argv[2];

    bool isVerified;

    
    // generating random float-point values
    float* A_h = (float*) malloc(sizeof(float) * m * k);
    for(unsigned int i = 0; i < n; i++){
        A_h[i] = rand()%100 / 100.0;
    }
    float* B_h = (float*) malloc(sizeof(float) * k * n);
    for(unsigned int i = 0; i < n; i++){
        B_h[i] = rand()%100 / 100.0;
    }
    float* C_h = (float*) calloc(n, sizeof(float));
    float* CPU_Answer = (float*) calloc(n, sizeof(float));


    // CPU
    startTime = myCPUTimer();
    basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h)
    endTime = myCPUTimer();

    printf("basicSgemm_h (CPU): %f s\n\n", endTime -startTime);
    fflush(stdout);

    CPU_Answer = C_h;

    // basicSgemm_d_1thread1element (GPU)
    startTime = myCPUTimer();
    basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h)
    endTime = myCPUTimer();

    printf("basicSgemm_d_1thread1element (GPU): %f s\n\n", endTime -startTime);
    fflush(stdout);

    isVerified = verify(CPU_Answer, C_h, m, n);
    printf("    verified: %d s\n\n", isVerified);

    // basicSgemm_d_1thread1column (GPU)
    startTime = myCPUTimer();
    basicSgemm_d_1thread1column(int m, int k, int n, const float *A_h, const float *B_h, float* C_h)
    endTime = myCPUTimer();

    printf("basicSgemm_d_1thread1column (GPU): %f s\n\n", endTime -startTime);
    fflush(stdout);

    isVerified = verify(CPU_Answer, C_h, m, n);
    printf("    verified: %d s\n\n", isVerified);

    // basicSgemm_d_1thread1row (GPU)
    startTime = myCPUTimer();
    basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    endTime = myCPUTimer();

    printf("basicSgemm_d_1thread1row (GPU): %f s\n\n", endTime -startTime);
    fflush(stdout);

    isVerified = verify(CPU_Answer, C_h, m, n);
    printf("    verified: %d s\n\n", isVerified);


    free(A_h);
    free(B_h);
    free(C_h);

    free(CPU_Answer);
    free(isVerified);
    free(startTime);
    free(endTime);

    return 0;
}