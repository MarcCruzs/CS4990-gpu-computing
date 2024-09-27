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

void vecAdd_h(float* x_h, float* y_h, float* z_h, unsigned int n){
    for(unsigned int i = 0; i < n; i++){
        z_h[i] = x_h[i] + y_h[i];
    }
}

__global__ void vecAddKernel(float *x_d, float *y_d, float *z_d, unsigned int n){
    unsigned int i  = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n){
        z_d[i] = x_d[i] + y_d[i];
    }
}

int main( int argc, char** argv){
    unsigned int n = 1024;

    float* x_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++){
        x_h[i] = (float)rand()/(float)(RAND_MAX);
    }
    float* y_h = (float*) malloc(sizeof(float)*n);
    for(unsigned int i = 0; i < n; i++){
        y_h[i] = (float)rand()/(float)(RAND_MAX);
    }
    float* z_h = (float*) calloc(n, sizeof(float));

    float *x_d, *y_d, *z_d;
    CHECK(cudaMalloc((void**) &x_d, sizeof(float)*n));
    CHECK(cudaMalloc((void**) &y_d, sizeof(float)*n));
    CHECK(cudaMalloc((void**) &z_d, sizeof(float)*n));


    CHECK(cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, y_h, sizeof(float)*n, cudaMemcpyHostToDevice));

    double startTime = myCPUTimer();
    vecAdd_h(x_h, y_h, z_h, n);
    double endTime = myCPUTimer();

    printf("Host Function: %f s\n", endTime - startTime);

    startTime = myCPUTimer();
    vecAddKernel<<<ceil(n/256.0), 256>>>(x_d, y_d, z_d, n);
    CHECK(cudaDeviceSynchronize());
    endTime = myCPUTimer();

    printf("CUDA Kernal: %f s\n", endTime - startTime);

    CHECK(cudaMemcpy(z_h, z_d, sizeof(float)*n, cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(x_d));
    CHECK(cudaFree(y_d));
    CHECK(cudaFree(z_d));

    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}