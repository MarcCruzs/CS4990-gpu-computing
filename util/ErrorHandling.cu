#include<stdio.h>
#include<stdlib.h>

#define CHECK(call){ \ 
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d, ",__FILE__, __LINE__); \
        printf("code: %d, reason %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}