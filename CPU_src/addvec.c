#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

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

int main( int argc, char** arv){
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

    double startTime = myCPUTimer();
    vecAdd_h(x_h, y_h, z_h, n);
    double endTime = myCPUTimer();
    
    printf("%f s\n", endTime - startTime);

    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}