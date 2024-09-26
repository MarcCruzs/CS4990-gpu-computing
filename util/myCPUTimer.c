#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}