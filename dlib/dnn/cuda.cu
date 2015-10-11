
#include <stdlib.h>
#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

void hello_cuda()
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());

#ifndef DLIB_USE_CUDA
#error why is this not defined?
#endif

    auto x = 4;
}

