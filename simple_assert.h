#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define ASSERT(condition, error)                          \
    {                                                     \
        if (!(condition))                                 \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("reason: %s\n", (error));              \
            exit(1);                                      \
        }                                                 \
    }

#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                           \
        }                                                                      \
    }