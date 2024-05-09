#include <cuda_runtime.h>
#include <cstdio>
#include <cublas_v2.h>

#include "simple_assert.h"

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // set up date size of vectors
    int nElem = 1024;
    int stride = 5;
    int nTotal = nElem * stride;

    printf("Vector size %d\n", nTotal);

    // malloc host memory
    size_t nBytes = nTotal * sizeof(float);
    float *h_data;
    h_data = (float *)malloc(nBytes);

    for (int i = 0; i < nTotal; i++)
    {
        h_data[i] = i;
    }

    float host_res[5];
    float gpu_res[5];

    for (int j = 0; j < stride; j++)
    {
        host_res[j] = 0;
        for (int i = 0; i < nElem; i++)
        {
            host_res[j] += h_data[i * stride + j];
        }
    }

    // malloc device global memory
    float *d_data;
    cudaMalloc((float **)&d_data, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_data, h_data, nBytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < stride; i++)
    {

        float res;
        cublasSasum(handle, nElem, d_data + i, stride, &res);
        gpu_res[i] = res;
    }
    cublasDestroy(handle);

    for (int i = 0; i < stride; i++)
    {
        printf("res %d, CPU sum = %f, GPU sum = %f\n", i, host_res[i], gpu_res[i]);
    }

    for (int i = 0; i < stride; i++)
    {
        ASSERT(fabs(host_res[i] - gpu_res[i]) < 1E-6, "cublas sum error");
    }

    // free device global memory
    cudaFree(d_data);
    // free host memory
    free(h_data);

    return 0;
}