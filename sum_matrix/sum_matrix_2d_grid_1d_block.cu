#include <cstdio>

#include "simple_assert.h"
#include "gpu_device_info.h"
#include "timer.h"

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC,
                                  int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    // set up device
    int dev = 0;
    printf("Using Device %d: %s\n", dev, get_device_prop(dev).name);
    CUDA_CHECK(cudaSetDevice(dev));
    // set up date size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    // initialize data at host side
    struct timespec tstart;
    cpu_timer_start(&tstart);
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    printf("Initial data, Time elapsed %f sec\n", cpu_timer_stop(tstart));
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    cpu_timer_start(&tstart);
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    printf("sumMatrixOnHost, Time elapsed %f sec\n", cpu_timer_stop(tstart));

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cpu_timer_start(&tstart);
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);
    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);
    printf("malloc and memcpy, Time elapsed %f sec\n", cpu_timer_stop(tstart));
    // invoke kernel at host side
    dim3 block(1024);
    dim3 grid((nx + block.x - 1) / block.x, ny);
    cpu_timer_start(&tstart);
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
           grid.y, block.x, block.y, cpu_timer_stop(tstart));

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    // check device results
    checkResult(hostRef, gpuRef, nxy);
    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    // reset device
    cudaDeviceReset();
    return 0;
}