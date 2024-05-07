#include <cstdio>

#include "gpu_device_info.h"

int main(int argc, char **argv)
{
    int device = 0;

    float tflops = static_cast<float>(device_peek_flops(device)) / 1E12;
    float bandwidth = static_cast<float>(device_peek_memory_bandwidth(device)) / 1E9;
    float ratio = device_instructions_bytes_ratio(device);

    printf("Using Device %d: %s\n", device, get_device_prop(device).name);
    printf("peek flops: %.2f TFLOPS\n", tflops);
    printf("peek bandwidth: %.2f GB/s\n", bandwidth);
    printf("instructions bytes ratio: %.2f instruction/bytes\n", ratio);

    return 0;
}