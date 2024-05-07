#include "gpu_device_info.h"
#include "simple_assert.h"

struct cudaDeviceProp get_device_prop(int device)
{
    struct cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device));
    return device_prop;
}

int get_device_attribute(int device, enum cudaDeviceAttr attr)
{
    int attr_value;
    CUDA_CHECK(cudaDeviceGetAttribute(&attr_value, attr, device));
    return attr_value;
}

size_t device_core_clock(int device)
{
    size_t clock;
    int attr = get_device_attribute(device, cudaDeviceAttr::cudaDevAttrClockRate);
    clock = static_cast<size_t>(attr) * 1000;
    return clock;
}

size_t device_memory_clock(int device)
{
    size_t clock;
    int attr = get_device_attribute(device, cudaDeviceAttr::cudaDevAttrMemoryClockRate);
    clock = static_cast<size_t>(attr) * 1000;
    return clock;
}

size_t device_memory_bus_width(int device)
{
    size_t bus_width;
    int attr = get_device_attribute(device, cudaDeviceAttr::cudaDevAttrGlobalMemoryBusWidth);
    bus_width = static_cast<size_t>(attr);
    return bus_width;
}

size_t device_multiprocessor_count(int device)
{
    size_t count;
    int attr = get_device_attribute(device, cudaDeviceAttr::cudaDevAttrMultiProcessorCount);
    count = static_cast<size_t>(attr);
    return count;
}

size_t device_cores_per_multiprocessor(int device)
{
    struct cudaDeviceProp dev_prop = get_device_prop(device);

    size_t cores = 0;
    bool device_unknown = false;

    switch (dev_prop.major)
    {
    case 2: // Fermi
        if (dev_prop.minor == 1)
            cores = 48;
        else
            cores = 32;
        break;
    case 3: // Kepler
        cores = 192;
        break;
    case 5: // Maxwell
        cores = 128;
        break;
    case 6: // Pascal
        if ((dev_prop.minor == 1) || (dev_prop.minor == 2))
            cores = 128;
        else if (dev_prop.minor == 0)
            cores = 64;
        else
            device_unknown = true;
        break;
    case 7: // Volta and Turing
        if ((dev_prop.minor == 0) || (dev_prop.minor == 5))
            cores = 64;
        else
            device_unknown = true;
        break;
    case 8: // Ampere
        if (dev_prop.minor == 0)
            cores = 64;
        else if (dev_prop.minor == 6)
            cores = 128;
        else if (dev_prop.minor == 9)
            cores = 128; // ada lovelace
        else
            device_unknown = true;
        break;
    case 9: // Hopper
        if (dev_prop.minor == 0)
            cores = 128;
        else
            device_unknown = true;
        break;
    default:
        device_unknown = true;
        break;
    }
    ASSERT(!device_unknown, "Unknown device type");
    return cores;
}

size_t device_peek_flops(int device)
{
    size_t clock_rate = device_core_clock(device);
    size_t mp = device_multiprocessor_count(device);
    size_t cores_per_mp = device_cores_per_multiprocessor(device);
    size_t flops = clock_rate * mp * cores_per_mp;
    return flops;
}

// unit is bytes/s
size_t device_peek_memory_bandwidth(int device)
{
    return device_memory_clock(device) * device_memory_bus_width(device) / 8;
}

float device_instructions_bytes_ratio(int device)
{
    float flops = static_cast<float>(device_peek_flops(device));
    float bandwidth = static_cast<float>(device_peek_memory_bandwidth(device));
    return flops / bandwidth;
}
