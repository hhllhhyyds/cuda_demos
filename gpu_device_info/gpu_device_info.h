#pragma once

#include <cuda_runtime.h>

struct cudaDeviceProp get_device_prop(int device);

int get_device_attribute(int device, enum cudaDeviceAttr attr);

size_t device_core_clock(int device);

size_t device_memory_clock(int device);

size_t device_memory_bus_width(int device);

size_t device_multiprocessor_count(int device);

size_t device_cores_per_multiprocessor(int device);

size_t device_peek_flops(int device);

// unit is bytes/s
size_t device_peek_memory_bandwidth(int device);

float device_instructions_bytes_ratio(int device);
