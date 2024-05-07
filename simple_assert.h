#pragma once

#include <cstdio>

#define ASSERT(condition, error)                          \
    {                                                     \
        if (!(condition))                                 \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("reason: %s\n", (error));              \
            exit(1);                                      \
        }                                                 \
    }