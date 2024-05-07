#include <windows.h>
#include <cmath>

#include "timer.h"
#include "simple_assert.h"

int main(int argc, char **argv)
{
    struct timespec tstart;
    cpu_timer_start(&tstart);
    Sleep(1000);
    double interval = cpu_timer_stop(tstart);

    ASSERT(fabs(interval - 1) < 1e-1, "timer not working properly");

    return 0;
}