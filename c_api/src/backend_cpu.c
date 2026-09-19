/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * CPU backend: not implemented yet. anbcCreateDevice(ANBC_DEVICE_BACKEND_CPU)
 * returns NULL until it is.
 */

#include "anbc_internal.h"

anbcResult anbcBackendCpuInit(anbcDevice* device)
{
    (void)device;
    return ANBC_ERROR_UNSUPPORTED;
}

#ifndef __APPLE__
anbcResult anbcBackendMetalInit(anbcDevice* device)
{
    (void)device;
    return ANBC_ERROR_UNSUPPORTED;
}
#endif
