/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NvtxPermissions.h"

#include <nvtx3/nvToolsExtMem.h>
#include <nvtx3/nvToolsExtMemCudaRt.h>

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#define checkCudaErrors(Code) assert((Code) == cudaSuccess)
#define checkCudaLaunch(...) checkCudaErrors((__VA_ARGS__, cudaPeekAtLastError()))

__global__ void IncrementTwice(unsigned int* v)
{
    unsigned int i = *v;
    *v = i + 1u;
    atomicAdd(v, 1u);
}

int main()
{
    auto nvtxDomain = nvtxDomainCreateA("my-domain");

    unsigned int* ptr;
    checkCudaErrors(cudaMalloc((void**)&ptr, sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(ptr, 0, sizeof(unsigned int)));

    // Success: allocation is readable and writable
    checkCudaLaunch(IncrementTwice<<<1, 1>>>(ptr));
    checkCudaErrors(cudaDeviceSynchronize());

    // Violation: 4 bytes written on a read-only allocation
    NV::PermissionsAssign(nvtxDomain, ptr, NV::PERMISSIONS_READ);
    checkCudaLaunch(IncrementTwice<<<1, 1>>>(ptr));
    checkCudaErrors(cudaDeviceSynchronize());

    // Violation: 4 bytes read on a write-only allocation
    NV::PermissionsAssign(nvtxDomain, ptr, NV::PERMISSIONS_WRITE);
    checkCudaLaunch(IncrementTwice<<<1, 1>>>(ptr));
    checkCudaErrors(cudaDeviceSynchronize());

    // Violation: 4 bytes read on a no-permissions allocation
    NV::PermissionsAssign(nvtxDomain, ptr, NV::PERMISSIONS_NONE);
    checkCudaLaunch(IncrementTwice<<<1, 1>>>(ptr));
    checkCudaErrors(cudaDeviceSynchronize());

    // Violation: 4 bytes atomic operation on a no-atomic allocation
    NV::PermissionsAssign(nvtxDomain, ptr, NV::PERMISSIONS_READ | NV::PERMISSIONS_WRITE);
    checkCudaLaunch(IncrementTwice<<<1, 1>>>(ptr));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(ptr));
}
