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

#include "NvtxMemoryPool.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#define checkCudaErrors(Code) assert((Code) == cudaSuccess)
#define checkCudaLaunch(...) checkCudaErrors((__VA_ARGS__, cudaPeekAtLastError()))

__global__ void Iota(uint8_t* v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    v[i] = static_cast<uint8_t>(i);
}

int main(void)
{
    constexpr size_t PoolSize = 4096 * sizeof(uint8_t);
    constexpr size_t NumThreads = 63;
    constexpr size_t AllocSize = NumThreads * sizeof(uint8_t);

    auto nvtxDomain = nvtxDomainCreateA("my-domain");
    void *pool;
    checkCudaErrors(cudaMalloc(&pool, PoolSize));

    {
        // Suballocator object creation (c.f. NvtxMemoryPool.h)
        auto suballocator = NV::Suballocator(nvtxDomain, pool, PoolSize);

        // Create a suballocation of size AllocSize at offset 16
        auto alloc = (uint8_t*)pool + 16;
        suballocator.Register(alloc, AllocSize);

        // Success: allocation is valid
        checkCudaLaunch(Iota<<<1, NumThreads>>>(alloc));
        checkCudaErrors(cudaDeviceSynchronize());

        // Violation: last byte out of bounds
        checkCudaLaunch(Iota<<<1, NumThreads + 1>>>(alloc));
        checkCudaErrors(cudaDeviceSynchronize());

        // Success: resizing
        suballocator.Resize(alloc, AllocSize + 1);
        checkCudaLaunch(Iota<<<1, NumThreads + 1>>>(alloc));
        checkCudaErrors(cudaDeviceSynchronize());

        // Violation: access after free
        suballocator.Unregister(alloc);
        checkCudaLaunch(Iota<<<1, 1>>>(alloc));
        checkCudaErrors(cudaDeviceSynchronize());

        // Violation: access after reset
        suballocator.Register(alloc, AllocSize);
        suballocator.Reset();
        checkCudaLaunch(Iota<<<1, 1>>>(alloc));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaFree(pool));
}
