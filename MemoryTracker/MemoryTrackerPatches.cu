/* Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "MemoryTracker.h"

#include <sanitizer_patching.h>

static __device__
SanitizerPatchResult CommonCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    MemoryAccessType type)
{
    auto* pTracker = (MemoryAccessTracker*)userdata;

    uint32_t old = atomicAdd(&(pTracker->currentEntry), 1);

    // no more space!
    if (old >= pTracker->maxEntry)
        return SANITIZER_PATCH_SUCCESS;

    MemoryAccess& access = pTracker->accesses[old];
    access.address = (uint64_t)(uintptr_t)ptr;
    access.accessSize = accessSize;
    access.flags = flags;
    access.threadId = threadIdx;
    access.type = type;

    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryGlobalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Global);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemorySharedAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryLocalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Local);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemcpyAsyncCallback(void* userdata, uint64_t pc, void* src, uint32_t dst, uint32_t accessSize, uint32_t totalShmemSize)
{
    if (src)
    {
        CommonCallback(userdata, pc, src, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_READ, MemoryAccessType::Global);
    }

    return CommonCallback(userdata, pc, (void*)dst, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_WRITE, MemoryAccessType::Shared);
}
