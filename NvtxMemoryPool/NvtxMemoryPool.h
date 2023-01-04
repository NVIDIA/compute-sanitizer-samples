/* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#if !defined(__cplusplus)
#error "C++ only header. Please use a C++ compiler."
#endif

#include <nvtx3/nvToolsExtMem.h>

#include <cstddef>
#include <stdexcept>

namespace NV {

    // Class designed to handle memory pool management
    class Suballocator
    {
    public:
        Suballocator(nvtxDomainHandle_t nvtxDomain, void *start, size_t capacity)
            : m_nvtxDomain(nvtxDomain), m_start(start), m_capacity(capacity)
        {
            nvtxMemVirtualRangeDesc_t nvtxRangeDesc = {};
            nvtxRangeDesc.size = m_capacity;
            nvtxRangeDesc.ptr = m_start;

            nvtxMemHeapDesc_t nvtxHeapDesc = {};
            nvtxHeapDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
            nvtxHeapDesc.structSize = sizeof(nvtxHeapDesc);
            nvtxHeapDesc.usage = NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR;
            nvtxHeapDesc.type = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
            nvtxHeapDesc.typeSpecificDescSize = sizeof(nvtxRangeDesc);
            nvtxHeapDesc.typeSpecificDesc = &nvtxRangeDesc;

            m_nvtxPool = nvtxMemHeapRegister(
                nvtxDomain,
                &nvtxHeapDesc);

            if (!m_nvtxPool)
            {
                throw std::runtime_error("Memory pool registration failed.");
            }
        }

        ~Suballocator()
        {
            nvtxMemHeapUnregister(m_nvtxDomain, m_nvtxPool);
        }

        void Reset()
        {
            nvtxMemHeapReset(m_nvtxDomain, m_nvtxPool);
        }

        void Register(void *ptr, size_t size)
        {
            nvtxMemVirtualRangeDesc_t nvtxRangeDesc = {};
            nvtxRangeDesc.size = size;
            nvtxRangeDesc.ptr = ptr;

            nvtxMemRegionsRegisterBatch_t nvtxRegionsDesc = {};
            nvtxRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
            nvtxRegionsDesc.structSize = sizeof(nvtxRegionsDesc);
            nvtxRegionsDesc.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
            nvtxRegionsDesc.heap = m_nvtxPool;
            nvtxRegionsDesc.regionCount = 1;
            nvtxRegionsDesc.regionDescElementSize = sizeof(nvtxRangeDesc);
            nvtxRegionsDesc.regionDescElements = &nvtxRangeDesc;

            nvtxMemRegionsRegister(m_nvtxDomain, &nvtxRegionsDesc);
        }

        void Resize(void *ptr, size_t newSize)
        {
            nvtxMemVirtualRangeDesc_t nvtxRangeDesc = {};
            nvtxRangeDesc.size = newSize;
            nvtxRangeDesc.ptr = ptr;

            nvtxMemRegionsResizeBatch_t nvtxRegionsDesc = {};
            nvtxRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
            nvtxRegionsDesc.structSize = sizeof(nvtxRegionsDesc);
            nvtxRegionsDesc.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
            nvtxRegionsDesc.regionDescCount = 1;
            nvtxRegionsDesc.regionDescElementSize = sizeof(nvtxRangeDesc);
            nvtxRegionsDesc.regionDescElements = &nvtxRangeDesc;

            nvtxMemRegionsResize(m_nvtxDomain, &nvtxRegionsDesc);
        }

        void Unregister(void *ptr)
        {
            nvtxMemRegionRef_t nvtxRegionRef;
            nvtxRegionRef.pointer = ptr;

            nvtxMemRegionsUnregisterBatch_t nvtxRegionsDesc = {};
            nvtxRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
            nvtxRegionsDesc.structSize = sizeof(nvtxRegionsDesc);
            nvtxRegionsDesc.refType = NVTX_MEM_REGION_REF_TYPE_POINTER;
            nvtxRegionsDesc.refCount = 1;
            nvtxRegionsDesc.refElementSize = sizeof(nvtxRegionRef);
            nvtxRegionsDesc.refElements = &nvtxRegionRef;

            nvtxMemRegionsUnregister(m_nvtxDomain, &nvtxRegionsDesc);
        }

    private:
        nvtxDomainHandle_t const m_nvtxDomain;
        void* const m_start;
        size_t const m_capacity;
        nvtxMemHeapHandle_t m_nvtxPool;
    };

} // namespace NV
