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

#pragma once

#if !defined(__cplusplus)
#error "C++ only header. Please use a C++ compiler."
#endif

#include <nvtx3/nvToolsExtMem.h>
#include <nvtx3/nvToolsExtMemCudaRt.h>

#include <cstddef>

namespace NV {

    enum Permissions
    {
        PERMISSIONS_NONE = NVTX_MEM_PERMISSIONS_REGION_FLAGS_NONE,
        PERMISSIONS_READ = NVTX_MEM_PERMISSIONS_REGION_FLAGS_READ,
        PERMISSIONS_WRITE = NVTX_MEM_PERMISSIONS_REGION_FLAGS_WRITE,
        PERMISSIONS_ATOMIC = NVTX_MEM_PERMISSIONS_REGION_FLAGS_ATOMIC,
        PERMISSIONS_RESET = NVTX_MEM_PERMISSIONS_REGION_FLAGS_RESET
    };

    void PermissionsAssign(nvtxDomainHandle_t nvtxDomain, const void* ptr, uint32_t flags, nvtxMemPermissionsHandle_t handle)
    {
        nvtxMemPermissionsAssignRegionDesc_t nvtxPermDesc;
        nvtxPermDesc.flags = flags;
        nvtxPermDesc.regionRefType = NVTX_MEM_REGION_REF_TYPE_POINTER;
        nvtxPermDesc.region.pointer = ptr;

        nvtxMemPermissionsAssignBatch_t nvtxRegionsDesc = {};
        nvtxRegionsDesc.extCompatID = NVTX_EXT_COMPATID_MEM;
        nvtxRegionsDesc.structSize = sizeof(nvtxRegionsDesc);
        nvtxRegionsDesc.permissions = handle;
        nvtxRegionsDesc.regionCount = 1;
        nvtxRegionsDesc.regionElementSize = sizeof(nvtxPermDesc);
        nvtxRegionsDesc.regionElements = &nvtxPermDesc;

        nvtxMemPermissionsAssign(nvtxDomain, &nvtxRegionsDesc);
    }

    void PermissionsAssign(nvtxDomainHandle_t nvtxDomain, const void* ptr, uint32_t flags)
    {
        auto processPermHandle = nvtxMemCudaGetProcessWidePermissions(nvtxDomain);

        PermissionsAssign(nvtxDomain, ptr, flags, processPermHandle);
    }

} // namespace NV
