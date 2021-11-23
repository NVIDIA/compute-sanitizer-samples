/* Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <sanitizer.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

struct LaunchData
{
    std::string functionName;
    MemoryAccessTracker* pTracker;
};

using LaunchVector = std::vector<LaunchData>;
using StreamMap = std::map<Sanitizer_StreamHandle, LaunchVector>;
using ContextMap = std::map<CUcontext, StreamMap>;

struct CallbackTracker
{
    std::ostream* out = nullptr;
    std::shared_ptr<std::ofstream> outFile = nullptr;

    ContextMap memoryTrackers;

    CallbackTracker()
    {
        const char *pOutName = std::getenv("OUT_FILE_NAME");
        if (pOutName)
        {
            outFile = std::make_shared<std::ofstream>(pOutName);
            out = outFile.get();
        }
        else
        {
            out = &std::cout;
        }
    }

    // very basic singleton
    static CallbackTracker& GetInstance()
    {
        static CallbackTracker instance;
        return instance;
    }
};

void ModuleLoaded(Sanitizer_ResourceModuleData* pModuleData)
{
    // Instrument user code!
    sanitizerAddPatchesFromFile("MemoryTrackerPatches.fatbin", 0);
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, pModuleData->module, "MemoryGlobalAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, pModuleData->module, "MemorySharedAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, pModuleData->module, "MemoryLocalAccessCallback");
    sanitizerPatchModule(pModuleData->module);
}

static size_t GetMemAccessSize()
{
    constexpr size_t MemAccessDefaultSize = 1024;

    const char* pValue = std::getenv("MEM_ACCESS_SIZE");
    if (!pValue)
    {
        return MemAccessDefaultSize;
    }

    return std::stoi(pValue);
}

void LaunchBegin(
    CallbackTracker* pCallbackTracker,
    CUcontext context,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle stream)
{
    const size_t MemAccessSize = GetMemAccessSize();

    // alloc MemoryAccess array
    MemoryAccess* accesses = nullptr;
    sanitizerAlloc(context, (void**)&accesses, sizeof(MemoryAccess) * MemAccessSize);
    sanitizerMemset(accesses, 0, sizeof(MemoryAccess) * MemAccessSize, stream);

    MemoryAccessTracker hTracker;
    hTracker.currentEntry = 0;
    hTracker.maxEntry = MemAccessSize;
    hTracker.accesses = accesses;

    MemoryAccessTracker* dTracker = nullptr;
    sanitizerAlloc(context, (void**)&dTracker, sizeof(*dTracker));
    sanitizerMemcpyHostToDeviceAsync(dTracker, &hTracker, sizeof(*dTracker), stream);

    sanitizerSetCallbackData(function, dTracker);

    LaunchData launchData = {functionName, dTracker};
    std::vector<LaunchData>& deviceTrackers = pCallbackTracker->memoryTrackers[context][stream];
    deviceTrackers.push_back(launchData);
}

static std::string GetMemoryRWString(uint32_t flags)
{
    if (flags & (SANITIZER_MEMORY_DEVICE_FLAG_READ | SANITIZER_MEMORY_DEVICE_FLAG_WRITE))
    {
        return "Atomic";
    }
    else if (flags & SANITIZER_MEMORY_DEVICE_FLAG_READ)
    {
        return "Read";
    }
    else if (flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE)
    {
        return "Write";
    }
    else
    {
        return "Unknown";
    }
}

static std::string GetMemoryTypeString(MemoryAccessType type)
{
    if (type == MemoryAccessType::Local)
    {
        return "local";
    }
    else if (type == MemoryAccessType::Shared)
    {
        return  "shared";
    }
    else
    {
        return "global";
    }
}

void StreamSynchronized(
    CallbackTracker* pCallbackTracker,
    CUcontext context,
    Sanitizer_StreamHandle stream)
{
    MemoryAccessTracker hTracker = {0};

    std::vector<LaunchData>& deviceTrackers = pCallbackTracker->memoryTrackers[context][stream];

    for (auto& tracker : deviceTrackers)
    {
        *pCallbackTracker->out << "Kernel Launch: " << tracker.functionName << std::endl;

        sanitizerMemcpyDeviceToHost(&hTracker, tracker.pTracker, sizeof(*tracker.pTracker), stream);

        uint32_t numEntries = std::min(hTracker.currentEntry, hTracker.maxEntry);

        *pCallbackTracker->out << "  Memory accesses: " << numEntries << std::endl;

        std::vector<MemoryAccess> accesses(numEntries);
        sanitizerMemcpyDeviceToHost(accesses.data(), hTracker.accesses, sizeof(MemoryAccess) * numEntries, stream);

        for (uint32_t i = 0; i < numEntries; ++i)
        {
            MemoryAccess& access = accesses[i];

            *pCallbackTracker->out << "  [" << i << "] " << GetMemoryRWString(access.flags)
                << " access of " << GetMemoryTypeString(access.type)
                << " memory by thread (" << access.threadId.x
                << "," << access.threadId.y
                << "," << access.threadId.z
                << ") at address 0x" << std::hex << access.address << std::dec
                << " (size is " << access.accessSize << " bytes)" << std::endl;
        }

        sanitizerFree(context, hTracker.accesses);
        sanitizerFree(context, tracker.pTracker);
    }

    deviceTrackers.clear();
}

void ContextSynchronized(CallbackTracker* pCallbackTracker, CUcontext context)
{
    auto& contextTracker = pCallbackTracker->memoryTrackers[context];

    for (auto& streamTracker : contextTracker)
    {
        StreamSynchronized(pCallbackTracker, context, streamTracker.first);
    }
}

void MemoryTrackerCallback(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
    auto* callbackTracker = (CallbackTracker*)userdata;

    switch (domain)
    {
        case SANITIZER_CB_DOMAIN_RESOURCE:
            switch (cbid)
            {
                case SANITIZER_CBID_RESOURCE_MODULE_LOADED:
                {
                    auto* pModuleData = (Sanitizer_ResourceModuleData*)cbdata;
                    ModuleLoaded(pModuleData);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_LAUNCH:
            switch (cbid)
            {
                case SANITIZER_CBID_LAUNCH_BEGIN:
                {
                    auto* pLaunchData = (Sanitizer_LaunchData*)cbdata;
                    LaunchBegin(callbackTracker, pLaunchData->context, pLaunchData->function, pLaunchData->functionName, pLaunchData->hStream);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_SYNCHRONIZE:
            switch (cbid)
            {
                case SANITIZER_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED:
                {
                    auto* pSyncData = (Sanitizer_SynchronizeData*)cbdata;
                    StreamSynchronized(callbackTracker, pSyncData->context, pSyncData->hStream);
                    break;
                }
                case SANITIZER_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED:
                {
                    auto* pSyncData = (Sanitizer_SynchronizeData*)cbdata;
                    ContextSynchronized(callbackTracker, pSyncData->context);
                    break;
                }
                default:
                    break;
            }
            break;
        default:
            break;
    }
}

int InitializeInjection()
{
    Sanitizer_SubscriberHandle handle;
    CallbackTracker& tracker = CallbackTracker::GetInstance();

    sanitizerSubscribe(&handle, MemoryTrackerCallback, &tracker);
    sanitizerEnableAllDomains(1, handle);

    return 0;
}

int __global_initializer__ = InitializeInjection();
