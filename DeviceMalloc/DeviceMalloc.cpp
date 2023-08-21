/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "DeviceMalloc.h"

#include <cstring>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

void Work(std::future<void> futureObj);

struct Worker
{
    std::thread t;
    std::promise<void> exitSignal;

    Worker()
    {
        std::future<void> futureObj = exitSignal.get_future();
        t = std::thread(&Work, std::move(futureObj));
    }

    ~Worker()
    {
        exitSignal.set_value();
        t.join();
    }
};

Worker worker;
std::vector<EventTracker*> trackers;

void FlushTracker(EventTracker& tracker)
{
    for (size_t i = 0; i < tracker.numEvents; ++i)
    {
        EventData& event = tracker.events[i];

        switch (event.instructionId)
        {
            case SANITIZER_INSTRUCTION_DEVICE_SIDE_MALLOC:
                std::cout << "malloc(" << event.size << ") = 0x";
                std::cout << std::hex << event.address << std::dec << std::endl;
                break;
            case SANITIZER_INSTRUCTION_DEVICE_SIDE_FREE:
                std::cout << "free(0x";
                std::cout << std::hex << event.address << std::dec << ")" << std::endl;
                break;
            case SANITIZER_INSTRUCTION_DEVICE_ALIGNED_MALLOC:
                std::cout << "__nv_aligned_device_malloc(" <<  event.size << ") = 0x";
                std::cout << std::hex << event.address << std::dec << std::endl;
                break;
            default:
                break;
        }
    }
}

void FlushData()
{
    for (auto& pTracker : trackers)
    {
        FlushTracker(*pTracker);
    }
}

void Work(std::future<void> futureObj)
{
    while (futureObj.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout)
    {
        for (auto pTracker : trackers)
        {
            if (!pTracker->doorbell)
            {
                continue;
            }

            FlushTracker(*pTracker);

            pTracker->currentIndex = 0;
            pTracker->numEvents = 0;
            pTracker->doorbell = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void ModuleLoaded(Sanitizer_ResourceModuleData* pModuleData)
{
    // Instrument user code!
    if (SANITIZER_SUCCESS != sanitizerAddPatchesFromFile("DeviceMallocPatches.fatbin", 0))
    {
        std::cerr << "Failed to load fatbin. Please check that it is in the current directory and contains the correct SM architecture" << std::endl;
    }

    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_DEVICE_SIDE_MALLOC, pModuleData->module, "DeviceMalloc");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_DEVICE_SIDE_FREE, pModuleData->module, "DeviceFree");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_DEVICE_ALIGNED_MALLOC, pModuleData->module, "AlignedMalloc");
    sanitizerPatchModule(pModuleData->module);
}

void LaunchBegin(Sanitizer_LaunchData* pLaunchData)
{
    EventTracker* pTracker = nullptr;
    sanitizerAllocHost(pLaunchData->context, (void**)&pTracker, sizeof(EventTracker));
    std::memset(pTracker, 0, sizeof(EventTracker));

    sanitizerSetLaunchCallbackData(
        pLaunchData->hLaunch,
        pLaunchData->function,
        pLaunchData->hStream,
        pTracker);

    // not thread-safe!
    trackers.push_back(pTracker);
}

void cbFunction(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
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
                    LaunchBegin(pLaunchData);
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
                case SANITIZER_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED:
                    FlushData();
                    break;
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

    sanitizerSubscribe(&handle, cbFunction, nullptr);
    sanitizerEnableAllDomains(1, handle);

    return 0;
}

int __global_initializer__ = InitializeInjection();

