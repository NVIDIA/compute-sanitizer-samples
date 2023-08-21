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

static __device__ __inline__
void FlushData(EventTracker& tracker)
{
    // make sure everything is visible in memory
    __threadfence_system();

    tracker.doorbell = true;

    while (tracker.doorbell)
    {
    }

    tracker.numEvents = 0;
    __threadfence();
    tracker.currentIndex = 0;
}

static __device__ __inline__
uint32_t GetEventIndex(EventTracker& tracker)
{
    uint32_t idx = kMaxEvents;

    while (idx >= kMaxEvents)
    {
        idx = atomicAdd(&tracker.currentIndex, 1);

        if (idx >= kMaxEvents)
        {
            // buffer is full, wait for last writing thread to flush
            do
            {
            }
            while (*(volatile uint32_t*)&tracker.currentIndex >= kMaxEvents);
        }
    }

    return idx;
}

static __device__ inline
void IncrementNumEvents(EventTracker& tracker)
{
    __threadfence();
    const uint32_t old = atomicAdd(&tracker.numEvents, 1);

    if (old == kMaxEvents - 1)
    {
        // buffer is full, require a flush
        FlushData(tracker);
    }
}

extern "C" __device__ __noinline__
SanitizerPatchResult DeviceMalloc(
    void* userdata,
    uint64_t pc,
    void* allocatedPtr,
    uint64_t allocatedSize)
{
    auto& tracker = *(EventTracker*)userdata;

    const uint32_t idx = GetEventIndex(tracker);

    EventData& event = tracker.events[idx];
    event.instructionId = SANITIZER_INSTRUCTION_DEVICE_SIDE_MALLOC;
    event.address = (uint64_t)(uintptr_t)allocatedPtr;
    event.size = allocatedSize;

    IncrementNumEvents(tracker);

    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult DeviceFree(
    void* userdata,
    uint64_t pc,
    void* ptr)
{
    auto& tracker = *(EventTracker*)userdata;

    const uint32_t idx = GetEventIndex(tracker);

    EventData& event = tracker.events[idx];
    event.instructionId = SANITIZER_INSTRUCTION_DEVICE_SIDE_FREE;
    event.address = (uint64_t)(uintptr_t)ptr;

    IncrementNumEvents(tracker);

    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult AlignedMalloc(
    void* userdata,
    uint64_t pc,
    void* allocatedPtr,
    uint64_t allocatedSize)
{
    auto& tracker = *(EventTracker*)userdata;

    const uint32_t idx = GetEventIndex(tracker);

    EventData& event = tracker.events[idx];
    event.instructionId = SANITIZER_INSTRUCTION_DEVICE_ALIGNED_MALLOC;
    event.address = (uint64_t)(uintptr_t)allocatedPtr;
    event.size = allocatedSize;

    IncrementNumEvents(tracker);

    return SANITIZER_PATCH_SUCCESS;
}
