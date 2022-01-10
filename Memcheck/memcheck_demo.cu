/* Copyright (c) 2021-2021, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>

__device__ int x;

__global__ void unaligned_kernel(void)
{
    *(int*) ((char*)&x + 1) = 42;
}

__device__ void out_of_bounds_function(void)
{
    *(int*) 0x87654320 = 42;
}

__global__ void out_of_bounds_kernel(void)
{
    out_of_bounds_function();
}

static void run_unaligned(void)
{
    std::cout << "Running unaligned_kernel: ";
    unaligned_kernel<<<1,1>>>();
    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
}

static void run_out_of_bounds(void)
{
    std::cout << "Running out_of_bounds_kernel: ";
    out_of_bounds_kernel<<<1,1>>>();
    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
}

int main() {
    int *devMem = nullptr;

    std::cout << "Mallocing memory" << std::endl;
    cudaMalloc((void**)&devMem, 1024);

    run_unaligned();
    run_out_of_bounds();

    // Omitted to demo leakcheck
    // cudaFree(devMem);

    return 0;
}
