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

#include <cassert>

#define checkCudaErrors(Code) assert((Code) == cudaSuccess)
#define checkCudaLaunch(...) checkCudaErrors((__VA_ARGS__, cudaPeekAtLastError()))

static constexpr int Warps = 2;
static constexpr int WarpSize = 32;
static constexpr int NumThreads = Warps * WarpSize;

__shared__ int smem_first[NumThreads];
__shared__ int smem_second[Warps];

__global__
void sumKernel(int *data_in, int *sum_out)
{
    int tx = threadIdx.x;
    smem_first[tx] = data_in[tx] + tx;

    if (tx % WarpSize == 0)
    {
        int wx = tx / WarpSize;

        smem_second[wx] = 0;

        // Avoid loop unrolling for the purpose of racecheck demo
        #pragma unroll 1
        for (int i = 0; i < WarpSize; ++i)
        {
            smem_second[wx] += smem_first[wx * WarpSize + i];
        }
    }

    __syncthreads();

    if (tx == 0)
    {
        *sum_out = 0;
        for (int i = 0; i < Warps; ++i)
        {
            *sum_out += smem_second[i];
        }
    }
}

int main()
{
    int *data_in = nullptr;
    int *sum_out = nullptr;

    checkCudaErrors(cudaMalloc((void**)&data_in, sizeof(int) * NumThreads));
    checkCudaErrors(cudaMalloc((void**)&sum_out, sizeof(int)));
    checkCudaErrors(cudaMemset(data_in, 0, sizeof(int) * NumThreads));

    checkCudaLaunch(sumKernel<<<1, NumThreads>>>(data_in, sum_out));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(data_in));
    checkCudaErrors(cudaFree(sum_out));
    return 0;
}
