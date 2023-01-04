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

static constexpr int NumThreads = 64;
static constexpr int DataBlocks = 16;
static constexpr int Size = (NumThreads * DataBlocks) - 16;

__shared__ int smem[NumThreads];

__global__
void myKernel(int *data_in, int *sum_out)
{
    int tx = threadIdx.x;

    smem[tx] = 0;

    __syncthreads();

    for (int b = 0; b < DataBlocks; ++b)
    {
        const int offset = NumThreads * b + tx;
        if (offset < Size)
        {
            smem[tx] += data_in[offset];
            __syncthreads();
        }
    }

    if (tx == 0)
    {
        *sum_out = 0;
        for (int i = 0; i < NumThreads; ++i)
        {
            *sum_out += smem[i];
        }
    }
}

int main()
{
    int *data_in = nullptr;
    int *sum_out = nullptr;

    checkCudaErrors(cudaMalloc((void**)&data_in, Size * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&sum_out, sizeof(int)));

    checkCudaLaunch(myKernel<<<1, NumThreads>>>(data_in, sum_out));
    cudaDeviceSynchronize();

    cudaFree(data_in);
    cudaFree(sum_out);
    return 0;
}
