################################################################################
# Copyright (c) 2021-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH      ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer
SANITIZER_BIN  ?= $(SANITIZER_PATH)/compute-sanitizer

HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDE_FLAGS  := -I$(CUDA_PATH)/include

COMPILE_FLAGS  := -MMD
ifeq ($(dbg),1)
    COMPILE_FLAGS  += -G -Xcompiler -rdynamic
endif

# Pattern rules to build binary from .cu file
%.o: %.cu
	$(NVCC) $(INCLUDE_FLAGS) $(COMPILE_FLAGS) -c $<
%: %.o
	$(NVCC) $(LINK_FLAGS) -o $@ $<

# Target rules
TARGET_BINARY   := memcheck_demo

all: build

build: $(TARGET_BINARY)

clean:
	rm -f $(TARGET_BINARY) memcheck_demo.o

clobber: clean

# Run sanitizer tools
run_memcheck: $(TARGET_BINARY)
	$(SANITIZER_BIN) --destroy-on-device-error kernel $(TARGET_BINARY)

run_leakcheck: $(TARGET_BINARY)
	$(SANITIZER_BIN) --destroy-on-device-error kernel --leak-check=full $(TARGET_BINARY)
