################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

INCLUDE_FLAGS  := -I$(CUDA_PATH)/include -I$(SANITIZER_PATH)/include

LINK_FLAGS     := -L$(SANITIZER_PATH) -fPIC -shared
LINK_LIBS      := -lsanitizer-public

NVCC_FLAGS     := --fatbin --compile-as-tools-patch
NVCC_FLAGS     += $(INCLUDE_FLAGS)

ifeq ($(dbg),1)
    NVCC_FLAGS += -g -G
endif

################################################################################

# architecture
TARGET_ARCH   := $(shell uname -m)

ifeq ($(TARGET_ARCH),aarch64)
    SMS        ?= 53 61 70 72 75 80 86 87 90
else
    SMS        ?= 52 60 70 75 80 86 90
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM     := $(lastword $(sort $(SMS)))
GENCODE_FLAGS  += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

################################################################################

# Target rules
all: build

build: libDeviceMalloc.so DeviceMallocPatches.fatbin

libDeviceMalloc.so: DeviceMalloc.cpp
	$(HOST_COMPILER) $(INCLUDE_FLAGS) $(LINK_FLAGS) -o $@ $< $(LINK_LIBS)

DeviceMallocPatches.fatbin: DeviceMallocPatches.cu
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -f libDeviceMalloc.so DeviceMallocPatches.fatbin

clobber: clean

