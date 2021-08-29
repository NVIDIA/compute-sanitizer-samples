# NVTX Suballocation sample

Sample demonstrating how to use NVTX Suballocation API:
* `nvtxMemHeapRegister`
* `nvtxMemHeapUnregister`
* `nvtxMemHeapReset`
* `nvtxMemRegionsRegister`
* `nvtxMemRegionsResize`
* `nvtxMemRegionsUnregister`

```
$ make --quiet
$ compute-sanitizer --nvtx=yes --destroy-on-device-error=kernel --show-backtrace=no ./NvtxMemoryPool
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 1 bytes
=========     at 0x60 in Iota(unsigned char*)
=========     by thread (63,0,0) in block (0,0,0)
=========     Address 0x7f004700004f is out of bounds
=========
========= Invalid __global__ write of size 1 bytes
=========     at 0x60 in Iota(unsigned char*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f0047000010 is out of bounds
=========
========= Invalid __global__ write of size 1 bytes
=========     at 0x60 in Iota(unsigned char*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f0047000010 is out of bounds
=========
========= ERROR SUMMARY: 3 errors
```
