# NVTX Suballocation sample

Sample demonstrating how to use NVTX Permissions API (basic):
* `nvtxMemPermissionsAssign`
* `nvtxMemCudaGetProcessWidePermissions`

```
$ make --quiet
$ compute-sanitizer --nvtx=yes --destroy-on-device-error=kernel --show-backtrace=no ./NvtxPermissions
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x90 in IncrementTwice(unsigned int*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f6f67000000 is not writable
=========
========= Invalid __global__ read of size 4 bytes
=========     at 0x20 in IncrementTwice(unsigned int*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f6f67000000 is not readable
=========
========= Invalid __global__ read of size 4 bytes
=========     at 0x20 in IncrementTwice(unsigned int*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f6f67000000 is not readable
=========
========= Invalid __global__ atomic of size 4 bytes
=========     at 0xa0 in IncrementTwice(unsigned int*)
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f6f67000000 is not readable/writable with atomic operations
=========
========= ERROR SUMMARY: 4 errors
```
