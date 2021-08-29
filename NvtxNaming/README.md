# NVTX memory pool sample

Sample demonstrating how to use `nvtxMemRegionsName`.

```
$ make --quiet
$ compute-sanitizer --nvtx=yes --leak-check=full --destroy-on-device-error=kernel --show-backtrace=no ./NvtxNaming
========= COMPUTE-SANITIZER
========= Leaked 1 bytes at 0x7f26c3000000 called My allocation
=========
========= LEAK SUMMARY: 1 bytes leaked in 1 allocations
========= ERROR SUMMARY: 1 error
$ compute-sanitizer --nvtx=yes --tool=initcheck --track-unused-memory=yes --destroy-on-device-error=kernel --show-backtrace=no ./NvtxNaming
========= COMPUTE-SANITIZER
=========  Unused memory in allocation 0x7efc4d000000 called My allocation of size 1
=========     Not written any memory.
=========     100% of allocation were unused.
=========
========= ERROR SUMMARY: 1 error
```
