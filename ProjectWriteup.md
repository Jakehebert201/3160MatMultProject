CSCI 3160 Matrix Multiplication Project Writeup

Team 3: Brendan Dalhover, Deep Desai, Jacob Hebert, James Letterman, Russell Payne

# C: 
>In the C program, we utilized NVIDIA’s CUDA cores to make matrix multiplication as efficient as possible. This involved the use of an RTX 3090, and a lot of electricity.
>
>The CUDA toolkit, which includes the nvcc to compile .cu files can be downloaded from NVIDIA’s website: 
>>https://developer.nvidia.com/cuda-downloads. 
>The test desktop that used the toolkit used version 11.7, as that was what was available on the AUR. The matrix multiplication file is from the Matmult sample in NVIDIA’s cud-samples githup repo:
>>https://github.com/NVIDIA/cuda-samples/tree/master 
>
>Initial tests of CUDA showed that 128, 256, and 512 sized matrices would not be enough to even show up on a results page, so we went with 1024, 2048, and 4096 sized matrices. The complexity of each operation is O(N^3), so each increase in size results in a cubic increase in the number of operations performed. In the default code, an 8196 sized matrix takes half a second to complete, but about 2 minutes to output to the console.
>
>Each run of the matrix multiplication is done 300 times and then averaged together. 


### Results of the original code:
```c
        MatrixA(64,64), MatrixB(64,64)
        Performance= 95.58 GFlop/s, Time= 0.005 msec, Size= 524288 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(128,128), MatrixB(128,128)
        Performance= 453.45 GFlop/s, Time= 0.009 msec, Size= 4194304 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(256,256), MatrixB(256,256)
        Performance= 1628.63 GFlop/s, Time= 0.021 msec, Size= 33554432 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(512,512), MatrixB(512,512)
        Performance= 1733.29 GFlop/s, Time= 0.155 msec, Size= 268435456 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(1024,1024), MatrixB(1024,1024)
        Performance= 2249.34 GFlop/s, Time= 0.955 msec, Size= 2147483648 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(2048,2048), MatrixB(2048,2048)
        Performance= 2261.73 GFlop/s, Time= 7.596 msec, Size= 17179869184 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(4096,4096), MatrixB(4096,4096)
        Performance= 2123.39 GFlop/s, Time= 64.726 msec, Size= 137438953472 Ops, WorkgroupSize= 1024 threads/block

```

#### Optimizations:

    -Added dynamic memory allocation
        Allows VRAM to be allocated in a more efficient manner.
    -Increased Block Size

```c
          for (int j = 0; j < nIter; j++) {
    if (block_size == 16) {
      MatrixMulCUDA<16>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
      MatrixMulCUDA<32>
          <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }
  }
```
This block of code determines the block size for multiplications, increasing it further allows for higher performance with beefier GPUs



    
We attempted to utilize Tensor Cores and the lower precision floating point data type TF32 to squeeze more performance out of the program and it was very successful with larger operations!
    To do this we used the cuBLAS library, which uses single-precision point multiplication and the GPU's Tensor Cores.
### Results of the TF32 code:
```c

        MatrixA(64,64), MatrixB(64,64)
        Performance= 96.06 GFlop/s, Time= 0.005 msec, Size= 524288 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(128,128), MatrixB(128,128)
        Performance= 453.26 GFlop/s, Time= 0.009 msec, Size= 4194304 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(256,256), MatrixB(256,256)
        Performance= 1959.81 GFlop/s, Time= 0.017 msec, Size= 33554432 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(512,512), MatrixB(512,512)
        Performance= 2134.14 GFlop/s, Time= 0.126 msec, Size= 268435456 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(1024,1024), MatrixB(1024,1024)
        Performance= 2595.00 GFlop/s, Time= 0.828 msec, Size= 2147483648 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(2048,2048), MatrixB(2048,2048)
        Performance= 2781.55 GFlop/s, Time= 6.176 msec, Size= 17179869184 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(4096,4096), MatrixB(4096,4096)
        Performance= 2536.09 GFlop/s, Time= 54.193 msec, Size= 137438953472 Ops, WorkgroupSize= 1024 threads/block
```

### Results of Optimized Code:
```c
        MatrixA(128,128), MatrixB(128,128)
        Performance= 455.96 GFlop/s, Time= 0.009 msec, Size= 4194304 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(256,256), MatrixB(256,256)
        Performance= 1956.30 GFlop/s, Time= 0.017 msec, Size= 33554432 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(512,512), MatrixB(512,512)
        Performance= 2199.87 GFlop/s, Time= 0.122 msec, Size= 268435456 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(1024,1024), MatrixB(1024,1024)
        Performance= 2904.07 GFlop/s, Time= 0.739 msec, Size= 2147483648 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(2048,2048), MatrixB(2048,2048)
        Performance= 2872.19 GFlop/s, Time= 5.981 msec, Size= 17179869184 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(4096,4096), MatrixB(4096,4096)
        Performance= 2688.05 GFlop/s, Time= 51.130 msec, Size= 137438953472 Ops, WorkgroupSize= 1024 threads/block


```
This implementation is in "matrixMul.cu" in the C folder, all the previous iterations are in the "Unusedmatmults" folder.
This version does not use the TF32, it just messes with block size.
Overall, we improved the base speed by 25.5%!
# Python:


# Java:



