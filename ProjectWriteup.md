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
        MatrixA(128,128), MatrixB(128,128)
        Performance= 452.11 GFlop/s, Time= 0.009 msec, Size= 4194304 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(256,256), MatrixB(256,256)
        Performance= 1959.03 GFlop/s, Time= 0.017 msec, Size= 33554432 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(512,512), MatrixB(512,512)
        Performance= 2175.76 GFlop/s, Time= 0.123 msec, Size= 268435456 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(1024,1024), MatrixB(1024,1024)
        Performance= 2731.52 GFlop/s, Time= 0.786 msec, Size= 2147483648 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(2048,2048), MatrixB(2048,2048)
        Performance= 2844.22 GFlop/s, Time= 6.040 msec, Size= 17179869184 Ops, WorkgroupSize= 1024 threads/block

        MatrixA(4096,4096), MatrixB(4096,4096)
        Performance= 2658.80 GFlop/s, Time= 51.692 msec, Size= 137438953472 Ops, WorkgroupSize= 1024 threads/block
```



We optimized the code:

    Added dynamic memory allocation-
        Allows VRAM to be allocated in a more efficient manner.


### Results of Optimized Code:
    
    
We attempted to utilize Tensor Cores and the lower precision floating point data type TF32 to squeeze more performance out of the program, but unfortunately the implementation we used was insufficient and **didn't** actually increase performance at all.
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
# Python:


# Java:



