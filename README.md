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

### Optimizations:

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

There's a line of code towards the end that allows for manual adjustment
```c
    int block_size = 1024; //by default 32

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
```




    
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
This implementation had four iterations: Base Python, Slightly Optimized Base Python, Numpy, Slightly Optimized Numpy. Below are the outputs of each of the tests in csvs format. 

```
STRING_LABEL,Size,A Value,B Value,Execution Time (milliseconds)
INTERPRETER_UNOPT,128,1,1,99.80201721191406
INTERPRETER_UNOPT,256,1,1,804.1632175445557
INTERPRETER_UNOPT,512,1,1,7192.05117225647
INTERPRETER_UNOPT,1024,1,1,64153.44786643982
INTERPRETER_UNOPT,2048,1,1,536228.2063961029
```

```
STRING_LABEL,Size,A Value,B Value,Execution Time (milliseconds)
INTERPRETER_OPT,128,1,1,69.48280334472656
INTERPRETER_OPT,256,1,1,519.0598964691162
INTERPRETER_OPT,512,1,1,4933.332920074463
INTERPRETER_OPT,1024,1,1,44637.27378845215
INTERPRETER_OPT,2048,1,1,377366.8270111084
```

```
STRING_LABEL,Size,A Value,B Value,Execution Time (milliseconds)
NUMPY_UNOPT,128,1,1.0,1.5020370483398438
NUMPY_UNOPT,256,1,1.0,2.515077590942383
NUMPY_UNOPT,512,1,1.0,2.000093460083008
NUMPY_UNOPT,1024,1,1.0,6.001710891723633
NUMPY_UNOPT,2048,1,1.0,37.08314895629883
```

```
STRING_LABEL,Size,A Value,B Value,Execution Time (milliseconds)
NUMPY_OPT,128,1,1.0,0.5004405975341797
NUMPY_OPT,256,1,1.0,0.49996376037597656
NUMPY_OPT,512,1,1.0,0.9996891021728516
NUMPY_OPT,1024,1,1.0,5.082130432128906
NUMPY_OPT,2048,1,1.0,29.539108276367188
```
Also with Python, we combined all other tests and displayed sizes, execution times, etc. using MathPlotLib, Tkinter, Tabulate. 
All of the python files with a '1' end, output a csv which the combine_results.py take in to display the times as graphs and tables.
All python files not ending in '1' independently run displaying a table and graph with different A and B values. 

# Java:
This implementation had a naive matrix multiplication aswell as a loop unrolled version to be more optimized. I focused on JIT to test how effective it can be at different ranges.

