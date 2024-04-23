/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//tensor shit
#include <cublas_v2.h>
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int blockSize) {
    extern __shared__ float sharedMem[];  // Dynamic shared memory
    float* As = sharedMem;
    float* Bs = &sharedMem[blockSize * blockSize];
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Dynamic shared memory allocation
  extern __shared__ float sharedMem[];
  float* As = sharedMem;
  float* Bs = &sharedMem[BLOCK_SIZE * BLOCK_SIZE];

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty * BLOCK_SIZE + tx] = A[a + wA * ty + tx];
    Bs[ty * BLOCK_SIZE + tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, const dim3 &dimsA, const dim3 &dimsB) {
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    // Initialize host memory
    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int mem_size_C = dimsB.x * dimsA.y * sizeof(float);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // Allocate host matrix C
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    // Create multiple streams
    cudaStream_t stream1, stream2, computeStream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    checkCudaErrors(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

    // Copy host memory to device asynchronously on separate streams
    checkCudaErrors(
        cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream1));
    checkCudaErrors(
        cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream2));

    // Initialize cuBLAS context and set the compute stream
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(handle, computeStream);

    // Ensure data transfers are complete before starting computation
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Test various block sizes to find the best one
    int blockSizes[] = {16, 32, 64, 128, 256, 512, 1024};
    int numBlockSizes = sizeof(blockSizes) / sizeof(int);
    float bestPerf = 0.0f;
    int bestBlockSize = block_size;  // Start with the initial block_size passed to the function

    for (int i = 0; i < numBlockSizes; ++i) {
        int testBlockSize = blockSizes[i];
        dim3 threads(testBlockSize, testBlockSize);
        dim3 grid((dimsB.x + threads.x - 1) / threads.x, (dimsA.y + threads.y - 1) / threads.y);
        size_t sharedMemSize = 2 * testBlockSize * testBlockSize * sizeof(float);

        // Check the shared memory constraints are not exceeded
        if (testBlockSize * testBlockSize <= 1024) {
            // Execute the kernel with the current block size
            cudaEvent_t start, stop;
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
            checkCudaErrors(cudaEventRecord(start, computeStream));

            MatrixMulCUDA<<<grid, threads, sharedMemSize, computeStream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x, testBlockSize);

            checkCudaErrors(cudaEventRecord(stop, computeStream));
            checkCudaErrors(cudaEventSynchronize(stop));

            float msecTotal = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

            // Compute and print the performance
            float msecPerMatrixMul = msecTotal;
            double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
            double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
            
            if (gigaFlops > bestPerf) {
                bestPerf = gigaFlops;
                bestBlockSize = testBlockSize;
            }

            checkCudaErrors(cudaEventDestroy(start));
            checkCudaErrors(cudaEventDestroy(stop));
        }
    }

    // Output the best performance and corresponding block size
    printf("Best block size: %d\nBest performance: %.2f GFlop/s\n", bestBlockSize, bestPerf);

    // Copy result from device to host on computeStream
    checkCudaErrors(
        cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, computeStream));
    checkCudaErrors(cudaStreamSynchronize(computeStream));

    // Validate and print results
    printf("Checking computed result for correctness: ");
    bool correct = true;
    double eps = 1.e-6;  // machine zero
    for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory and resources
    cublasDestroy(handle);
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaStreamDestroy(computeStream));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}




/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  // width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
