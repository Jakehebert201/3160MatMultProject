#ifndef CUDA_MATMULT_H
#define CUDA_MATMULT_H

#include <cuda_runtime.h>

struct fn_args {
    int n;
    int *mem1;      // Host memory pointer for the first matrix
    int *mem2;      // Host memory pointer for the second matrix
    int *mem3;      // Host memory pointer for the result matrix
    unsigned long long fac;
    int *d_mem1;    // Device memory pointer for the first matrix
    int *d_mem2;    // Device memory pointer for the second matrix
    int *d_mem3;    // Device memory pointer for the result matrix
};

// Function to perform matrix multiplication using CUDA
void matrix_multiply_cuda(struct fn_args *args);

// Function to initialize matrices using CUDA
void matrix_initialize_cuda(struct fn_args *args);

// Function to transpose a matrix using CUDA
void matrix_transpose_cuda(int *src, int *dst, int n);

// CUDA kernel for matrix multiplication
__global__ void matrix_mult_kernel(const int *d_mat1, const int *d_mat2, int *d_res, int n);

// Functions to allocate and free device memory
void allocate_device_memory(struct fn_args *args);
void free_device_memory(struct fn_args *args);

// Functions to copy data between host and device
void copy_host_to_device(struct fn_args *args);
void copy_device_to_host(struct fn_args *args);

#endif // CUDA_MATMULT_H

