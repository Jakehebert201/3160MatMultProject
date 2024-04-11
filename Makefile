# Makefile for CUDA and C project

# Compiler settings - Can be customized.
NVCC = nvcc
GCC = gcc
NVCCFLAGS = -I/opt/cuda-11.7/include -L/opt/cuda-11.7/lib64 -lcudart
GCCFLAGS = -I/opt/cuda-11.7/include
LDFLAGS = -L/opt/cuda-11.7/lib64 -lcudart
CUDASRC = $(wildcard *.cu)
CSRC = $(wildcard *.c)
CUDA_OBJ = $(CUDASRC:.cu=.o)
C_OBJ = $(CSRC:.c=.o)
EXEC = cudamatmult

# Makefile rules start here

all: $(EXEC)

$(EXEC): $(CUDA_OBJ) $(C_OBJ)
	$(GCC) $(GCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.c
	$(GCC) $(GCCFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) $(CUDA_OBJ) $(C_OBJ)

.PHONY: all clean

