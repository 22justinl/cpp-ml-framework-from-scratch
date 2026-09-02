# C++ ML Framework
Run `mkdir build` first

Build: `./scripts/build_release.sh`

Run tests: `./scripts/run_tests.sh`

Run XOR MLP example: `./build/xor`

Run MNIST Digit MLP example: `./build/mnist_digit_mlp`

Time matrix multiplication: `time ./build/matmul4096`

## Overview

<img width="2646" height="1687" alt="C++ ML Framework Overview" src="https://github.com/user-attachments/assets/7ff22f60-2d19-473e-a4ca-3bcfb24e9f79" />

### Matrix Multiplication Algorithm Overview
<img width="3639" height="1899" alt="BLAS GEMM Overview" src="https://github.com/user-attachments/assets/13573f74-1a7a-41b8-94c3-9b7ef436a4d6" />

## Matrix Multiplication Rough Benchmarks

Average time taken for 5 trials of multiplication of two 4096x4096 matrices.

Naive triple loop implementation with no optimizations: 38.8692s

|Kernel Type|Auto vectorization*|Time (s)|
|-|-|-|
|scalar|false|9.3688|
|scalar|true|3.9184|
|SSE (4x4)|false|7.3904|
|SSE (4x4)|true|7.1604|
|SSE (4x8)|false|6.7326|
|SSE (4x8)|true|6.9728|
|SSE (8x4)|false|6.929|
|SSE (8x4)|true|6.0934|
|AVX (8x8)|false|5.2458|
|AVX2 + FMA (8x8)|false|3.6972|
|AVX2 + FMA (8x8)|true|3.8048|
|AVX2 + FMA (16x8)|false|5.8852|
|AVX2 + FMA (16x8)|true|5.6234|
|**AVX2 + FMA (4x16)**|**false**|**3.095**|
|AVX2 + FMA (4x16)|true|3.1862|

*autovectorization with compiler flags: `-march=x86-64 -O3 -mavx2 -mfma`

** using untuned parameters: $M_c=N_c=K_c=128$

Best kernel: AVX2 + FMA (4x16)

Best time after tuning parameters: 2.7848s

Best parameters: $M_c=512, N_c=1024, K_c=256, M_r=4, N_r=16$

## Tasks
- [x] Add, sub, mul, div, matmul, etc.
- [x] Autograd
- [x] Math functions
- [x] Reduction operations
- [x] Activation functions
- [x] Loss functions
- [x] Broadcasting
- [x] Neural network abstraction
- [x] Optimizers
- [x] Tensor data storage
- [x] View, reshape
- [x] Slicing
- [x] Data ingestion
- [x] Train MLP on MNIST digits
- [ ] Storing/loading models
- [x] Matmul cache blocking
- [x] Naive GEMM Implementation
- [x] Matrix packing
- [x] SIMD Microkernels
- [x] Tuning GEMM parameters
- [x] kernels for matmul, scaled_matmul, mmadd, mmadd_general (GEMM)
- [x] nD GEMM functions
- [x] nD GEMM functions broadcasting
- [ ] **Tensor iteration overhaul (using ideas from PyTorch TensorIterator)**
- [ ] TensorIterator optimization (coalescing, reordering dimensions, etc.)
- [ ] Refactor all operations to use TensorIterator
- [ ] Matmul multithreading
- [ ] Initial convolutional layer implementation
- [ ] Train CNN
- [ ] Optimize convolutional layer
