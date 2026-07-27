# C++ ML Framework
Run `mkdir build` first

Build: `./scripts/build.sh`

Run tests: `./scripts/run_tests.sh`

Run XOR MLP example: `./build/xor`

Run MNIST Digit MLP example: `./build/mnist_digit_mlp`

Time matrix multiplication: `time ./build/matmul4096`

## Overview
<img width="1188" height="760" alt="C++ ML Framework Overview" src="https://github.com/user-attachments/assets/4aa35986-9a33-4291-91d0-1bf0fbd9522c" />

### Matrix Multiplication Algorithm Overview
<img width="1410" height="747" alt="BLAS GEMM Overview" src="https://github.com/user-attachments/assets/ef01881a-c47a-4c2f-bc42-828eb3b16c95" />

## Matrix Multiplication Rough Benchmarks

Average time taken for 5 trials of multiplication of two 4096x4096 matrices.

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

Best kernel: AVX2 + FMA (4x16)

Best time after tuning block sizes: 2.7848s

Best parameters: $M_c=512, N_c=1024, K_c=256, M_r=4, N_r=16$
