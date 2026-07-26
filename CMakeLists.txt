cmake_minimum_required(VERSION 4.0)

project(cpp_ml_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#  NOTE: clangd shows error
set(CXXFLAGS "-march=x86-64 -mavx2 -mfma") # show auto vectorizations: -Rpass=loop-vectorize
# set(CXXFLAGS "-march=x86-64 -mavx2 -mfma -Rpass=loop-vectorize")
# set(CXXFLAGS "-march=x86-64 -mavx2 -mfma -fno-vectorize -fno-slp-vectorize")
set(CMAKE_CXX_FLAGS "${CXXFLAGS}")

# Main library
add_library(cpp_ml_from_scratch
    src/core/tensor.cpp
    src/core/broadcast.cpp

    src/autograd/math_ops.cpp
    src/autograd/reduction_ops.cpp
    src/autograd/activation.cpp
    src/autograd/tensor_ops.cpp

    src/kernels/math_ops.cpp
    src/kernels/reduction_ops.cpp
    src/kernels/activation.cpp
    src/kernels/tensor_ops.cpp
    src/kernels/matmul.cpp

    src/ops/math_ops.cpp
    src/ops/reduction_ops.cpp
    src/ops/activation.cpp
    src/ops/loss.cpp
    src/ops/tensor_ops.cpp

    src/nn/module.cpp
    src/nn/parameter.cpp
    src/nn/linear.cpp
    src/nn/sequential.cpp
    src/nn/activation.cpp
    src/nn/loss.cpp
    src/nn/flatten.cpp

    src/utils/tensor_utils.cpp
    src/utils/nn_utils.cpp
    src/utils/train_utils.cpp

    src/optim/optimizer.cpp
    src/optim/sgd.cpp

    src/data/data.cpp
)

target_include_directories(cpp_ml_from_scratch PUBLIC
    src
)

# Tests
add_executable(tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_broadcast.cpp

    tests/test_math_ops.cpp
    tests/test_math_ops_autograd.cpp

    tests/test_reduction_ops.cpp
    tests/test_reduction_ops_autograd.cpp

    tests/test_activation.cpp
    tests/test_activation_autograd.cpp

    tests/test_loss.cpp
    tests/test_loss_autograd.cpp

    tests/test_tensor_ops.cpp
    tests/test_tensor_ops_autograd.cpp

    tests/test_nn.cpp
)

target_include_directories(tests PRIVATE
    external/doctest
)

target_link_libraries(tests PRIVATE
    cpp_ml_from_scratch
)

add_executable(curr_test
    tests/test_main.cpp
    tests/curr_test.cpp
)
target_include_directories(curr_test PRIVATE
    external/doctest
)

target_link_libraries(curr_test PRIVATE
    cpp_ml_from_scratch
)

add_executable(xor
    examples/xor.cpp
)
target_link_libraries(xor PRIVATE
    cpp_ml_from_scratch
)

add_executable(mnist_digit_mlp
    examples/mnist_digit_mlp.cpp
)
target_link_libraries(mnist_digit_mlp PRIVATE
    cpp_ml_from_scratch
)

add_executable(matmul1024
    examples/matmul1024.cpp
)
target_link_libraries(matmul1024 PRIVATE
    cpp_ml_from_scratch
)
add_executable(matmul2048
    examples/matmul2048.cpp
)
target_link_libraries(matmul2048 PRIVATE
    cpp_ml_from_scratch
)
add_executable(matmul4096
    examples/matmul4096.cpp
)
target_link_libraries(matmul4096 PRIVATE
    cpp_ml_from_scratch
)
