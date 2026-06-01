cmake_minimum_required(VERSION 4.0)

project(cpp_ml_from_scratch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main library
add_library(cpp_ml_from_scratch
    src/core/tensor.cpp

    src/autograd/math_ops.cpp
    src/autograd/reduction_ops.cpp
    src/autograd/activation.cpp

    src/kernels/math_ops.cpp
    src/kernels/reduction_ops.cpp
    src/kernels/activation.cpp

    src/ops/math_ops.cpp
    src/ops/reduction_ops.cpp
    src/ops/activation.cpp

    src/utils/tensor_utils.cpp
)

target_include_directories(cpp_ml_from_scratch PUBLIC
    src
)

# Tests
add_executable(tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_math_ops.cpp
    tests/test_math_ops_autograd.cpp
    tests/test_reduction_ops.cpp
    tests/test_reduction_ops_autograd.cpp
    tests/test_activation.cpp
    tests/test_activation_autograd.cpp
)

target_include_directories(tests PRIVATE
    external/doctest
)

target_link_libraries(tests PRIVATE
    cpp_ml_from_scratch
)
