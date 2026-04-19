cmake_minimum_required(VERSION 3.10)
project(MatrixCUDA)

# Находим CUDA
find_package(CUDA REQUIRED)

# Добавляем исполняемый файл
cuda_add_executable(lab4_p main.cu)

# Оптимизации
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -O3")