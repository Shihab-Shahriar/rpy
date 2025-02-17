cmake_minimum_required(VERSION 3.22)
project(MyKokkosProgram LANGUAGES CXX CUDA)

# Use C++14 (or adjust as needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architecture (compute capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Optional: If you want to enable extended lambda support for CUDA,
# you can add the flag below:
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Find the installed Kokkos package (ensure Kokkos is installed with GPU support)
find_package(Kokkos REQUIRED)

# Add your executable (replace main.cpp with your source file)
add_executable(bench_rpy benchmark.cpp)

# Link Kokkos to your executable
target_link_libraries(bench_rpy PRIVATE Kokkos::kokkos)
