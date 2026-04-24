# As simple as it gets
cmake_minimum_required(VERSION 3.18)
project(shadow_reduction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Project properties
set(SOURCES 
    src/HelperFunctions.cu
    src/ColorSpaceTransformation.cu
    src/OtsusMethod.cu
    src/Convolution.cu
    src/Erosion.cu
    src/ResultIntegration.cu
    src/main.cu 
)

# Prefer modern CMake's CUDAToolkit package
find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)