#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("CudaTensor"
        VERSION 0.1.0
        DESCRIPTION "A sample Jet project"
        LANGUAGES CXX CUDA
)

SET(ENABLE_CUTENSOR ON CACHE BOOL "Enable CuTensor support")

#############################
## II. Use Jet parent dir
#############################

add_subdirectory(../.. BINDIR)

#############################
## III. Create project target
#############################

add_executable(cudatensor_contract CudaTensor_Contract.cpp)
target_link_libraries(cudatensor_contract Jet)