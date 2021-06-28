
#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("Jet m=12 Sliced Benchmark"
    VERSION 0.1.0
    DESCRIPTION "Jet project GPU benchmarks"
    LANGUAGES CXX
)

#############################
## II. Fetch Jet project
#############################

Include(FetchContent)

FetchContent_Declare(
    Jet
    GIT_REPOSITORY  git@github.com:XanaduAI/jet.git
    GIT_TAG         tbc_cuda
)
FetchContent_MakeAvailable(Jet)

#############################
## III. Create project target
#############################

add_executable(jet_cuda_m12_single jet_sliced_single.cpp)
target_link_libraries(jet_cuda_m12_single Jet)
