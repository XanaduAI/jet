#############################
cmake_minimum_required(VERSION 3.14)

project("Jet GPU m=10 Benchmark"
    VERSION 0.1.0
    DESCRIPTION "Jet project benchmarks using GPU for m=10"
    LANGUAGES CXX
)

#############################
## II. Fetch Jet project
#############################

Include(FetchContent)

FetchContent_Declare(
    Jet
    GIT_REPOSITORY  git@github.com:XanaduAI/jet.git
    GIT_TAG         0.2.1
)
FetchContent_MakeAvailable(Jet)

#############################
## III. Create project target
#############################

add_executable(jet_cuda_sliced jet_sliced.cpp)
target_link_libraries(jet_cuda_sliced Jet)

add_executable(jet_cuda_full jet_full.cpp)
target_link_libraries(jet_cuda_full Jet)
