
#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("Jet m=10 CPU Benchmark"
    VERSION 0.1.0
    DESCRIPTION "Jet project CPU benchmarks"
    LANGUAGES CXX
)

#############################
## II. Fetch Jet project
#############################

Include(FetchContent)

FetchContent_Declare(
    Jet
    GIT_REPOSITORY  git@github.com:XanaduAI/jet.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(Jet)

#############################
## III. Create project target
#############################

add_executable(jet_sliced jet_sliced.cpp)
target_link_libraries(jet_sliced Jet)

add_executable(jet_full jet_full.cpp)
target_link_libraries(jet_full Jet)
