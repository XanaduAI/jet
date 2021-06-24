
#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("Jet m=12 Sliced Benchmark"
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

add_executable(jet_m12_single jet_sliced_single.cpp)
target_link_libraries(jet_m12_single Jet)

add_executable(jet_m12_all jet_sliced_all.cpp)
target_link_libraries(jet_m12_all Jet)

add_executable(jet_m12_subset jet_sliced_subset.cpp)
target_link_libraries(jet_m12_subset Jet)