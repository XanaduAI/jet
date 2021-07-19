
#############################
## I. Set project details
#############################
cmake_minimum_required(VERSION 3.14)

project("Jet GBS CPU Benchmark"
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
    GIT_TAG         0.2.0-benchmarks
)
FetchContent_MakeAvailable(Jet)

#############################
## III. Create project target
#############################

add_executable(jet_gbs_full jet_gbs_full.cpp)
target_link_libraries(jet_gbs_full Jet)
