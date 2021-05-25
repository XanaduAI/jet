# As the name suggests, Test_Main.cpp implements main().
add_executable(runner Test_Main.cpp)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1
)

FetchContent_MakeAvailable(Catch2)

target_link_libraries(runner Catch2::Catch2)

# Required for catch_discover_tests().
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# Modify `ctest` to only run the supported subset of tests.
include(CTest)
include(Catch)

target_sources(runner PRIVATE Test_Abort.cpp
                              Test_PathInfo.cpp
                              Test_Permute.cpp
                              Test_TaskBasedCpuContractor.cpp
                              Test_Tensor.cpp
                              Test_TensorNetwork.cpp
                              Test_TensorNetworkIO.cpp
                              Test_Utilities.cpp)

target_link_libraries(runner Jet)

# Run all tests which are not hidden.
catch_discover_tests(runner)
