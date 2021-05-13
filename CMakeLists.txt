##########################
## Set Project version
##########################
cmake_minimum_required(VERSION 3.14)
set(JET_LOGO "
      ▄▄     ▄▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄▄ 
      ██     ██▀▀▀▀▀▀    ▀▀▀██▀▀▀ 
      ██     ██             ██    
      ██     ███████        ██    
      ██     ██             ██    
█▄▄▄▄▄██     ██▄▄▄▄▄▄       ██    
 ▀▀▀▀▀       ▀▀▀▀▀▀▀▀       ▀▀    
")
message(${JET_LOGO})

project("Jet"
        VERSION 0.1.0
        DESCRIPTION "A task-based tensor network contraction engine"
        LANGUAGES CXX C
)

##########################
## Set Default Options
##########################
# Compiler options
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)
option(ENABLE_WARNINGS "Enable warnings" ON)
option(ENABLE_OPENMP "Enable OpenMP if supported" OFF)
option(ENABLE_NATIVE "Enable native build tuning" OFF)
option(ENABLE_IPO "Enable interprocedural/link-time optimisation" OFF)

# Build options
option(BUILD_PYTHON "Generate Python bindings" OFF)
option(BUILD_TESTS "Build tests" OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Default build type: Release" FORCE)
endif()

##########################
## Enfore Compiler Support
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nJet requires g++ at least v7.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nJet requires clang++ at least v5.0")
  endif() 
## AppleClang
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.0")
    message(FATAL_ERROR "\nJet requires AppleClang at least v8.0")
  endif()
option(USING_APPLECLANG "AppleClang" On)
## microsoft visual c++
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0.1")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nJet requires icpc at least v19.0.1")
  endif()
else()
  message(FATAL_ERROR "\nJet does not support the selected ${CMAKE_CXX_COMPILER_ID} compiler.")
endif()

##########################
## Include BLAS modules
##########################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(MKL QUIET)

if(MKL_FOUND)
    add_definitions("-DENABLE_MKL")
    set(BLAS_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
    set(BLAS_LIBRARIES ${MKL_LIBRARY})
else()
    find_package(CBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CBLAS_LIBRARIES})
    set(BLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIRS})
endif()


##########################
## Fetch dependencies
##########################
Include(FetchContent)

FetchContent_Declare(
  Taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v3.1.0
)

# Don't build the Taskflow examples or tests.
set(TF_BUILD_EXAMPLES OFF CACHE INTERNAL "Build Taskflow examples")
set(TF_BUILD_TESTS OFF CACHE INTERNAL "Build Taskflow tests")

FetchContent_MakeAvailable(Taskflow)

if(BUILD_PYTHON)
  FetchContent_Declare(
    Pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
  )
  FetchContent_MakeAvailable(Pybind11)
endif()

find_package(OpenMP QUIET)

##########################
## Create Jet target
##########################

add_library(Jet INTERFACE)
target_include_directories(Jet INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> 
    ${BLAS_INCLUDE_DIRS}
)

target_link_libraries(Jet INTERFACE ${BLAS_LIBRARIES} Taskflow)

##########################
## Compile options
##########################

if (ENABLE_OPENMP AND OPENMP_FOUND)
    target_link_libraries(Jet INTERFACE OpenMP::OpenMP_CXX)
elseif (ENABLE_OPENMP AND NOT OPENMP_FOUND)
    message(FATAL_ERROR "\nOpenMP is enabled but could not be found")
endif()

if(ENABLE_SANITIZERS)
    target_compile_options(Jet INTERFACE -g -fsanitize=address,undefined)
    target_link_options(Jet INTERFACE -fsanitize=address,undefined)
endif()

if(ENABLE_WARNINGS)
    target_compile_options(Jet INTERFACE -Wall -Wextra -Werror)
endif()

if(ENABLE_NATIVE)
    target_compile_options(Jet INTERFACE -march=native)
endif()

if(ENABLE_IPO)
    target_compile_options(Jet INTERFACE -flto)
endif()

##########################
## Report
##########################

message(STATUS "BLAS_INCLUDE_DIRS: ${BLAS_INCLUDE_DIRS}")
message(STATUS "BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
message(STATUS "BUILD_PYTHON: ${BUILD_PYTHON}")
message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "ENABLE_IPO: ${ENABLE_IPO}")
message(STATUS "ENABLE_NATIVE: ${ENABLE_NATIVE}")
message(STATUS "ENABLE_SANITIZERS: ${ENABLE_SANITIZERS}")
message(STATUS "ENABLE_WARNINGS: ${ENABLE_WARNINGS}")

##########################
## Build targets
##########################

if(BUILD_PYTHON)
  pybind11_add_module(jet python/src/Python.cpp)
  target_link_libraries(jet PRIVATE Jet)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif(BUILD_TESTS)
