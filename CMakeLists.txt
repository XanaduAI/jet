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
option(ENABLE_CUTENSOR "Build with CuTensor GPU support" OFF)
option(DISABLE_CUDA_SAFETY "Build without CUDA call safety checks" OFF)


# Build options
option(BUILD_PYTHON "Generate Python bindings" OFF)
option(BUILD_TESTS "Build tests" OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Default build type: Release" FORCE)
endif()

option(ENABLE_HPTT "Enable support for hptt tensor transpose backend" OFF)

option(WINDOWS_BUNDLE_CBLAS "Fetch and statically link CBLAS during build" OFF)
if(NOT WIN32)
  set(WINDOWS_BUNDLE_CBLAS OFF)
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
## Microsoft Visual c++
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.0.1")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\nJet requires MSVC at least v19.0.1")
  endif()
  set(ENABLE_WARNINGS OFF)
else()
  message(FATAL_ERROR "\nJet does not support the selected ${CMAKE_CXX_COMPILER_ID} compiler.")
endif()


##########################
## Fetch dependencies
##########################
include(FetchContent)

FetchContent_Declare(
  Taskflow
  GIT_REPOSITORY https://github.com/taskflow/taskflow.git
  GIT_TAG        v3.1.0
)

# Don't build the Taskflow examples or tests.
set(TF_BUILD_EXAMPLES OFF CACHE INTERNAL "Build Taskflow examples")
set(TF_BUILD_TESTS OFF CACHE INTERNAL "Build Taskflow tests")

FetchContent_MakeAvailable(Taskflow)

if(ENABLE_HPTT)
  FetchContent_Declare(
    hptt
    GIT_REPOSITORY  https://github.com/springer13/hptt
    GIT_TAG         v1.0.5 
  )
  FetchContent_MakeAvailable(hptt)
endif()

find_package(OpenMP QUIET)



if(ENABLE_CUTENSOR)
    SET(CUDA_SEPARABLE_COMPILATION ON)

    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60)
    endif()

    find_package(CUDA REQUIRED)

    find_library(CURAND_LIB
        NAMES   curand libcurand curand_static libcurand_static
        HINTS   /usr/lib
                /usr/local/cuda
                /usr/local/lib
                /opt
                lib
                lib64
                ${CUDAToolkit_LIBRARY_DIR}
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

    find_library(CUTENSOR_LIB
        NAMES   cutensor.so.1.3.0 libcutensor.so.1.3.0
        HINTS   /usr/lib
                /usr/local/cuda
                /usr/local/lib
                /opt
                lib
                lib64
                ${CUDAToolkit_LIBRARY_DIR}
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ENV LD_LIBRARY_PATH
    )

    find_file(CUTENSOR_INC
        NAMES   cutensor.h
        HINTS   /usr/include
                /usr/local/cuda
                /usr/local/include
                /opt
                include
                ${CUDAToolkit_INCLUDE_DIRS}
                ${CUDA_TOOLKIT_ROOT_DIR}/include
                ENV CPATH
    )

    if(NOT CUTENSOR_LIB OR NOT CUTENSOR_INC)
        message(FATAL_ERROR "\nUnable to find cutensor installation. Please ensure v1.3.0+ is correctly installed.")
    endif()
endif()


##########################
## Include BLAS modules
##########################

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(WINDOWS_BUNDLE_CBLAS)
  FetchContent_Declare(
    OpenBLAS
    URL       https://github.com/xianyi/OpenBLAS/releases/download/v0.3.17/OpenBLAS-0.3.17-x64.zip
    URL_HASH  MD5=1ee19f55bfd46120689cd260e16e7ce6
  )
  FetchContent_Populate(OpenBLAS)

  set(BLAS_INCLUDE_DIRS "${openblas_SOURCE_DIR}/include")
  set(BLAS_LIBRARIES "${openblas_SOURCE_DIR}/lib/libopenblas.a")
else()
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
endif()

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

if(ENABLE_OPENMP AND OPENMP_FOUND)
  target_link_libraries(Jet INTERFACE OpenMP::OpenMP_CXX)
elseif (ENABLE_OPENMP AND NOT OPENMP_FOUND)
  message(FATAL_ERROR "\nOpenMP is enabled but could not be found")
endif()

if(ENABLE_SANITIZERS)
  target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-g;-fsanitize=address,undefined>)
  target_link_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>)
endif()

if(ENABLE_WARNINGS)
  target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
endif()

if(ENABLE_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
        target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>)
        target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>)
    else()
        target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()
if(ENABLE_IPO)
    target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-flto>)
endif()
if(MKL_FOUND)
    target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DENABLE_MKL>)
endif()

if(ENABLE_HPTT)
  target_link_libraries(Jet INTERFACE hptt)
endif()

if (ENABLE_CUTENSOR)
    # Avoid DSO errors on platforms preferring static linkage
    string(REPLACE "libcudart_static.a" "libcudart.so" CUDA_SHARED_RT "${CUDA_LIBRARIES}")

    target_include_directories(Jet INTERFACE ${CUDA_TOOLKIT_ROOT_DIR}/include)
    target_link_libraries(Jet INTERFACE ${CUTENSOR_LIB} ${CURAND_LIB} ${CUDA_SHARED_RT})
    target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUTENSOR>)
    if(DISABLE_CUDA_SAFETY)
        target_compile_options(Jet INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-DCUDATENSOR_UNSAFE>)
    endif()
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
  add_subdirectory(python)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif(BUILD_TESTS)
