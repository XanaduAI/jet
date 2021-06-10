#pragma once

#include <cuda.h>
#include <cutensor.h>

#include "Abort.hpp"

namespace Jet {
namespace CudaTensorHelpers {

/**
 * @brief Throws Exception from CUDA error codes
 *
 * @param err CUDA function error-code
 */
inline void ThrowCudaError(cudaError_t &err)
{
    if (err != cudaSuccess) {
        throw Jet::Exception(std::string(cudaGetErrorString(err)));
    }
}

/**
 * @brief Throws Exception from CuTensor error codes
 *
 * @param err CuTensor function error-code
 */
inline void ThrowCuTensorError(cutensorStatus_t &err)
{
    if (err != CUTENSOR_STATUS_SUCCESS) {
        throw Jet::Exception(std::string(cutensorGetErrorString(err)));
    }
}


}} // Jet::CudaTensorHelpers
