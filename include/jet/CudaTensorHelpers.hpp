#pragma once

#include <cuda.h>
#include <cutensor.h>

#include "Abort.hpp"
#include "Utilities.hpp"

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

/**
 * @brief Calculate the strides for each dimension for the CUDA array.
 *
 * @param extents Vector of the size for each dimension.
 * @return std::vector<int64_t> Memory strides for each dimension.
 */
std::vector<int64_t> GetStrides(const std::vector<size_t> &extents)
{
    std::vector<int64_t> strides(std::max(extents.size(), 1UL), 1);
    for (int64_t i = 1; i < static_cast<int64_t>(extents.size()); ++i) {
        strides[i] = static_cast<int64_t>(extents[i - 1]) * strides[i - 1];
    }
    return strides;
}

/**
 * @brief Convertor between row-major and column-major indices.
 *
 * @param row_order_linear_index Lexicographic ordered data index.
 * @param sizes The size of each independent dimension of the tensor data.
 * @return size_t Single index mapped to column-major (colexicographic) form.
 */
size_t RowMajToColMaj(size_t row_order_linear_index,
                      const std::vector<size_t> &sizes)
{
    using namespace Jet::Utilities;
    auto unraveled_index = UnravelIndex(row_order_linear_index, sizes);

    auto strides = GetStrides(sizes);

    size_t column_order_linear_index = 0;
    int d = sizes.size();
    for (int k = 0; k < d; k++) {
        column_order_linear_index += unraveled_index[k] * strides[k];
    }
    return column_order_linear_index;
}

}} // Jet::CudaTensorHelpers
