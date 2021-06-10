#pragma once

#include <cuda.h>
#include <cutensor.h>

#include "Abort.hpp"
#include "Utilities.hpp"


namespace Jet {
namespace CudaTensorHelpers {

/**
 * @brief Macro that throws Exception from CUDA failure error codes.
 *
 * @param err CUDA function error-code.
 */
#define JET_CUDA_IS_SUCCESS(err) JET_ABORT_IF_NOT(err==cudaSuccess, cudaGetErrorString(err))

/**
 * @brief Macro that throws Exception from CuTensor failure error codes.
 *
 * @param err CuTensor function error-code.
 */
#define JET_CUTENSOR_IS_SUCCESS(err) JET_ABORT_IF_NOT(err==CUTENSOR_STATUS_SUCCESS, cutensorGetErrorString(err))

/**
 * @brief Calculate the strides for each dimension for the CUDA array.
 *
 * @param extents Vector of the size for each dimension.
 * @return std::vector<int64_t> Memory strides for each dimension.
 */
std::vector<int64_t> GetStrides(const std::vector<size_t> &extents)
{
    using namespace Jet::Utilities;

    std::vector<int64_t> strides(std::max(extents.size(), 1UL), 1);
    std::exclusive_scan(extents.begin(), extents.end(), strides.begin(),
                    1, std::multiplies<int64_t>{});

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
    for (size_t k = 0; k < sizes.size(); k++) {
        column_order_linear_index += unraveled_index[k] * strides[k];
    }
    
    return column_order_linear_index;
}

}} // Jet::CudaTensorHelpers
