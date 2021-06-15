#pragma once
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cuComplex.h>
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
#define JET_CUDA_IS_SUCCESS(err)                                               \
    JET_ABORT_IF_NOT(err == cudaSuccess, cudaGetErrorString(err))

/**
 * @brief Macro that throws Exception from CuTensor failure error codes.
 *
 * @param err CuTensor function error-code.
 */
#define JET_CUTENSOR_IS_SUCCESS(err)                                           \
    JET_ABORT_IF_NOT(err == CUTENSOR_STATUS_SUCCESS,                           \
                     cutensorGetErrorString(err))

/**
 * @brief Calculate the strides for each dimension for the CUDA array.
 *
 * @param extents Vector of the size for each dimension.
 * @return std::vector<int64_t> Memory strides for each dimension.
 */
static inline std::vector<int64_t> GetStrides(const std::vector<size_t> &extents)
{
    using namespace Jet::Utilities;

    std::vector<int64_t> strides(std::max(extents.size(), 1UL), 1);
    std::exclusive_scan(extents.begin(), extents.end(), strides.begin(), 1,
                        std::multiplies<int64_t>{});

    return strides;
}

/**
 * @brief Convertor between row-major and column-major indices.
 *
 * @param row_order_linear_index Lexicographic ordered data index.
 * @param sizes The size of each independent dimension of the tensor data.
 * @return size_t Single index mapped to column-major (colexicographic) form.
 */
static inline size_t RowMajToColMaj(size_t row_order_linear_index,
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

/**
 * If T is a supported data type for tensors, this expression will
 * evaluate to `true`. Otherwise, it will evaluate to `false`.
 *
 * Supported data types are `float2`, `double2`, and their aliases.
 *
 * @tparam T candidate data type
 */
template <class T>
constexpr bool is_supported_data_type =
    std::is_same_v<T, cuComplex> || std::is_same_v<T, float2> ||
    std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, double2>;


/**
 * @brief Copy and reverse a given vector.
 * 
 * @tparam DataType Vector underlying data type.
 * @param input Input vector.
 * @return Reversed copy of vector.
 */
template<class DataType>
static inline std::vector<DataType> ReverseVector(const std::vector<DataType> &input)
{
    return std::vector<DataType>{input.rbegin(), input.rend()};
}


} // namespace CudaTensorHelpers
} // namespace Jet
