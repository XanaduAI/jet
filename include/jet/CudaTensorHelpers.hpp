#pragma once
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cuComplex.h>
#include <cuda.h>
#include <curand.h>
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
 * @brief Macro that throws Exception from CuRand failure error codes.
 *
 * @param err CuRand function error-code.
 */
#define JET_CURAND_IS_SUCCESS(err)                                             \
    JET_ABORT_IF_NOT(err == CURAND_STATUS_SUCCESS, GetCuRandErrorString(err))

static const std::string GetCuRandErrorString(const curandStatus_t &err)
{
    static const std::unordered_map<curandStatus_t, std::string> error_map(
        {{CURAND_STATUS_SUCCESS, "No errors"},
         {CURAND_STATUS_VERSION_MISMATCH,
          "Header file and linked library version do not match"},
         {CURAND_STATUS_NOT_INITIALIZED, "Generator not initialized"},
         {CURAND_STATUS_ALLOCATION_FAILED, "Memory allocation failed"},
         {CURAND_STATUS_TYPE_ERROR, "Generator is wrong type"},
         {CURAND_STATUS_OUT_OF_RANGE, "Argument out of range"},
         {CURAND_STATUS_LENGTH_NOT_MULTIPLE,
          "Length requested is not a multple of dimension"},
         {CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
          "GPU does not have double precision required by MRG32k3a"},
         {CURAND_STATUS_LAUNCH_FAILURE, "Kernel launch failure"},
         {CURAND_STATUS_PREEXISTING_FAILURE,
          "Preexisting failure on library entry"},
         {CURAND_STATUS_INITIALIZATION_FAILED, "Initialization of CUDA failed"},
         {CURAND_STATUS_ARCH_MISMATCH,
          "Architecture mismatch, GPU does not support requested "
          "featurecurandStatus_t"},
         {CURAND_STATUS_INTERNAL_ERROR, "Internal library error"}});
    return error_map.at(err);
}

/**
 * @brief Calculate the strides for each dimension for the CUDA array.
 *
 * @param extents Vector of the size for each dimension.
 * @return std::vector<int64_t> Memory strides for each dimension.
 */
static inline std::vector<int64_t>
GetStrides(const std::vector<size_t> &extents)
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
template <class DataType>
static inline std::vector<DataType>
ReverseVector(const std::vector<DataType> &input)
{
    return std::vector<DataType>{input.rbegin(), input.rend()};
}

} // namespace CudaTensorHelpers
} // namespace Jet
