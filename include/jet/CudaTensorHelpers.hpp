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

#ifndef CUDATENSOR_UNSAFE

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
    JET_ABORT_IF_NOT(err == CURAND_STATUS_SUCCESS,                             \
                     GetCuRandErrorString(err).c_str())

#else
#define JET_CUDA_IS_SUCCESS(err)                                               \
    {                                                                          \
        static_cast<void>(err);                                                \
    }
#define JET_CUTENSOR_IS_SUCCESS(err)                                           \
    {                                                                          \
        static_cast<void>(err);                                                \
    }
#define JET_CURAND_IS_SUCCESS(err)                                             \
    {                                                                          \
        static_cast<void>(err);                                                \
    }
#endif

static const std::string GetCuRandErrorString(const curandStatus_t &err)
{
    std::string result;
    switch (err) {
    case CURAND_STATUS_SUCCESS:
        result = "No errors";
        break;
    case CURAND_STATUS_VERSION_MISMATCH:
        result = "Header file and linked library version do not match";
        break;
    case CURAND_STATUS_NOT_INITIALIZED:
        result = "Generator not initialized";
        break;
    case CURAND_STATUS_ALLOCATION_FAILED:
        result = "Memory allocation failed";
        break;
    case CURAND_STATUS_TYPE_ERROR:
        result = "Generator is wrong type";
        break;
    case CURAND_STATUS_OUT_OF_RANGE:
        result = "Argument out of range";
        break;
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        result = "Length requested is not a multple of dimension";
        break;
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        result = "GPU does not have double precision required by MRG32k3a";
        break;
    case CURAND_STATUS_LAUNCH_FAILURE:
        result = "Kernel launch failure";
        break;
    case CURAND_STATUS_PREEXISTING_FAILURE:
        result = "Preexisting failure on library entry";
        break;
    case CURAND_STATUS_INITIALIZATION_FAILED:
        result = "Initialization of CUDA failed";
        break;
    case CURAND_STATUS_ARCH_MISMATCH:
        result = "Architecture mismatch, GPU does not support requested "
                 "feature curandStatus_t";
        break;
    case CURAND_STATUS_INTERNAL_ERROR:
        result = "Internal library error";
        break;
    default:
        result = "Status not found";
    }
    return result;
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

/** @class CudaScopedDevice

@brief RAII-styled device context switch. Code taken from Taskflow.

%cudaScopedDevice is neither movable nor copyable.
*/
class CudaScopedDevice {

  public:
    /**
    @brief constructs a RAII-styled device switcher

    @param device device context to scope in the guard
    */
    explicit CudaScopedDevice(int device);

    /**
    @brief destructs the guard and switches back to the previous device context
    */
    ~CudaScopedDevice();

  private:
    CudaScopedDevice() = delete;
    CudaScopedDevice(const CudaScopedDevice &) = delete;
    CudaScopedDevice(CudaScopedDevice &&) = delete;

    int _p;
};

inline CudaScopedDevice::CudaScopedDevice(int dev)
{
    JET_CUDA_IS_SUCCESS(cudaGetDevice(&_p));
    if (_p == dev) {
        _p = -1;
    }
    else {
        JET_CUDA_IS_SUCCESS(cudaSetDevice(dev));
    }
}

inline CudaScopedDevice::~CudaScopedDevice()
{
    if (_p != -1) {
        cudaSetDevice(_p);
    }
}

} // namespace CudaTensorHelpers
} // namespace Jet
