#pragma once

#include <complex>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Abort.hpp"

#if defined ENABLE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace Jet {
namespace TensorHelpers {

/**
 * If T is a supported data type for tensors, this expression will
 * evaluate to `true`. Otherwise, it will evaluate to `false`.
 *
 * Supported data types are std::complex<float> and std::complex<double>.
 *
 * @tparam T candidate data type
 */
template <class T>
constexpr bool is_supported_data_type =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

/**
 * @brief Compile-time binding for BLAS GEMM operation (matrix-matrix product).
 *
 * @tparam T Data type (`float`, `double`, `%complex<float>` or
 * `%complex<double>`)
 * @param m Number of rows in left matrix A and output matrix C
 * @param n Number of cols in right matrix B and output matrix C
 * @param k Number of cols in left matrix A and rows in right matrix B
 * @param alpha Scalar multiplier of A*B
 * @param beta Scalar multiplier of C pre-existing data
 * @param A_data Left matrix A
 * @param B_data Right matrix B
 * @param C_data Output matrix C
 */
template <typename T>
constexpr void gemmBinding(size_t m, size_t n, size_t k, T alpha, T beta,
                           const T *A_data, const T *B_data, T *C_data)
{
    if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), beta,
                    C_data, std::max(1ul, n));
    else if constexpr (std::is_same_v<T, double>)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), beta,
                    C_data, std::max(1ul, n));
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), &beta,
                    C_data, std::max(1ul, n));
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), &beta,
                    C_data, std::max(1ul, n));
};

/**
 * @brief Compile-time binding for BLAS GEMV operation (matrix-vector product).
 *
 * @tparam T Data type (`float`, `double`, `%complex<float>` or
 * `%complex<double>`)
 * @param m Number of rows in matrix A
 * @param k Number of columns in matrix A
 * @param alpha Scalar multiplier of A*B
 * @param beta Scalar multiplier for existing C vector data
 * @param A_data Complex data matrix A
 * @param B_data Complex data vector B
 * @param C_data Output data vector
 * @param transpose Transpose flag for matrix A
 */
template <typename T>
constexpr void gemvBinding(size_t m, size_t k, T alpha, T beta, const T *A_data,
                           const T *B_data, T *C_data,
                           const CBLAS_TRANSPOSE &transpose)
{
    if constexpr (std::is_same_v<T, float>)
        cblas_sgemv(CblasRowMajor, transpose, m, k, alpha, (A_data),
                    std::max(1ul, k), (B_data), 1, beta, (C_data), 1);
    else if constexpr (std::is_same_v<T, double>)
        cblas_dgemv(CblasRowMajor, transpose, m, k, alpha, (A_data),
                    std::max(1ul, k), (B_data), 1, beta, (C_data), 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cgemv(CblasRowMajor, transpose, m, k, (&alpha), (A_data),
                    std::max(1ul, k), (B_data), 1, (&beta), (C_data), 1);
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zgemv(CblasRowMajor, transpose, m, k, (&alpha), (A_data),
                    std::max(1ul, k), (B_data), 1, (&beta), (C_data), 1);
};

/**
 * @brief Compile-time binding for BLAS DOTU operation (vector-vector dot
 * product), C=A*B.
 * @tparam T Data type (`float`, `double`, `%complex<float>` or
 * `%complex<double>`)
 * @param k Number of elements in vector-vector dot-product
 * @param A_data Left vector in dot product
 * @param B_data Right vector in dot product
 * @param C_data Output vector from dot product
 */
template <typename T>
constexpr void dotuBinding(size_t k, const T *A_data, const T *B_data,
                           T *C_data)
{
    if constexpr (std::is_same_v<T, float>)
        C_data[0] = cblas_sdot(k, (A_data), 1, (B_data), 1);
    else if constexpr (std::is_same_v<T, double>)
        C_data[0] = cblas_ddot(k, (A_data), 1, (B_data), 1);
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        cblas_cdotu_sub(k, (A_data), 1, (B_data), 1, (C_data));
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        cblas_zdotu_sub(k, (A_data), 1, (B_data), 1, (C_data));
};

/**
 * @brief Perform BLAS-enabled tensor multiplication for four distinct cases:
 *  - GEMM: complex matrix-matrix product (M*M)
 *  - GEMV: complex matrix-vector product (M*v, v*M)
 *  - DOTU: complex element-wise product (v*v)
 *
 * @param A Row-major encoded left tensor complex data.
 * @param B Row-major encoded right tensor complex data.
 * @param C Resulting complex tensor data in row-major encoding.
 * @param left_indices Left-tensor indices participating in multiplication.
 * @param right_indices Right-tensor indices participating in multiplication.
 * @param left_dim Rows in left tensor A and resulting tensor C.
 * @param right_dim Columns in right tensor B and resulting tensor C.
 * @param common_dim Rows in left tensor A and columns in right tensor B.
 */
template <typename T, std::enable_if_t<is_supported_data_type<T>, bool> = true>
inline void MultiplyTensorData(const std::vector<T> &A, const std::vector<T> &B,
                               std::vector<T> &C,
                               const std::vector<std::string> &left_indices,
                               const std::vector<std::string> &right_indices,
                               size_t left_dim, size_t right_dim,
                               size_t common_dim)
{
    T alpha(1.0);
    T beta(0.0);

    auto A_data = A.data();
    auto B_data = B.data();
    auto C_data = C.data();

    // Multiply. Four cases: MxM, Mxv, vxM, vxv.
    if (left_indices.size() > 0 && right_indices.size() > 0) {
        size_t m = left_dim;
        size_t n = right_dim;
        size_t k = common_dim;
        gemmBinding<T>(m, n, k, alpha, beta, A_data, B_data, C_data);
    }
    else if (left_indices.size() > 0 && right_indices.size() == 0) {
        size_t m = left_dim;
        size_t k = common_dim;
        gemvBinding(m, k, alpha, beta, A_data, B_data, C_data, CblasNoTrans);
    }
    else if (left_indices.size() == 0 && right_indices.size() > 0) {
        size_t n = right_dim;
        size_t k = common_dim;
        gemvBinding(k, n, alpha, beta, B_data, A_data, C_data, CblasTrans);
    }
    else if (left_indices.size() == 0 && right_indices.size() == 0) {
        size_t k = common_dim;
        dotuBinding(k, A_data, B_data, C_data);
    }
}

}; // namespace TensorHelpers
}; // namespace Jet
