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
 * @brief Compile-time binding for BLAS GEMM operation (matrix-matrix product).
 *
 * @tparam ComplexPrecision Precision of complex data (`%complex<float>` or
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
template <typename ComplexPrecision>
constexpr void
gemmBinding(size_t m, size_t n, size_t k, ComplexPrecision alpha,
            ComplexPrecision beta, const ComplexPrecision *A_data,
            const ComplexPrecision *B_data, ComplexPrecision *C_data)
{
    static_assert(
        (std::is_same_v<ComplexPrecision, std::complex<float>> ||
         std::is_same_v<ComplexPrecision, std::complex<double>>),
        "Please use complex<float> or complex<double> for Tensor data");

    if constexpr (std::is_same_v<ComplexPrecision, std::complex<float>>)
        cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), &beta,
                    C_data, std::max(1ul, n));
    else if constexpr (std::is_same_v<ComplexPrecision, std::complex<double>>)
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha,
                    A_data, std::max(1ul, k), B_data, std::max(1ul, n), &beta,
                    C_data, std::max(1ul, n));
};

/**
 * @brief Compile-time binding for BLAS GEMV operation (matrix-vector product).
 *
 * @tparam ComplexPrecision Precision of complex data (`%complex<float>` or
 * `%complex<double>`)
 * @param m Number of rows in matrix A
 * @param k Number of columns in matrix A
 * @param alpha Scalar multiplier of A*B
 * @param beta Scalar multiplier for existing C vector data
 * @param A_data Complex data matrix A
 * @param B_data Complex data vector B
 * @param C_data Output data vector
 */
template <typename ComplexPrecision>
constexpr void
gemvBinding(size_t m, size_t k, ComplexPrecision alpha, ComplexPrecision beta,
            const ComplexPrecision *A_data, const ComplexPrecision *B_data,
            ComplexPrecision *C_data)
{
    static_assert(
        (std::is_same_v<ComplexPrecision, std::complex<float>> ||
         std::is_same_v<ComplexPrecision, std::complex<double>>),
        "Please use complex<float> or complex<double> for Tensor data");

    if constexpr (std::is_same_v<ComplexPrecision, std::complex<float>>)
        cblas_cgemv(CblasRowMajor, CblasNoTrans, m, k, (&alpha), (A_data),
                    std::max(1ul, k), (B_data), 1, (&beta), (C_data), 1);
    else if constexpr (std::is_same_v<ComplexPrecision, std::complex<double>>)
        cblas_zgemv(CblasRowMajor, CblasNoTrans, m, k, (&alpha), (A_data),
                    std::max(1ul, k), (B_data), 1, (&beta), (C_data), 1);
};

/**
 * @brief Compile-time binding for BLAS DOTU operation (vector-vector dot
 * product), C=A*B.
 *
 * @tparam ComplexPrecision Precision of complex data (`%complex<float>` or
 * `%complex<double>`)
 * @param k Number of elements in vector-vector dot-product
 * @param A_data Left vector in dot product
 * @param B_data Right vector in dot product
 * @param C_data Output vector from dot product
 */
template <typename ComplexPrecision>
constexpr void dotuBinding(size_t k, const ComplexPrecision *A_data,
                           const ComplexPrecision *B_data,
                           ComplexPrecision *C_data)
{
    static_assert(
        (std::is_same_v<ComplexPrecision, std::complex<float>> ||
         std::is_same_v<ComplexPrecision, std::complex<double>>),
        "Please use complex<float> or complex<double> for Tensor data");

    if constexpr (std::is_same_v<ComplexPrecision, std::complex<float>>)
        cblas_cdotu_sub(k, (A_data), 1, (B_data), 1, (C_data));
    else if constexpr (std::is_same_v<ComplexPrecision, std::complex<double>>)
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
template <typename ComplexPrecision>
inline void MultiplyTensorData(const std::vector<ComplexPrecision> &A,
                               const std::vector<ComplexPrecision> &B,
                               std::vector<ComplexPrecision> &C,
                               const std::vector<std::string> &left_indices,
                               const std::vector<std::string> &right_indices,
                               size_t left_dim, size_t right_dim,
                               size_t common_dim)
{
    ComplexPrecision alpha{1.0, 0.0};
    ComplexPrecision beta{0.0, 0.0};

    auto A_data = A.data();
    auto B_data = B.data();
    auto C_data = C.data();

    // Multiply. Four cases: MxM, Mxv, vxM, vxv.
    if (left_indices.size() > 0 && right_indices.size() > 0) {
        size_t m = left_dim;
        size_t n = right_dim;
        size_t k = common_dim;
        gemmBinding<ComplexPrecision>(m, n, k, alpha, beta, A_data, B_data,
                                      C_data);
    }
    else if (left_indices.size() > 0 && right_indices.size() == 0) {
        size_t m = left_dim;
        size_t k = common_dim;
        gemvBinding(m, k, alpha, beta, A_data, B_data, C_data);
    }
    else if (left_indices.size() == 0 && right_indices.size() > 0) {
        size_t n = right_dim;
        size_t k = common_dim;
        gemvBinding(k, n, alpha, beta, B_data, A_data, C_data);
    }
    else if (left_indices.size() == 0 && right_indices.size() == 0) {
        size_t k = common_dim;
        dotuBinding(k, A_data, B_data, C_data);
    }
}

/**
 * @brief Calulate the size of data from the tensor size.
 *
 * @param tensor_shape Size of each tensor index label.
 */
inline size_t ShapeToSize(const std::vector<size_t> &tensor_shape)
{
    size_t total_dim = 1;
    for (const auto &dim : tensor_shape)
        total_dim *= dim;
    return total_dim;
}

}; // namespace TensorHelpers
}; // namespace Jet
