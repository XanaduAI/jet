#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutt.h"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace Jet {

template <typename T> bool InVector(const T &s, const std::vector<T> &v)
{
    if (std::find(v.cbegin(), v.cend(), s) != v.cend())
        return true;
    else
        return false;
}

template <typename T>
std::vector<T> VectorIntersection(const std::vector<T> &v,
                                  const std::vector<T> &w)
{
    std::vector<T> temp;
    for (auto it = v.cbegin(); it != v.cend(); ++it) {
        if (InVector(*it, w))
            temp.push_back(*it);
    }
    return temp;
}

template <typename T>
std::vector<T> VectorUnion(const std::vector<T> &v, const std::vector<T> &w)
{
    std::vector<T> temp(v);
    for (auto it = w.cbegin(); it != w.cend(); ++it) {
        if (!InVector(*it, v))
            temp.push_back(*it);
    }
    return temp;
}

template <typename T>
std::vector<T> VectorSubtraction(const std::vector<T> &v,
                                 const std::vector<T> &w)
{
    std::vector<T> temp;
    for (auto it = v.cbegin(); it != v.cend(); ++it) {
        if (!InVector(*it, w))
            temp.push_back(*it);
    }
    return temp;
}

void CUDAMatrixMultiply(float *A, float *B, float *C, int A_row, int A_col,
                        int B_col)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1;
    const float beta = 0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m = A_row;
    int n = B_col;
    int k = A_col;
    int lda = A_row;
    int ldb = A_col;
    int ldc = A_row;

    cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta,
                C, ldc);
}

void CUDAMatrixMultiply(cuFloatComplex *A, cuFloatComplex *B, cuFloatComplex *C,
                        int A_row, int A_col, int B_col)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cuFloatComplex alpha;
    alpha.x = 1.;
    alpha.y = 0.;
    cuFloatComplex beta;
    beta.x = 0.;
    beta.y = 0.;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int m = A_row;
    int n = B_col;
    int k = A_col;
    int lda = A_row;
    int ldb = A_col;
    int ldc = A_row;

    cublasCgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta,
                C, ldc);
}

template <typename T> class CudaTensor {

  private:
    T *data_;
    size_t n_row_;
    size_t n_col_;

    std::vector<size_t> shape_;
    std::vector<std::string> indices_;
    std::unordered_map<std::string, size_t> ind_to_axes_;
    size_t upper_rank_;

    T *data_transpose_;

    void SetShapeAndUpperRank(const std::vector<size_t> &shape,
                              int upper_rank = -1)

    {
        size_t rank = shape.size();
        size_t n_row = 1;
        size_t n_col = 1;
        if (upper_rank >= 0)
            upper_rank_ = upper_rank;
        else
            upper_rank_ = rank / 2;

        for (size_t i = 0; i < upper_rank_; ++i)
            n_row *= shape[i];
        for (size_t i = upper_rank_; i < rank; ++i)
            n_col *= shape[i];
        shape_ = shape;
        n_row_ = n_row;
        n_col_ = n_col;
        indices_ = std::vector<std::string>(shape.size(), "?");
    }

    void SetMemory(bool store_transpose)
    {
        cudaMalloc(&data_, GetRows() * GetCols() * sizeof(T));
        if (store_transpose) {
            cudaMalloc(&data_transpose_, GetRows() * GetCols() * sizeof(T));
        }
    }

    void SetIndicesShapeAndMemory(const std::vector<size_t> &indices,
                                  const std::vector<size_t> &shape,
                                  bool store_transpose, int upper_rank)
    {

        SetShapeAndUpperRank(shape, upper_rank);
        SetMemory(store_transpose);
        SetIndices(indices);
    }

    CudaTensor() {}
    CudaTensor(const std::vector<size_t> &indices,
               const std::vector<size_t> &shape, bool store_transpose = true,
               int upper_rank = -1)
    {
        SetIndicesShapeAndMemory(indices, shape, store_transpose, upper_rank);
    }

    ~CudaTensor()
    {
        if (data_ != NULL) {
            cudaFree(data_);
        }
        if (data_transpose_ != NULL) {
            cudaFree(data_transpose_);
        }
    }

    std::vector<size_t>
    ConvertIndicesToAxes(const std::vector<std::string> &indices) const
    {
        std::vector<size_t> axes(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            axes[i] = ind_to_axes_.at(indices[i]);
        }
        return axes;
    }

    void SetIndices(const std::vector<std::string> &indices)
    {
        indices_ = indices;
        auto &&shape = GetShape();
        if (shape.size() != indices.size()) {
            // skip all axes that have size 1, we assume these are here
            // for no reason and that the indices are in order of non-size-one
            // axes
            size_t counter = 0;
            for (int i = 0; i < shape.size(); i++) {
                if (shape[i] != 1) {
                    ind_to_axes_.insert(
                        std::pair<std::string, size_t>(indices[counter], i));
                    counter++;
                }
            }
        }
        for (size_t i = 0; i < indices.size(); i++) {
            ind_to_axes_.insert(std::pair<std::string, size_t>(indices[i], i));
        }
    }

    const std::vector<std::string> &GetIndices() const { return indices_; }
    const std::unordered_map<std::string, size_t> &GetIndexToAxesMap() const
    {
        return ind_to_axes_;
    }

    void Clear_()
    {
        if (data_ != NULL) {
            cudaFree(data_);
            data_ = NULL;
        }
        if (data_transpose_ != NULL) {
            cudaFree(data_transpose_);
            data_transpose_ = NULL;
        }
    }

    void Move_(CudaTensor &&other)
    {
        Clear_();
        indices_ = std::move(other.indices_);
        shape_ = std::move(other.shape_);
        ind_to_axes_ = std::move(other.ind_to_axes_);
        data_ = other.data_;
        data_transpose_ = other.data_transpose_;
        n_row_ = other.n_row_;
        n_col_ = other.n_col_;
        upper_rank_ = other.upper_rank_;
        other.data_ = NULL;
        other.data_transpose_ = NULL;
    }

    CudaTensor(CudaTensor &&other) { Move_(std::move(other)); }

    CudaTensor(const CudaTensor &other) = delete;
    void operator=(const CudaTensor &) = delete;

    T *GetData() { return data_; }

    T *GetData() { return data_; }

    T *GetTransposeData() { return data_transpose_; }

    const std::vector<size_t> &GetShape() const { return shape_; }

    size_t GetUpperRank() const { return upper_rank_; }

    size_t GetSize() { return n_row_ * n_col_; }

    size_t GetRows() { return n_row_; }

    size_t GetCols() { return n_col_; }

    void CopyHostDataToGpu(T *host_tensor)
    {
        cudaMemcpy(data_, host_tensor, sizeof(T) * GetSize(),
                   cudaMemcpyHostToDevice);
    }

    void CopyGpuDataToHost(T *host_tensor)
    {
        cudaMemcpy(host_tensor, data_, sizeof(T) * GetSize(),
                   cudaMemcpyDeviceToHost);
    }
};

template <typename T>
void Transpose(CudaTensor<T> &a, CudaTensor<T> &trans_a,
               std::vector<size_t> &new_indices)
{
    cuttHandle plan;
    (cuttPlan(&plan, a.GetShape().size(), a.GetShape(), new_indices, sizeof(T),
              0));
    (cuttExecute(plan, a.GetData(), trans_a.GetData()));
    (cuttDestroy(plan));
}

struct CudaTransposePlan {
    bool no_transpose;
    cuttHandle plan;
    std::vector<size_t> axes;
    std::vector<size_t> output_shape;
    size_t output_urank;
};

struct CudaContractionPlan {

    CudaTransposePlan trans_a_plan;
    CudaTransposePlan trans_b_plan;

    std::vector<size_t> axes_a;
    std::vector<size_t> axes_b;

    std::vector<size_t> output_shape;
    size_t output_urank;
};

template <typename T>
CudaTransposePlan GetTransposePlan(CudaTensor<T> &input,
                                   const std::vector<size_t> &axes,
                                   int new_urank = -1)
{
    bool no_transpose = true;
    for (size_t i = 0; i < input.GetShape().size(); ++i) {
        if (axes[i] != i) {
            no_transpose = false;
            break;
        }
    }

    CudaTransposePlan transpose_plan;
    transpose_plan.no_transpose = no_transpose;
    if (no_transpose)
        return transpose_plan;

    std::vector<int> dim_old(input.GetShape().size());
    std::vector<int> axes_int(axes.size());
    for (int i = 0; i < axes.size(); i++) {
        axes_int[i] = axes[i];
    }
    for (int i = 0; i < dim_old.size(); i++) {
        dim_old[i] = input.GetShape()[i];
    }

    auto rank = dim_old.size();
    std::vector<size_t> dim_new(rank);
    for (size_t r = 0; r < rank; ++r) {
        dim_new[r] = dim_old[axes[r]];
    }

    cuttHandle plan;
    cuttPlan(&plan, rank, dim_old.data(), axes_int.data(), sizeof(T), 0);
    transpose_plan.plan = plan;
    transpose_plan.axes = axes;
    transpose_plan.output_urank =
        (new_urank == -1) ? input.GetUpperRank() : new_urank;
    transpose_plan.output_shape = dim_new;
    return transpose_plan;
}

template <typename T>
CudaContractionPlan GetContractionPlan(CudaTensor<T> &a, CudaTensor<T> &b,
                                       const std::vector<size_t> &axes_a,
                                       const std::vector<size_t> &axes_b)
{
    //    using namespace Jet::Utilities;
    auto &shape_a = a.GetShape();
    auto &shape_b = b.GetShape();

    auto a_rank = shape_a.size();
    auto b_rank = shape_b.size();

    std::vector<size_t> shape_c;
    const size_t rank_row_c = a_rank - axes_a.size();
    const size_t rank_col_c = b_rank - axes_b.size();
    shape_c.resize(rank_row_c + rank_col_c);

    std::vector<size_t> trans_axes_a;
    std::vector<size_t> trans_axes_b;
    size_t urank_a;
    size_t urank_b;

    {
        const size_t rank = a_rank;
        const size_t rank_row = rank - axes_a.size();
        const size_t rank_col = axes_a.size();
        size_t v[rank];
        for (size_t i = 0; i < rank; ++i)
            v[i] = i;
        for (size_t i = 0; i < rank_col; ++i)
            v[axes_a[i]] = rank;
        std::sort(v, v + rank);
        for (size_t i = 0; i < rank_col; ++i)
            v[rank_row + i] = axes_a[i];

        trans_axes_a.resize(rank);
        trans_axes_a.assign(v, v + rank);

        urank_a = rank_row;

        for (size_t i = 0; i < rank_row; ++i)
            shape_c[i] = shape_a[v[i]];
    }

    {
        const size_t rank = b_rank;
        const size_t rank_row = axes_b.size();
        const size_t rank_col = rank - axes_b.size();
        size_t v[rank];
        for (size_t i = 0; i < rank; ++i)
            v[i] = i;
        for (size_t i = 0; i < rank_row; ++i)
            v[axes_b[i]] = 0;
        std::sort(v, v + rank);
        for (size_t i = 0; i < rank_row; ++i)
            v[i] = axes_b[i];

        // trans_axes_b.assign(rank, v);
        trans_axes_b.resize(rank);
        trans_axes_b.assign(v, v + rank);
        urank_b = rank_row;

        for (size_t i = 0; i < rank_col; ++i)
            shape_c[i + rank_row_c] = shape_b[v[i + rank_row]];
    }

    CudaContractionPlan contraction_plan;
    contraction_plan.output_shape = shape_c;
    contraction_plan.output_urank = rank_row_c;
    contraction_plan.trans_a_plan = GetTransposePlan(a, trans_axes_a, urank_a);
    contraction_plan.trans_b_plan = GetTransposePlan(b, trans_axes_b, urank_b);
    contraction_plan.axes_a = axes_a;
    contraction_plan.axes_b = axes_b;
    return contraction_plan;
}

template <typename T>
CudaContractionPlan GetContractionPlan(CudaTensor<T> &a, CudaTensor<T> &b)
{
    auto &a_indices = a.GetIndices();
    auto &b_indices = b.GetIndices();

    auto &&contracted_indices = VectorIntersection(a_indices, b_indices);
    auto &&c_indices = VectorSubtraction(VectorUnion(a_indices, b_indices),
                                         contracted_indices);

    std::vector<size_t> &&axes_a = a.ConvertIndicesToAxes(contracted_indices);
    std::vector<size_t> &&axes_b = b.ConvertIndicesToAxes(contracted_indices);

    return GetContractionPlan(a, b, axes_a, axes_b);
}

template <typename T>
void Contract(CudaTensor<T> &a, CudaTensor<T> &b, CudaTensor<T> &c,
              const CudaContractionPlan &contraction_plan)
{
    int first_rows = 1;
    int first_cols = 1;
    int second_cols = 1;
    T *first;
    T *second;

    if (!contraction_plan.trans_a_plan.no_transpose) {
        auto rank = contraction_plan.trans_a_plan.output_shape.size();
        auto &shape = contraction_plan.trans_a_plan.output_shape;
        auto &urank = contraction_plan.trans_a_plan.output_urank;
        for (size_t i = 0; i < urank; ++i)
            first_rows *= shape[i];
        for (size_t i = urank; i < rank; ++i)
            first_cols *= shape[i];

        (cuttExecute(contraction_plan.trans_a_plan.plan, a.GetData(),
                     a.GetTransposeData()));
        first = a.GetTransposeData();
    }
    else {
        first_rows = a.GetRows();
        first_cols = a.GetCols();
        first = a.GetData();
    }

    if (!contraction_plan.trans_b_plan.no_transpose) {
        auto rank = contraction_plan.trans_a_plan.output_shape.size();
        auto &shape = contraction_plan.trans_a_plan.output_shape;
        auto &urank = contraction_plan.trans_a_plan.output_urank;
        for (size_t i = urank; i < rank; ++i)
            second_cols *= shape[i];
        (cuttExecute(contraction_plan.trans_b_plan.plan, b.GetData(),
                     b.GetTransposeData()));
        second = b.GetTransposeData();
    }
    else {
        second_cols = b.GetCols();
        second = b.GetData();
    }

    CUDAMatrixMultiply(first, second, c.GetData(), first_rows, first_cols,
                       second_cols);

    if (!contraction_plan.trans_a_plan.no_transpose) {
        (cuttDestroy(contraction_plan.trans_a_plan.plan));
    }
    if (!contraction_plan.trans_b_plan.no_transpose) {
        (cuttDestroy(contraction_plan.trans_b_plan.plan));
    }
}

}; // namespace Jet
