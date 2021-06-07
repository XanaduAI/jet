#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Abort.hpp"
#include "Tensor.hpp"
#include "Utilities.hpp"

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cutensor.h>

/**
 * @brief
 *
 * @param err
 */
inline void ThrowCudaError(cudaError_t &err)
{
    auto s = std::string(cudaGetErrorString(err));
    if (err != cudaSuccess) {
        throw Jet::Exception(std::string(cudaGetErrorString(err)));
    }
}

/**
 * @brief
 *
 * @param err
 */
inline void ThrowCuTensorError(cutensorStatus_t &err)
{
    if (err != CUTENSOR_STATUS_SUCCESS) {
        throw Jet::Exception(std::string(cutensorGetErrorString(err)));
    }
}

/**
 * @brief Calculate the strides for each dimension for the CUDA array
 *
 * @param extents
 * @return std::vector<int64_t>
 */
static std::vector<int64_t> GetStrides(const std::vector<size_t> &extents)
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
 * @param sizes The size of each indepedent dimension of the tensor data.
 * @return size_t Single index mapped to column-major (colexicographic) form.
 */
size_t RowOrderToColumnOrder(size_t row_order_linear_index,
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

namespace Jet {

template <class T = cuComplex> class CudaTensor {

  private:
    T *data_;

    std::vector<std::string> indices_;
    std::vector<size_t> shape_;
    std::unordered_map<std::string, size_t> index_to_dimension_;
    std::unordered_map<std::string, size_t> index_to_axes_;

  public:
    using scalar_type_t = T;
    using scalar_type_t_precision = decltype(std::declval<T>().x);

    void SetIndicesShapeAndMemory(const std::vector<std::string> &indices,
                                  const std::vector<size_t> &shape)
    {
        shape_ = shape;
        indices_ = indices;
        index_to_dimension_.clear();
        index_to_axes_.clear();

        for (std::size_t i = 0; i < shape_.size(); ++i) {
            index_to_dimension_[indices[i]] = shape[i];
            index_to_axes_[indices[i]] = i;
        }
        cudaMalloc(&data_, Jet::Utilities::ShapeToSize(shape_) * sizeof(T));
    }

    CudaTensor()
    {
        cuComplex h_dat({.x = 0.0, .y = 0.0});
        cudaMalloc(&data_, sizeof(T));
        cudaMemcpy(data_, &h_dat, sizeof(T), cudaMemcpyHostToDevice);
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape)
    {
        SetIndicesShapeAndMemory(indices, shape);
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const std::vector<T> data)
        : CudaTensor(indices, shape)
    {
        cudaMemcpy(data_, data.data(), sizeof(T) * data.size(),
                   cudaMemcpyHostToDevice);
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const T *data)
        : CudaTensor(indices, shape)
    {
        cudaMemcpy(data_, data, sizeof(T) * Jet::Utilities::ShapeToSize(shape),
                   cudaMemcpyHostToDevice);
    }

    CudaTensor(const std::vector<size_t> &shape) : CudaTensor()
    {
        using namespace Utilities;
        std::vector<std::string> indices(shape.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = "?" + GenerateStringIndex(i);
        }
        SetIndicesShapeAndMemory(indices, shape);
    }

    ~CudaTensor() { cudaFree(data_); }

    template <class U = T>
    static CudaTensor<U> ContractTensors(const CudaTensor<U> &a_tensor,
                                         const CudaTensor<U> &b_tensor)
    {
        using namespace Utilities;

        auto &&left_indices =
            VectorSubtraction(a_tensor.GetIndices(), b_tensor.GetIndices());
        auto &&right_indices =
            VectorSubtraction(b_tensor.GetIndices(), a_tensor.GetIndices());

        std::size_t left_dim = 1;
        std::size_t right_dim = 1;

        for (std::size_t i = 0; i < left_indices.size(); ++i) {
            left_dim *= a_tensor.GetIndexToDimension().at(left_indices[i]);
        }
        for (std::size_t i = 0; i < right_indices.size(); ++i) {
            right_dim *= b_tensor.GetIndexToDimension().at(right_indices[i]);
        }

        auto &&c_indices = VectorUnion(left_indices, right_indices);

        std::vector<std::size_t> c_shape(c_indices.size());
        for (std::size_t i = 0; i < left_indices.size(); ++i)
            c_shape[i] = a_tensor.GetIndexToDimension().at(left_indices[i]);
        for (std::size_t i = 0; i < right_indices.size(); ++i)
            c_shape[i + left_indices.size()] =
                b_tensor.GetIndexToDimension().at(right_indices[i]);

        CudaTensor<U> c_tensor(c_indices, c_shape);
        auto plan = GetCudaContractionPlan<T>(a_tensor, b_tensor, c_tensor);
        ContractTensorsWithoutAllocation<T>(a_tensor, b_tensor, c_tensor, plan);
        return c_tensor;
    }

    CudaTensor<T> ContractTensors(const CudaTensor<T> &other) const
    {
        return ContractTensors<T>(*this, other);
    }

    const std::vector<std::string> &GetIndices() const { return indices_; }

    std::vector<size_t>
    ConvertIndicesToAxes(const std::vector<std::string> &indices) const
    {
        using namespace Utilities;
        std::cout << "ConvertIndicesToAxes" << std::endl;
        std::cout << "indices = " << indices << std::endl;
        for (auto i : index_to_axes_) {
            std::cout << "i.first = " << i.first << std::endl;
            std::cout << "i.second = " << i.second << std::endl;
        }

        std::vector<size_t> axes(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            axes[i] = index_to_axes_.at(indices[i]);
        }
        return axes;
    }

    void Clear_()
    {
        index_to_dimension_.clear();
        index_to_axes_.clear();
        shape_.clear();
        indices_.clear();
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        cudaFree(data_);
        data_ = nullptr;
    }

    void Move_(CudaTensor &&other)
    {
        Clear_();
        indices_ = std::move(other.indices_);
        shape_ = std::move(other.shape_);
        index_to_dimension_ = std::move(other.index_to_dimension_);
        index_to_axes_ = std::move(other.index_to_axes_);
        data_ = other.data_;
        other.data_ = nullptr;
    }

    CudaTensor(CudaTensor &&other) { Move_(std::move(other)); }

    CudaTensor(const CudaTensor &other)
    {
        SetIndicesShapeAndMemory(other.GetIndices(), other.GetShape());
        cudaMemcpy(data_, other.GetData(), sizeof(T) * GetSize(),
                   cudaMemcpyDeviceToDevice);
    }

    CudaTensor &operator=(const CudaTensor &other)
    {
        if (this != &other) // not a self-assignment
        {
            SetIndicesShapeAndMemory(other.GetIndices(), other.GetShape());
            cudaMemcpy(data_, other.GetData(), sizeof(T) * GetSize(),
                       cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    T *GetData() { return data_; }
    const T *GetData() const { return data_; }

    const std::vector<size_t> &GetShape() const { return shape_; }

    size_t GetSize()
    {
        std::size_t total_dim = 1;
        for (std::size_t i = 0; i < shape_.size(); ++i)
            total_dim *= shape_[i];
        return total_dim;
    }

    inline void CopyHostDataToGpu(T *host_tensor)
    {
        cudaError_t retcode = cudaMemcpy(
            data_, host_tensor, sizeof(T) * GetSize(), cudaMemcpyHostToDevice);
        ThrowCudaError(retcode);
    }

    inline void CopyGpuDataToHost(T *host_tensor)
    {
        cudaError_t retcode = cudaMemcpy(
            host_tensor, data_, sizeof(T) * GetSize(), cudaMemcpyDeviceToHost);

        ThrowCudaError(retcode);
    }

    inline void CopyGpuDataToGpu(T *host_tensor)
    {
        cudaError_t retcode =
            cudaMemcpy(host_tensor, data_, sizeof(T) * GetSize(),
                       cudaMemcpyDeviceToDevice);
        ThrowCudaError(retcode);
    }

    inline void AsyncCopyHostDataToGpu(T *host_tensor, cudaStream_t stream = 0)
    {
        cudaError_t retcode =
            cudaMemcpyAsync(data_, host_tensor, sizeof(T) * GetSize(),
                            cudaMemcpyHostToDevice, stream);
        ThrowCudaError(retcode);
    }

    inline void AsyncCopyGpuDataToHost(T *host_tensor, cudaStream_t stream = 0)
    {
        cudaError_t retcode =
            cudaMemcpyAsync(host_tensor, data_, sizeof(T) * GetSize(),
                            cudaMemcpyDeviceToHost, stream);
        ThrowCudaError(retcode);
    }

    const std::unordered_map<std::string, size_t> &GetIndexToDimension() const
    {
        return index_to_dimension_;
    }

    explicit operator Tensor<std::complex<scalar_type_t_precision>>()
    {
        std::vector<std::complex<scalar_type_t_precision>> host_data(
            GetSize(), {0.0, 0.0});

        CopyGpuDataToHost(reinterpret_cast<T *>(host_data.data()));

        std::vector<std::complex<scalar_type_t_precision>> host_data_reshape =
            host_data;

        for (size_t idx = 0; idx < host_data_reshape.size(); idx++) {
            auto col_idx = RowOrderToColumnOrder(idx, GetShape());
            host_data_reshape[idx] = host_data[col_idx];
        }

        auto t = Tensor<std::complex<scalar_type_t_precision>>(
            GetIndices(), GetShape(), host_data_reshape);
        return t;
    }

    /**
     * @brief Randomly assign values to `%Tensor` object data. This method
     * will allow for reproducible random number generation with a given seed.
     *
     * @param seed Seed the RNG with a given value.
     */
    void FillRandom(size_t seed)
    {
        static curandGenerator_t rng;
        curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, seed);
        curandGenerateUniform(
            rng, reinterpret_cast<scalar_type_t_precision *>(data_),
            2 * GetSize());
    }

    /**
     * @brief Randomly assign values to `%CudaTensor` object data.
     *
     */
    void FillRandom()
    {
        static curandGenerator_t rng;
        curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, std::random_device{}());
        curandGenerateUniform(
            rng, reinterpret_cast<scalar_type_t_precision *>(data_),
            2 * GetSize());
    }

    /**
     * @brief Change `%CudaTensor` index label at given location.
     *
     * @param ind Location of `%CudaTensor` index label.
     * @param new_string New `%CudaTensor` index label.
     */
    void RenameIndex(size_t ind, std::string new_string)
    {
        std::string old_string = GetIndices()[ind];

        indices_[ind] = new_string;
        index_to_dimension_[new_string] = index_to_dimension_[old_string];
        index_to_dimension_.erase(old_string);
    }

    struct CudaContractionPlan {

        cutensorHandle_t handle;
        cutensorContractionPlan_t plan;
        size_t work_size;
        void *work;
    };

    template <class U = T>
    static CudaContractionPlan
    GetCudaContractionPlan(const CudaTensor<U> &a_tensor,
                           const CudaTensor<U> &b_tensor,
                           const CudaTensor<U> &c_tensor)
    {
        using namespace Jet::Utilities;

        // Note: the following 2 lines should be generalised
        cudaDataType_t data_type = CUDA_C_32F;
        cutensorComputeType_t compute_type = CUTENSOR_C_MIN_32F;

        const auto &a_indices = a_tensor.GetIndices();
        const auto &b_indices = b_tensor.GetIndices();
        const auto &c_indices = c_tensor.GetIndices();

        std::unordered_map<std::string, int> index_to_mode_map;
        std::unordered_map<int, int64_t> mode_to_dimension_map;

        for (size_t i = 0; i < a_indices.size(); i++) {
            if (!index_to_mode_map.count(a_indices[i])) {
                index_to_mode_map[a_indices[i]] = i;
                mode_to_dimension_map[i] = static_cast<int64_t>(
                    a_tensor.GetIndexToDimension().at(a_indices[i]));
            }
        }

        size_t stride = a_indices.size();
        for (size_t i = 0; i < b_indices.size(); i++) {
            if (!index_to_mode_map.count(b_indices[i])) {
                index_to_mode_map[b_indices[i]] = stride + i;
                mode_to_dimension_map[stride + i] = static_cast<int64_t>(
                    b_tensor.GetIndexToDimension().at(b_indices[i]));
            }
        }

        std::vector<int> a_modes(a_indices.size());
        std::vector<int> b_modes(b_indices.size());
        std::vector<int> c_modes(c_indices.size());

        for (size_t i = 0; i < a_indices.size(); i++) {
            a_modes[i] = index_to_mode_map[a_indices[i]];
        }
        for (size_t i = 0; i < b_indices.size(); i++) {
            b_modes[i] = index_to_mode_map[b_indices[i]];
        }
        for (size_t i = 0; i < c_indices.size(); i++) {
            c_modes[i] = index_to_mode_map[c_indices[i]];
        }

        std::vector<int64_t> c_dimensions;
        for (auto mode : c_modes)
            c_dimensions.push_back(mode_to_dimension_map[mode]);
        std::vector<int64_t> a_dimensions;
        for (auto mode : a_modes)
            a_dimensions.push_back(mode_to_dimension_map[mode]);
        std::vector<int64_t> b_dimensions;
        for (auto mode : b_modes)
            b_dimensions.push_back(mode_to_dimension_map[mode]);

        size_t a_elements = 1;
        for (auto mode : a_modes)
            a_elements *= mode_to_dimension_map[mode];

        size_t b_elements = 1;
        for (auto mode : b_modes)
            b_elements *= mode_to_dimension_map[mode];

        size_t c_elements = 1;
        for (auto mode : c_modes)
            c_elements *= mode_to_dimension_map[mode];

        cutensorHandle_t handle;
        cutensorInit(&handle);

        const std::vector<int64_t> a_strides = GetStrides(a_tensor.GetShape());
        const std::vector<int64_t> b_strides = GetStrides(b_tensor.GetShape());
        const std::vector<int64_t> c_strides = GetStrides(c_tensor.GetShape());

        cutensorStatus_t cutensor_err;
        cutensorTensorDescriptor_t a_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &a_descriptor, a_modes.size(), a_dimensions.data(),
            a_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        ThrowCuTensorError(cutensor_err);

        cutensorTensorDescriptor_t b_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &b_descriptor, b_modes.size(), b_dimensions.data(),
            b_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        ThrowCuTensorError(cutensor_err);

        cutensorTensorDescriptor_t c_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &c_descriptor, c_modes.size(), c_dimensions.data(),
            c_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        ThrowCuTensorError(cutensor_err);

        uint32_t a_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, a_tensor.GetData(), &a_descriptor,
            &a_alignment_requirement);
        ThrowCuTensorError(cutensor_err);

        uint32_t b_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, b_tensor.GetData(), &b_descriptor,
            &b_alignment_requirement);
        ThrowCuTensorError(cutensor_err);

        uint32_t c_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, c_tensor.GetData(), &c_descriptor,
            &c_alignment_requirement);
        ThrowCuTensorError(cutensor_err);

        cutensorContractionDescriptor_t descriptor;
        cutensor_err = cutensorInitContractionDescriptor(
            &handle, &descriptor, &a_descriptor, a_modes.data(),
            a_alignment_requirement, &b_descriptor, b_modes.data(),
            b_alignment_requirement, &c_descriptor, c_modes.data(),
            c_alignment_requirement, &c_descriptor, c_modes.data(),
            c_alignment_requirement, compute_type);
        ThrowCuTensorError(cutensor_err);

        cutensorContractionFind_t find;
        cutensor_err =
            cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT);
        ThrowCuTensorError(cutensor_err);

        uint64_t work_size = 0;
        cutensor_err = cutensorContractionGetWorkspace(
            &handle, &descriptor, &find, CUTENSOR_WORKSPACE_RECOMMENDED,
            &work_size);
        ThrowCuTensorError(cutensor_err);

        void *work = nullptr;
        if (work_size > 0) {
            if (cudaSuccess != cudaMalloc(&work, work_size)) {
                work = nullptr;
                work_size = 0;
            }
        }

        /**************************
         * Create Contraction Plan
         **************************/

        cutensorContractionPlan_t plan;
        cutensor_err = cutensorInitContractionPlan(&handle, &plan, &descriptor,
                                                   &find, work_size);
        ThrowCuTensorError(cutensor_err);

        CudaContractionPlan cplan;
        cplan.plan = plan;
        cplan.work = work;
        cplan.handle = handle;
        cplan.work_size = work_size;

        return cplan;
    }

    template <class U = T>
    static void ContractTensorsWithoutAllocation(const CudaTensor<U> &a,
                                                 const CudaTensor<U> &b,
                                                 CudaTensor<U> &c,
                                                 CudaContractionPlan &c_plan,
                                                 cudaStream_t stream = 0)
    {
        using namespace Utilities;

        U alpha;
        alpha.x = 1.0f;
        alpha.y = 0.0f;

        U beta;
        beta.x = 0.f;
        beta.y = 0.f;
        cutensorStatus_t cutensor_err;

        cutensor_err = cutensorContraction(
            &c_plan.handle, &c_plan.plan, (void *)&alpha, a.GetData(),
            b.GetData(), (void *)&beta, c.GetData(), c.GetData(), c_plan.work,
            c_plan.work_size, stream);
        ThrowCuTensorError(cutensor_err);
    }

    template <typename U = T>
    static CudaTensor<U> Reshape(const CudaTensor<U> &old_tens,
                                 const std::vector<size_t> &new_shape)
    {
        JET_ABORT("Reshape is not supported in this class yet");
        // dummy return
        return CudaTensor<U>();
    }

    template <typename U = T>
    static CudaTensor<U> SliceIndex(const CudaTensor<U> &tens,
                                    const std::string &index_str,
                                    std::size_t index_value)
    {
        JET_ABORT("SliceIndex is not supported in this class yet");
        // dummy return
        return CudaTensor<U>();
    }
};

} // namespace Jet
