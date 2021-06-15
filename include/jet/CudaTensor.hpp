#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Abort.hpp"
#include "CudaTensorHelpers.hpp"
#include "Tensor.hpp"
#include "Utilities.hpp"

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cutensor.h>

namespace {
using namespace Jet::CudaTensorHelpers;
}

namespace Jet {

template <class T = cuComplex> class CudaTensor {

    static_assert(CudaTensorHelpers::is_supported_data_type<T>,
                  "CudaTensor supports cuComplex (float2) and cuDoubleComplex "
                  "(double2) data types.");

  public:
    using scalar_type_t = T;
    using scalar_type_t_precision = decltype(std::declval<T>().x);

    void SetIndicesShapeAndMemory(const std::vector<std::string> &indices,
                                  const std::vector<size_t> &shape)
    {
        Clear_();
        shape_ = shape;
        indices_ = indices;

        for (size_t i = 0; i < shape_.size(); ++i) {
            index_to_dimension_[indices[i]] = shape[i];
            index_to_axes_[indices[i]] = i;
        }
        JET_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&data_),
                       Jet::Utilities::ShapeToSize(shape_) * sizeof(T)));
    }

    CudaTensor() : data_{nullptr}
    {
        T h_dat({.x = 0.0, .y = 0.0});
        JET_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&data_), sizeof(T)));
        JET_CUDA_IS_SUCCESS(
            cudaMemcpy(data_, &h_dat, sizeof(T), cudaMemcpyHostToDevice));
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape)
        : data_{nullptr}
    {
        SetIndicesShapeAndMemory(indices, shape);
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const std::vector<T> data)
        : CudaTensor(indices, shape)
    {
        JET_CUDA_IS_SUCCESS(cudaMemcpy(data_, data.data(),
                                       sizeof(T) * data.size(),
                                       cudaMemcpyHostToDevice));
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const T *data)
        : CudaTensor(indices, shape)
    {
        JET_CUDA_IS_SUCCESS(cudaMemcpy(
            data_, data, sizeof(T) * Jet::Utilities::ShapeToSize(shape),
            cudaMemcpyHostToDevice));
    }

    CudaTensor(const std::vector<size_t> &shape) : data_{nullptr}
    {
        using namespace Utilities;
        std::vector<std::string> indices(shape.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = std::string("?") + GenerateStringIndex(i);
        }
        SetIndicesShapeAndMemory(indices, shape);
    }

    ~CudaTensor() { JET_CUDA_IS_SUCCESS(cudaFree(data_)); }

    template <class U = T>
    static CudaTensor<U> ContractTensors(const CudaTensor<U> &a_tensor,
                                         const CudaTensor<U> &b_tensor)
    {
        using namespace Utilities;

        auto &&left_indices =
            VectorSubtraction(a_tensor.GetIndices(), b_tensor.GetIndices());
        auto &&right_indices =
            VectorSubtraction(b_tensor.GetIndices(), a_tensor.GetIndices());

        size_t left_dim = 1;
        size_t right_dim = 1;

        for (size_t i = 0; i < left_indices.size(); ++i) {
            left_dim *= a_tensor.index_to_dimension_.at(left_indices[i]);
        }
        for (size_t i = 0; i < right_indices.size(); ++i) {
            right_dim *= b_tensor.index_to_dimension_.at(right_indices[i]);
        }

        auto &&c_indices = VectorUnion(left_indices, right_indices);

        std::vector<size_t> c_shape(c_indices.size());
        for (size_t i = 0; i < left_indices.size(); ++i)
            c_shape[i] = a_tensor.GetIndexToDimension().at(left_indices[i]);
        for (size_t i = 0; i < right_indices.size(); ++i)
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
        JET_CUDA_IS_SUCCESS(cudaFree(data_));
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

    CudaTensor(CudaTensor &&other) : data_{nullptr} { Move_(std::move(other)); }

    CudaTensor(const CudaTensor &other) : data_{nullptr}
    {
        SetIndicesShapeAndMemory(other.GetIndices(), other.GetShape());
        cudaMemcpy(data_, other.GetData(), sizeof(T) * other.GetSize(),
                   cudaMemcpyDeviceToDevice);
    }

    template <class CPUData>
    CudaTensor(const Tensor<CPUData> &other) : data_{nullptr}
    {
        static_assert(sizeof(CPUData) == sizeof(T),
                      "Size of CPU and GPU data types do not match.");

        SetIndicesShapeAndMemory(other.GetIndices(), other.GetShape());
        CopyHostDataToGpu(const_cast<T *>(
            reinterpret_cast<const T *>(other.GetData().data())));
    }

    template <class CPUData> CudaTensor &operator=(const Tensor<CPUData> &other)
    {
        static_assert(sizeof(CPUData) == sizeof(T),
                      "Size of CPU and GPU data types do not match.");

        SetIndicesShapeAndMemory(ReverseVector(other.GetIndices()), ReverseVector(other.GetShape()));
        CopyHostDataToGpu(const_cast<T *>(
            reinterpret_cast<const T *>(other.GetData().data())));
        return *this;
    }

    CudaTensor &operator=(const CudaTensor &other)
    {
        if (this != &other) // not a self-assignment
        {
            SetIndicesShapeAndMemory(other.GetIndices(), other.GetShape());
            cudaMemcpy(data_, other.GetData(), sizeof(T) * other.GetSize(),
                       cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    T *GetData() { return data_; }
    const T *GetData() const { return data_; }

    const std::vector<size_t> &GetShape() const { return shape_; }

    size_t GetSize() const { return Jet::Utilities::ShapeToSize(shape_); }

    inline void CopyHostDataToGpu(T *host_tensor)
    {
        cudaError_t retcode = cudaMemcpy(
            data_, host_tensor, sizeof(T) * GetSize(), cudaMemcpyHostToDevice);
        JET_CUDA_IS_SUCCESS(retcode);
    }

    inline void CopyGpuDataToHost(T *host_tensor) const
    {
        cudaError_t retcode = cudaMemcpy(
            host_tensor, data_, sizeof(T) * GetSize(), cudaMemcpyDeviceToHost);

        JET_CUDA_IS_SUCCESS(retcode);
    }

    inline void CopyGpuDataToGpu(T *host_tensor)
    {
        cudaError_t retcode =
            cudaMemcpy(host_tensor, data_, sizeof(T) * GetSize(),
                       cudaMemcpyDeviceToDevice);
        JET_CUDA_IS_SUCCESS(retcode);
    }

    inline void AsyncCopyHostDataToGpu(T *host_tensor, cudaStream_t stream = 0)
    {
        cudaError_t retcode =
            cudaMemcpyAsync(data_, host_tensor, sizeof(T) * GetSize(),
                            cudaMemcpyHostToDevice, stream);
        JET_CUDA_IS_SUCCESS(retcode);
    }

    inline void AsyncCopyGpuDataToHost(T *host_tensor, cudaStream_t stream = 0)
    {
        cudaError_t retcode =
            cudaMemcpyAsync(host_tensor, data_, sizeof(T) * GetSize(),
                            cudaMemcpyDeviceToHost, stream);
        JET_CUDA_IS_SUCCESS(retcode);
    }

    const std::unordered_map<std::string, size_t> &GetIndexToDimension() const
    {
        return index_to_dimension_;
    }

    template<class DataType>
    static inline std::vector<DataType> ReverseVector(const std::vector<DataType> &input)
    {
        return std::vector<DataType>{input.rbegin(), input.rend()};
    }


    explicit operator Tensor<std::complex<scalar_type_t_precision>>() const
    {
        std::vector<std::complex<scalar_type_t_precision>> host_data(
            GetSize(), {0.0, 0.0});

        CopyGpuDataToHost(reinterpret_cast<T *>(host_data.data()));

        //std::vector<std::complex<scalar_type_t_precision>> host_data_reshape =
        //    host_data;

        /*for (size_t idx = 0; idx < host_data_reshape.size(); idx++) {
            auto col_idx = CudaTensorHelpers::RowMajToColMaj(idx, GetShape());
            host_data_reshape[idx] = host_data[col_idx];
        }*/

        auto t = Tensor<std::complex<scalar_type_t_precision>>(
            ReverseVector(GetIndices()), ReverseVector(GetShape()), host_data);
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

        if (old_string != new_string) {
            JET_ABORT_IF_NOT(
                index_to_dimension_.find(new_string) ==
                    index_to_dimension_.end(),
                "Renaming index to already existing value is not allowed.")

            indices_[ind] = new_string;
            index_to_dimension_[new_string] = index_to_dimension_[old_string];
            index_to_dimension_.erase(old_string);
        }
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

        cudaDataType_t data_type;
        cutensorComputeType_t compute_type;

        if constexpr (std::is_same<U, cuDoubleComplex>::value ||
                      std::is_same<U, double2>::value) {
            data_type = CUDA_C_64F;
            compute_type = CUTENSOR_COMPUTE_64F;
        }
        else {
            data_type = CUDA_C_32F;
            compute_type = CUTENSOR_COMPUTE_32F;
        }

        const auto &a_indices = a_tensor.GetIndices();
        const auto &b_indices = b_tensor.GetIndices();
        const auto &c_indices = c_tensor.GetIndices();

        std::unordered_map<std::string, int> index_to_mode_map;
        std::unordered_map<size_t, int64_t> mode_to_dimension_map;

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

        std::vector<int32_t> a_modes(a_indices.size());
        std::vector<int32_t> b_modes(b_indices.size());
        std::vector<int32_t> c_modes(c_indices.size());

        for (size_t i = 0; i < a_indices.size(); i++) {
            a_modes[i] = index_to_mode_map[a_indices[i]];
        }
        for (size_t i = 0; i < b_indices.size(); i++) {
            b_modes[i] = index_to_mode_map[b_indices[i]];
        }
        for (size_t i = 0; i < c_indices.size(); i++) {
            c_modes[i] = index_to_mode_map[c_indices[i]];
        }

        std::vector<int64_t> c_dimensions(c_modes.size());
        for (size_t idx = 0; idx < c_modes.size(); idx++) {
            c_dimensions[idx] = mode_to_dimension_map[c_modes[idx]];
        }

        std::vector<int64_t> a_dimensions(a_modes.size());
        for (size_t idx = 0; idx < a_modes.size(); idx++) {
            a_dimensions[idx] = mode_to_dimension_map[a_modes[idx]];
        }

        std::vector<int64_t> b_dimensions(b_modes.size());
        for (size_t idx = 0; idx < b_modes.size(); idx++) {
            b_dimensions[idx] = mode_to_dimension_map[b_modes[idx]];
        }

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

        const std::vector<int64_t> a_strides =
            CudaTensorHelpers::GetStrides(a_tensor.GetShape());
        const std::vector<int64_t> b_strides =
            CudaTensorHelpers::GetStrides(b_tensor.GetShape());
        const std::vector<int64_t> c_strides =
            CudaTensorHelpers::GetStrides(c_tensor.GetShape());

        cutensorStatus_t cutensor_err;
        cutensorTensorDescriptor_t a_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &a_descriptor, a_modes.size(), a_dimensions.data(),
            a_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensorTensorDescriptor_t b_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &b_descriptor, b_modes.size(), b_dimensions.data(),
            b_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensorTensorDescriptor_t c_descriptor;
        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &c_descriptor, c_modes.size(), c_dimensions.data(),
            c_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        uint32_t a_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, a_tensor.GetData(), &a_descriptor,
            &a_alignment_requirement);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        uint32_t b_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, b_tensor.GetData(), &b_descriptor,
            &b_alignment_requirement);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        uint32_t c_alignment_requirement;
        cutensor_err = cutensorGetAlignmentRequirement(
            &handle, c_tensor.GetData(), &c_descriptor,
            &c_alignment_requirement);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensorContractionDescriptor_t descriptor;
        cutensor_err = cutensorInitContractionDescriptor(
            &handle, &descriptor, &a_descriptor, a_modes.data(),
            a_alignment_requirement, &b_descriptor, b_modes.data(),
            b_alignment_requirement, &c_descriptor, c_modes.data(),
            c_alignment_requirement, &c_descriptor, c_modes.data(),
            c_alignment_requirement, compute_type);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensorContractionFind_t find;
        cutensor_err =
            cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        uint64_t work_size = 0;
        cutensor_err = cutensorContractionGetWorkspace(
            &handle, &descriptor, &find, CUTENSOR_WORKSPACE_RECOMMENDED,
            &work_size);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

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
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

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
            &c_plan.handle, &c_plan.plan, static_cast<void *>(&alpha),
            a.GetData(), b.GetData(), static_cast<void *>(&beta), c.GetData(),
            c.GetData(), c_plan.work, c_plan.work_size, stream);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);
    }

    template <typename U = T>
    static CudaTensor<U> Reshape(const CudaTensor<U> &old_tens,
                                 const std::vector<size_t> &new_shape)
    {
        // Avoid unused warning
        static_cast<void>(old_tens);
        static_cast<void>(new_shape);
        JET_ABORT("Reshape is not supported in this class yet");
        // dummy return
        return CudaTensor<U>();
    }

    CudaTensor<T> Reshape(const std::vector<size_t> &new_shape)
    {
        return Reshape<T>(*this, new_shape);
    }

    template <typename U = T>
    static CudaTensor<U> SliceIndex(const CudaTensor<U> &tens,
                                    const std::string &index_str,
                                    size_t index_value)
    {
        // Avoid unused warning
        static_cast<void>(tens);
        static_cast<void>(index_str);
        static_cast<void>(index_value);
        JET_ABORT("SliceIndex is not supported in this class yet");
        // dummy return
        return CudaTensor<U>();
    }

    CudaTensor<T> SliceIndex(const std::string &index_str, size_t index_value)
    {
        return SliceIndex<T>(*this, index_str, index_value);
    }

    /**
     * @brief Return GPU data as host-side data vector.
     *
     * @warn This operation copies data from the GPU to the CPU, and will slow
     * down any program execution if used.
     *
     * @return Vector containing the GPU data.
     */
    std::vector<std::complex<scalar_type_t_precision>> GetHostDataVector() const
    {
        std::vector<std::complex<scalar_type_t_precision>> host_data_buffer(
            GetSize());
        auto ptr = reinterpret_cast<T *>(host_data_buffer.data());
        CopyGpuDataToHost(ptr);
        return host_data_buffer;
    }

  private:
    T *data_;

    std::vector<std::string> indices_;
    std::vector<size_t> shape_;
    std::unordered_map<std::string, size_t> index_to_dimension_;
    std::unordered_map<std::string, size_t> index_to_axes_;
};

} // namespace Jet
