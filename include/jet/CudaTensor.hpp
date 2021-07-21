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

#include <taskflow/cudaflow.hpp>
#include <taskflow/taskflow.hpp>

namespace {
using namespace Jet::CudaTensorHelpers;
}

namespace Jet {

template <class T = cuComplex, int CUDA_DEVICE = 0> class CudaTensor {

    static_assert(CudaTensorHelpers::is_supported_data_type<T>,
                  "CudaTensor supports cuComplex (float2) and cuDoubleComplex "
                  "(double2) data types.");

  public:
    using scalar_type_t = T;
    using scalar_type_t_precision = decltype(std::declval<T>().x);

    template <class U = T, int d = cuda_device>
    static CudaTensor<U, d> AddTensors(const CudaTensor<U, d> &A,
                                       const CudaTensor<U, d> &B)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        static const CudaTensor<U, d> zero;

        // The zero tensor is used in reductions where the shape of an
        // accumulator is not known beforehand.
        if (A == zero) {
            return B;
        }
        else if (B == zero) {
            return A;
        }

        const auto disjoint_indices = Jet::Utilities::VectorDisjunctiveUnion(
            A.GetIndices(), B.GetIndices());

        JET_ABORT_IF_NOT(
            disjoint_indices.empty(),
            "Tensor addition with disjoint indices is not supported.");

        CudaTensor<U, d> C(A);

        // Align the underlying data vectors of `A` and `B`.
        cutensorHandle_t handle;
        cutensorInit(&handle);

        static const U one = {1.0, 0.0};

        cutensorTensorDescriptor_t b_descriptor, c_descriptor;
        cutensorStatus_t cutensor_err;
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<U, cuDoubleComplex> ||
                      std::is_same_v<U, double2>) {
            data_type = CUDA_C_64F;
        }
        else {
            data_type = CUDA_C_32F;
        }

        const auto &b_indices = B.GetIndices();
        const auto &c_indices = C.GetIndices();

        const std::vector<int64_t> b_strides =
            CudaTensorHelpers::GetStrides(B.GetShape());
        const std::vector<int64_t> c_strides =
            CudaTensorHelpers::GetStrides(C.GetShape());

        std::unordered_map<std::string, int> index_to_mode_map;
        std::unordered_map<size_t, int64_t> mode_to_dimension_map;

        for (size_t i = 0; i < c_indices.size(); i++) {

            if (index_to_mode_map.insert({c_indices[i], i}).second) {
                mode_to_dimension_map.emplace(
                    i, static_cast<int64_t>(
                           C.GetIndexToDimension().at(c_indices[i])));
            }
        }

        std::vector<int32_t> b_modes(b_indices.size());
        std::vector<int32_t> c_modes(c_indices.size());

        for (size_t i = 0; i < b_indices.size(); i++) {
            b_modes[i] = index_to_mode_map[b_indices[i]];
        }
        for (size_t i = 0; i < c_indices.size(); i++) {
            c_modes[i] = index_to_mode_map[c_indices[i]];
        }

        std::vector<int64_t> b_dimensions(b_modes.size());
        for (size_t i = 0; i < b_modes.size(); i++) {
            b_dimensions[i] = mode_to_dimension_map[b_modes[i]];
        }
        std::vector<int64_t> c_dimensions(c_modes.size());
        for (size_t i = 0; i < c_modes.size(); i++) {
            c_dimensions[i] = mode_to_dimension_map[c_modes[i]];
        }

        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &b_descriptor, b_modes.size(), b_dimensions.data(),
            b_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensor_err = cutensorInitTensorDescriptor(
            &handle, &c_descriptor, c_modes.size(), c_dimensions.data(),
            c_strides.data(), data_type, CUTENSOR_OP_IDENTITY);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cutensor_err = cutensorElementwiseBinary(
            &handle, &one, B.GetData(), &b_descriptor, b_modes.data(), &one,
            C.GetData(), &c_descriptor, c_modes.data(), C.GetData(),
            &c_descriptor, c_modes.data(), CUTENSOR_OP_ADD, data_type, nullptr);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        return C;
    }

    CudaTensor<T, cuda_device>
    AddTensor(const CudaTensor<T, cuda_device> &other) const
    {
        return AddTensors<T, cuda_device>(*this, other);
    }

    void InitIndicesAndShape(const std::vector<std::string> &indices,
                             const std::vector<size_t> &shape)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        Clear_();
        shape_ = shape;
        indices_ = indices;

        for (size_t i = 0; i < shape_.size(); ++i) {
            index_to_dimension_[indices_[i]] = shape_[i];
            index_to_axes_[indices_[i]] = i;
        }

        JET_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&data_),
                       Jet::Utilities::ShapeToSize(shape_) * sizeof(T)));
    }

    CudaTensor() : data_{nullptr}
    {
        tf::cudaScopedDevice ctx(cuda_device);
        T h_dat;
        h_dat.x = 0.;
        h_dat.y = 0.;
        JET_CUDA_IS_SUCCESS(
            cudaMalloc(reinterpret_cast<void **>(&data_), sizeof(T)));
        JET_CUDA_IS_SUCCESS(
            cudaMemcpy(data_, &h_dat, sizeof(T), cudaMemcpyHostToDevice));
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape)
        : data_{nullptr}
    {
        InitIndicesAndShape(indices, shape);
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const std::vector<T> data)
        : CudaTensor(indices, shape)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpy(data_, data.data(),
                                       sizeof(T) * data.size(),
                                       cudaMemcpyHostToDevice));
    }

    CudaTensor(const std::vector<std::string> &indices,
               const std::vector<size_t> &shape, const T *data)
        : CudaTensor(indices, shape)
    {
        tf::cudaScopedDevice ctx(cuda_device);
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
        InitIndicesAndShape(indices, shape);
    }

    ~CudaTensor() { JET_CUDA_IS_SUCCESS(cudaFree(data_)); }

    template <class U = T, int d = cuda_device>
    static CudaTensor<U, d> ContractTensors(const CudaTensor<U, d> &a_tensor,
                                            const CudaTensor<U, d> &b_tensor)
    {
        tf::cudaScopedDevice ctx(cuda_device);
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

        CudaTensor<U, d> c_tensor(c_indices, c_shape);

        CudaContractionPlan cplan;

        GetCudaContractionPlan<T>(cplan, a_tensor, b_tensor, c_tensor);
        ContractTensorsWithoutAllocation<T>(a_tensor, b_tensor, c_tensor,
                                            cplan);
        return c_tensor;
    }

    CudaTensor<T, cuda_device>
    ContractTensors(const CudaTensor<T, cuda_device> &other) const
    {
        return ContractTensors<T, cuda_device>(*this, other);
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

    CudaTensor(CudaTensor &&other) : data_{nullptr}
    {
        tf::cudaScopedDevice ctx(cuda_device);
        Move_(std::move(other));
    }

    CudaTensor(const CudaTensor &other) : data_{nullptr}
    {
        tf::cudaScopedDevice ctx(cuda_device);
        InitIndicesAndShape(other.GetIndices(), other.GetShape());

        JET_CUDA_IS_SUCCESS(cudaMemcpy(data_, other.GetData(),
                                       sizeof(T) * other.GetSize(),
                                       cudaMemcpyDeviceToDevice));
    }

    template <class CPUData>
    CudaTensor(const Tensor<CPUData> &other) : data_{nullptr}
    {
        tf::cudaScopedDevice ctx(cuda_device);
        static_assert(sizeof(CPUData) == sizeof(T),
                      "Size of CPU and GPU data types do not match.");

        InitIndicesAndShape(ReverseVector(other.GetIndices()),
                            ReverseVector(other.GetShape()));
        CopyHostDataToGpu(const_cast<T *>(
            reinterpret_cast<const T *>(other.GetData().data())));
    }

    template <class CPUData> CudaTensor &operator=(const Tensor<CPUData> &other)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        static_assert(sizeof(CPUData) == sizeof(T),
                      "Size of CPU and GPU data types do not match.");

        InitIndicesAndShape(ReverseVector(other.GetIndices()),
                            ReverseVector(other.GetShape()));
        CopyHostDataToGpu(const_cast<T *>(
            reinterpret_cast<const T *>(other.GetData().data())));
        return *this;
    }

    CudaTensor &operator=(const CudaTensor &other)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        if (this != &other) // not a self-assignment
        {
            InitIndicesAndShape(other.GetIndices(), other.GetShape());
            JET_CUDA_IS_SUCCESS(cudaMemcpy(data_, other.GetData(),
                                           sizeof(T) * other.GetSize(),
                                           cudaMemcpyDeviceToDevice));
        }
        return *this;
    }

    T *GetData() { return data_; }
    const T *GetData() const { return data_; }

    const std::vector<size_t> &GetShape() const { return shape_; }

    size_t GetSize() const { return Jet::Utilities::ShapeToSize(shape_); }

    inline void CopyHostDataToGpu(T *host_tensor)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpy(
            data_, host_tensor, sizeof(T) * GetSize(), cudaMemcpyHostToDevice));
    }

    inline void CopyGpuDataToHost(T *host_tensor) const
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpy(
            host_tensor, data_, sizeof(T) * GetSize(), cudaMemcpyDeviceToHost));
    }

    inline void CopyGpuDataToGpu(T *host_tensor)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpy(host_tensor, data_,
                                       sizeof(T) * GetSize(),
                                       cudaMemcpyDeviceToDevice));
    }

    inline void AsyncCopyHostDataToGpu(T *host_tensor, cudaStream_t stream = 0)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpyAsync(data_, host_tensor,
                                            sizeof(T) * GetSize(),
                                            cudaMemcpyHostToDevice, stream));
    }

    inline void AsyncCopyGpuDataToHost(T *host_tensor, cudaStream_t stream = 0)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_CUDA_IS_SUCCESS(cudaMemcpyAsync(host_tensor, data_,
                                            sizeof(T) * GetSize(),
                                            cudaMemcpyDeviceToHost, stream));
    }

    const std::unordered_map<std::string, size_t> &GetIndexToDimension() const
    {
        return index_to_dimension_;
    }

    explicit operator Tensor<std::complex<scalar_type_t_precision>>() const
    {
        tf::cudaScopedDevice ctx(cuda_device);
        std::vector<std::complex<scalar_type_t_precision>> host_data(
            GetSize(), {0.0, 0.0});

        CopyGpuDataToHost(reinterpret_cast<T *>(host_data.data()));
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
        tf::cudaScopedDevice ctx(cuda_device);
        static curandGenerator_t rng;
        JET_CURAND_IS_SUCCESS(
            curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        JET_CURAND_IS_SUCCESS(curandSetPseudoRandomGeneratorSeed(rng, seed));
        JET_CURAND_IS_SUCCESS(curandGenerateUniform(
            rng, reinterpret_cast<scalar_type_t_precision *>(data_),
            2 * GetSize()));
    }

    /**
     * @brief Randomly assign values to `%CudaTensor` object data.
     *
     */
    void FillRandom()
    {
        tf::cudaScopedDevice ctx(cuda_device);
        static curandGenerator_t rng;
        JET_CURAND_IS_SUCCESS(
            curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
        JET_CURAND_IS_SUCCESS(
            curandSetPseudoRandomGeneratorSeed(rng, std::random_device{}()));
        JET_CURAND_IS_SUCCESS(curandGenerateUniform(
            rng, reinterpret_cast<scalar_type_t_precision *>(data_),
            2 * GetSize()));
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

        // ~CudaContractionPlan() { JET_CUDA_IS_SUCCESS(cudaFree(work)); }
    };

    template <class U = T, int d = cuda_device>
    static void GetCudaContractionPlan(CudaContractionPlan &cplan,
                                       const CudaTensor<U, d> &a_tensor,
                                       const CudaTensor<U, d> &b_tensor,
                                       const CudaTensor<U, d> &c_tensor)
    {
        using namespace Jet::Utilities;

        cudaDataType_t data_type;
        cutensorComputeType_t compute_type;

        if constexpr (std::is_same_v<U, cuDoubleComplex> ||
                      std::is_same_v<U, double2>) {
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
                mode_to_dimension_map.emplace(
                    i, static_cast<int64_t>(
                           a_tensor.GetIndexToDimension().at(a_indices[i])));
            }
        }

        size_t stride = a_indices.size();
        for (size_t i = 0; i < b_indices.size(); i++) {
            if (!index_to_mode_map.count(b_indices[i])) {
                index_to_mode_map[b_indices[i]] = stride + i;
                mode_to_dimension_map.emplace(
                    stride + i,
                    static_cast<int64_t>(
                        b_tensor.GetIndexToDimension().at(b_indices[i])));
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
        for (size_t i = 0; i < c_modes.size(); i++) {
            c_dimensions[i] = mode_to_dimension_map[c_modes[i]];
        }

        std::vector<int64_t> a_dimensions(a_modes.size());
        for (size_t i = 0; i < a_modes.size(); i++) {
            a_dimensions[i] = mode_to_dimension_map[a_modes[i]];
        }

        std::vector<int64_t> b_dimensions(b_modes.size());
        for (size_t i = 0; i < b_modes.size(); i++) {
            b_dimensions[i] = mode_to_dimension_map[b_modes[i]];
        }

        cutensorHandle_t handle;
        JET_CUTENSOR_IS_SUCCESS(cutensorInit(&handle));

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

        /**************************
         * Create Contraction Plan
         **************************/

        cutensorContractionPlan_t plan;
        cutensor_err = cutensorInitContractionPlan(&handle, &plan, &descriptor,
                                                   &find, work_size);
        JET_CUTENSOR_IS_SUCCESS(cutensor_err);

        cplan.plan = plan;

        if (work_size > 0) {
            JET_CUDA_IS_SUCCESS(cudaMalloc(&cplan.work, work_size));
        }

        cplan.handle = handle;
        cplan.work_size = work_size;
    }

    template <class U = T, int d = cuda_device>
    static void ContractTensorsWithoutAllocation(const CudaTensor<U, d> &a,
                                                 const CudaTensor<U, d> &b,
                                                 CudaTensor<U, d> &c,
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

    template <typename U = T, int d = cuda_device>
    static CudaTensor<U, d> Reshape(const CudaTensor<U> &old_tensor,
                                    const std::vector<size_t> &new_shape)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        JET_ABORT_IF_NOT(old_tensor.GetSize() ==
                             Jet::Utilities::ShapeToSize(new_shape),
                         "Size is inconsistent between tensors.");

        CudaTensor<U, d> reshaped_tensor(new_shape);
        JET_CUDA_IS_SUCCESS(cudaMemcpy(reshaped_tensor.data_, old_tensor.data_,
                                       sizeof(U) * old_tensor.GetSize(),
                                       cudaMemcpyDeviceToDevice));
        return reshaped_tensor;
    }

    CudaTensor<T> Reshape(const std::vector<size_t> &new_shape)
    {
        return Reshape<T>(*this, new_shape);
    }

    /**
     * @brief Transposes the indices of a `%CudaTensor` object to a new
     * ordering.
     *
     * @tparam U `%CudaTensor` data type.
     * @param A Reference `%CudaTensor` object.
     * @param new_indices New `%Tensor` index label ordering.
     * @return Transposed `%CudaTensor` object.
     */
    template <class U = T, int d = cuda_device>
    static CudaTensor<U, d>
    Transpose(const CudaTensor<U, d> &tensor,
              const std::vector<std::string> &new_indices)
    {
        tf::cudaScopedDevice ctx(cuda_device);
        using namespace Jet::Utilities;

        if (tensor.GetIndices() == new_indices)
            return tensor;

        JET_ABORT_IF(tensor.GetIndices().empty(),
                     "Number of indices cannot be zero.");

        std::vector<size_t> output_shape(tensor.GetShape().size());
        for (size_t i = 0; i < new_indices.size(); i++)
            output_shape[i] = tensor.GetIndexToDimension().at(new_indices[i]);

        // 0. Perform up-front util allocations
        static const U one = {1.0, 0.0};

        const std::vector<std::string> &old_indices = tensor.GetIndices();

        // 1. Allocate permuted tensor memory
        CudaTensor<U, d> permuted_tensor(new_indices, output_shape);

        // 2. Initialise CuTensor runtime & setup necessary options
        cutensorHandle_t handle;
        cudaDataType_t data_type;
        JET_CUTENSOR_IS_SUCCESS(cutensorInit(&handle));

        if constexpr (std::is_same<U, cuDoubleComplex>::value ||
                      std::is_same<U, double2>::value) {
            data_type = CUDA_C_64F;
        }
        else {
            data_type = CUDA_C_32F;
        }

        // 3. Build mode & dimensions array for Input
        std::unordered_map<std::string, int> index_to_mode_map;
        std::unordered_map<size_t, int64_t> mode_to_dimension_map;

        for (size_t i = 0; i < old_indices.size(); i++) {
            if (index_to_mode_map.insert({old_indices[i], i}).second) {
                mode_to_dimension_map.emplace(
                    i, static_cast<int64_t>(
                           tensor.GetIndexToDimension().at(old_indices[i])));
            }
        }

        std::vector<int32_t> input_modes(old_indices.size());
        std::vector<int64_t> input_dimensions(old_indices.size());

        for (size_t i = 0; i < old_indices.size(); i++) {
            input_modes[i] = index_to_mode_map[old_indices[i]];
        }

        for (size_t idx = 0; idx < input_modes.size(); idx++) {
            input_dimensions[idx] = mode_to_dimension_map[input_modes[idx]];
        }

        // 4. Build Descriptor for input
        cutensorTensorDescriptor_t input_descriptor;

        const std::vector<int64_t> input_strides =
            CudaTensorHelpers::GetStrides(tensor.GetShape());

        JET_CUTENSOR_IS_SUCCESS(cutensorInitTensorDescriptor(
            &handle, &input_descriptor, input_modes.size(),
            input_dimensions.data(), input_strides.data(), data_type,
            CUTENSOR_OP_IDENTITY));

        // 5. Build mode array for output
        for (size_t i = 0; i < new_indices.size(); i++) {
            if (index_to_mode_map.insert({new_indices[i], i}).second) {
                mode_to_dimension_map.emplace(
                    i, static_cast<int64_t>(
                           permuted_tensor.GetIndexToDimension().at(
                               new_indices[i])));
            }
        }

        std::vector<int32_t> output_modes(new_indices.size());
        std::vector<int64_t> output_dimensions(new_indices.size());

        for (size_t i = 0; i < new_indices.size(); i++) {
            output_modes[i] = index_to_mode_map[new_indices[i]];
        }

        for (size_t idx = 0; idx < output_modes.size(); idx++) {
            output_dimensions[idx] = mode_to_dimension_map[output_modes[idx]];
        }

        // 6. Build descriptor for output
        cutensorTensorDescriptor_t output_descriptor;

        const std::vector<int64_t> output_strides =
            CudaTensorHelpers::GetStrides(output_shape);

        JET_CUTENSOR_IS_SUCCESS(cutensorInitTensorDescriptor(
            &handle, &output_descriptor, output_modes.size(),
            output_dimensions.data(), output_strides.data(), data_type,
            CUTENSOR_OP_IDENTITY));

        // 7. Permute tensor indices
        JET_CUTENSOR_IS_SUCCESS(cutensorPermutation(
            &handle, &one, tensor.data_, &input_descriptor, input_modes.data(),
            permuted_tensor.data_, &output_descriptor, output_modes.data(),
            data_type, nullptr));

        return permuted_tensor;
    }

    /**
     * @brief Transposes the indices of a `%Tensor` to a new ordering.
     *
     * @warning The program is aborted if the number of elements in the new
     * ordering does match the number of indices in the tensor.
     *
     * @tparam U `%Tensor` data type.
     * @param A Reference `%Tensor` object.
     * @param new_ordering New `%Tensor` index permutation.
     * @return Transposed `%Tensor` object.
     */
    template <class U = T, int d = cuda_device>
    static CudaTensor<U, d> Transpose(const CudaTensor<U, d> &A,
                                      const std::vector<size_t> &new_ordering)
    {
        const size_t num_indices = A.GetIndices().size();
        JET_ABORT_IF_NOT(
            num_indices == new_ordering.size(),
            "Size of ordering must match number of tensor indices.");

        std::vector<std::string> new_indices(num_indices);
        const auto &old_indices = A.GetIndices();

        for (size_t i = 0; i < num_indices; i++) {
            new_indices[i] = old_indices[new_ordering[i]];
        }

        return Transpose<U, d>(A, new_indices);
    }
    /**
     * @brief Transposes the indices of the `%Tensor` object to a new ordering.
     *
     * @see Transpose(const Tensor<U>&, const std::vector<size_t>&)
     */
    CudaTensor<T, cuda_device>
    Transpose(const std::vector<size_t> &new_ordering) const
    {
        return Transpose<T, cuda_device>(*this, new_ordering);
    }
    /**
     * @brief Transposes the indices of the `%Tensor` object to a new ordering.
     *
     * @see Transpose(const Tensor<U>&, const std::vector<std::string>&)
     */
    CudaTensor<T, cuda_device>
    Transpose(const std::vector<std::string> &new_indices) const
    {
        return Transpose<T, cuda_device>(*this, new_indices);
    }

    template <typename U = T, int d = cuda_device>
    static CudaTensor<U, d> SliceIndex(const CudaTensor<U, d> &tens,
                                       const std::string &index_str,
                                       size_t index_value)
    {
        tf::cudaScopedDevice ctx(d);
        std::vector<std::string> new_indices = tens.GetIndices();
        std::vector<std::string> old_indices = tens.GetIndices();

        std::vector<size_t> output_shape(tens.GetShape());

        if (new_indices.back() != index_str) {
            auto it = std::find(new_indices.begin(), new_indices.end() - 1,
                                index_str);
            const size_t offset = std::distance(new_indices.begin(), it);

            std::swap(*it, new_indices.back());
            std::swap(output_shape[offset], output_shape.back());
        }

        CudaTensor<U, d> permuted_tensor = Transpose(tens, new_indices);

        CudaTensor<U, d> sliced_tensor({permuted_tensor.GetIndices().begin(),
                                        permuted_tensor.GetIndices().end() - 1},
                                       {permuted_tensor.GetShape().begin(),
                                        permuted_tensor.GetShape().end() - 1});

        const size_t ptr_offset =
            Jet::Utilities::ShapeToSize(sliced_tensor.GetShape());

        JET_CUDA_IS_SUCCESS(
            cudaMemcpy(sliced_tensor.data_,
                       permuted_tensor.data_ + (ptr_offset * index_value),
                       sizeof(U) * ptr_offset, cudaMemcpyDeviceToDevice));

        old_indices.erase(
            std::remove(old_indices.begin(), old_indices.end(), index_str),
            old_indices.end());

        return sliced_tensor.Transpose(old_indices);
    }

    CudaTensor<T, cuda_device> SliceIndex(const std::string &index_str,
                                          size_t index_value)
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

    bool operator==(const CudaTensor<T, cuda_device> &other) const noexcept
    {
        return shape_ == other.GetShape() && indices_ == other.GetIndices() &&
               index_to_dimension_ == other.GetIndexToDimension() &&
               GetHostDataVector() == other.GetHostDataVector();
    }
};

} // namespace Jet
