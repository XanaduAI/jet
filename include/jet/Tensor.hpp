#pragma once

/**
 * Cache friendly size (for complex<float>) to move things around.
 */
#ifndef MAX_RIGHT_DIM
#define MAX_RIGHT_DIM 1024
#endif

/**
 * Smallest size of cache friendly blocks (for complex<float>).
 */
#ifndef MIN_RIGHT_DIM
#define MIN_RIGHT_DIM 32
#endif

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "TensorHelpers.hpp"
#include "Utilities.hpp"

namespace Jet {

/**
 * @brief `%Tensor` class represents an n-rank data-structure of
 * complex-valued data for tensor operations. We use the following conventions:
 * - rank & order are used interchangeably, referring to the number of tensor
 * indices.
 * - dimension refers to the number of elements for a given tensor index.
 * - shape refers to all dimensions for the given tensor.
 *
 * @tparam T Underlying complex tensor data type (`complex<float>`
 * or `complex<double>`).
 */
template <class T = std::complex<float>> class Tensor {

    static_assert(TensorHelpers::is_supported_data_type_v<T>,
                  "Tensor data type must be one of std::complex<float>, "
                  "std::complex<double>");

  private:
    std::vector<std::string> indices_;
    std::vector<size_t> shape_;
    std::unordered_map<std::string, size_t> index_to_dimension_;
    std::vector<T> data_;

  public:
    /**
     * @brief Initialize `%Tensor` indices and size of each index. Indices
     * and shapes are ordered to map directly (e.g `indices[i]` has size
     * `shape[i]`).
     *
     * @param indices Index labels for `%Tensor` object.
     * @param shape Number of elements per index.
     */
    void InitIndicesAndShape(const std::vector<std::string> &indices,
                             const std::vector<size_t> &shape)
    {
        indices_ = indices;
        shape_ = shape;
        for (size_t i = 0; i < shape_.size(); ++i)
            index_to_dimension_[indices[i]] = shape[i];
    }

    /**
     * @brief Allow ease-of-access to tensor data type.
     *
     */
    using scalar_type_t = T;

    virtual ~Tensor() {}

    /**
     * @brief Default `%Tensor` objects will have a zero-initialized scalar
     * complex data-value, and a given size of `1`. Shape and indices for the
     * tensor will not be set.
     *
     */
    Tensor() : data_(1) {}

    /**
     * @brief Initialize a `%Tensor` object with a given shape and
     * zero-initialized complex data-values, and a given size of
     * (\f$\prod_i{\textrm{shape}_i}\f$). Indices for the tensor will default to
     * values from the set `?[a-zA-Z]`.
     *
     * @param shape Dimension of each `%Tensor` index.
     */
    Tensor(const std::vector<size_t> &shape)
        : data_(TensorHelpers::ShapeToSize(shape), {0.0, 0.0})
    {
        using namespace Utilities;
        std::vector<std::string> indices(shape.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = "?" + GenerateStringIndex(i);
        }

        InitIndicesAndShape(indices, shape);
    }

    /**
     * @brief Initialize a `%Tensor` object with a given shape,
     * zero-initialized complex data-values, a given size of
     * (\f$\prod_i{\textrm{shape}_i}\f$), and labeled indices.
     *
     * @param indices Index labels for `%Tensor` object.
     * @param shape Dimension of each `%Tensor` object index.
     */
    Tensor(const std::vector<std::string> &indices,
           const std::vector<size_t> &shape)
        : data_(TensorHelpers::ShapeToSize(shape), {0.0, 0.0})
    {
        InitIndicesAndShape(indices, shape);
    }

    /**
     * @brief Initialize a `%Tensor` object with a given shape, complex
     * data-values, a given size of (\f$\prod_i{\textrm{shape}_i}\f$), and
     * labeled indices.
     *
     * @param indices Index labels of the `%Tensor` object.
     * @param dimensions Size of each rank (index) of the `%Tensor` object.
     * @param data Row-major encoded complex data representation of the
     * `%Tensor` object.
     */
    Tensor(const std::vector<std::string> &indices,
           const std::vector<size_t> &dimensions, const std::vector<T> &data)
        : Tensor(indices, dimensions)
    {
        Utilities::FastCopy(data, data_);
    }

    /**
     * @brief Construct a new `%Tensor` object using copy constructor.
     *
     * @param other `%Tensor` object to copy state from.
     */
    Tensor(const Tensor &other)
    {
        InitIndicesAndShape(other.GetIndices(), other.GetShape());
        Utilities::FastCopy(other.GetData(), GetData());
    }

    /**
     * @brief Construct a new `%Tensor` object using move constructor.
     *
     * @param other `%Tensor` of which to take ownership.
     */
    Tensor(Tensor &&other)
    {
        indices_ = std::move(other.indices_);
        shape_ = std::move(other.shape_);
        index_to_dimension_ = std::move(other.index_to_dimension_);
        data_ = std::move(other.data_);
    }

    /**
     * @brief Set the `%Tensor` object shape. This defines the number of
     * elements per rank (index).
     *
     * @param shape Vector of elements per rank (index).
     */
    void SetShape(const std::vector<size_t> &shape) { shape_ = shape; }

    /**
     * @brief Get the `%Tensor` shape.
     *
     */
    const std::vector<size_t> &GetShape() const { return shape_; }

    /**
     * @brief Return reference to `%Tensor` object data at a given index.
     *
     * @warning Supplying an index greater than or equal to `%GetSize()` is
     *          undefined behaviour.
     *
     * @param local_index 1D data row-major index (lexicographic ordering).
     */
    T &operator[](size_t local_index) { return data_[local_index]; }

    /**
     * @see `operator[](size_t local_index)`.
     */
    const T &operator[](size_t local_index) const { return data_[local_index]; }

    /**
     * @brief Change `%Tensor` index label at given location.
     *
     * @param ind Location of `%Tensor` index label.
     * @param new_string New `%Tensor` index label.
     */
    void RenameIndex(size_t ind, std::string new_string)
    {

        std::string old_string = GetIndices()[ind];

        indices_[ind] = new_string;
        index_to_dimension_[new_string] = index_to_dimension_[old_string];
        index_to_dimension_.erase(old_string);
    }

    /**
     * @brief Equality operator for `%Tensor` objects.
     *
     * @param other `%Tensor` object to compare from.
     */
    bool operator==(const Tensor<T> &other) const noexcept
    {
        return shape_ == other.GetShape() && indices_ == other.GetIndices() &&
               index_to_dimension_ == other.GetIndexToDimension() &&
               data_ == other.GetData();
    }

    /**
     * @brief Inequality operator for `%Tensor` objects.
     *
     * @param other `%Tensor` object to compare from.
     */
    bool operator!=(const Tensor<T> &other) const { return !(*this == other); }

    /**
     * @brief Assignment operator for `%Tensor` objects.
     *
     * @param other `%Tensor` object to assign from.
     */
    const Tensor<T> &operator=(const Tensor<T> &other)
    {
        InitIndicesAndShape(other.GetIndices(), other.GetShape());
        Utilities::FastCopy(other.GetData(), GetData());
        return *this;
    }

    /**
     * @brief Assignment operator for `%Tensor` objects using move semantics.
     *
     * @param other `%Tensor` object to take ownership from.
     */
    const Tensor<T> &operator=(Tensor<T> &&other)
    {
        indices_ = std::move(other.indices_);
        shape_ = std::move(other.shape_);
        index_to_dimension_ = std::move(other.index_to_dimension_);
        data_ = std::move(other.data_);
        return *this;
    }

    /**
     * @brief Return mapping from `%Tensor` index label to dimension.
     *
     */
    const std::unordered_map<std::string, size_t> &GetIndexToDimension() const
    {
        return index_to_dimension_;
    }

    /**
     * @brief Set the `%Tensor` data value for a given n-dimensional index.
     *
     * @tparam V Type of data values stored by `%Tensor`.
     * @param indices n-dimensional `%Tensor` data index.
     * @param val Data value to set at given index.
     */
    template <class V>
    void SetValue(const std::vector<size_t> &indices, const V &val)
    {
        data_[Jet::Utilities::RavelIndex(indices, GetShape())] = val;
    }

    /**
     * @brief Get the `%Tensor` data value at given tensor index.
     *
     * @param indices n-dimensional `%Tensor` data index.
     */
    T GetValue(const std::vector<size_t> &indices) const
    {
        return data_[Jet::Utilities::RavelIndex(indices, GetShape())];
    }

    /**
     * @brief Get the `%Tensor` data in row-major order.
     *
     */
    const std::vector<T> &GetData() const { return data_; }

    /**
     * @see const std::vector<T> &GetData() const.
     *
     */
    std::vector<T> &GetData() { return data_; }

    /**
     * @brief Get the `%Tensor` index labels.
     *
     */
    const std::vector<std::string> &GetIndices() const { return indices_; }

    /**
     * @brief Get the total number of data elements in `%Tensor` object.
     *
     */
    size_t GetSize() const { return TensorHelpers::ShapeToSize(shape_); }

    /**
     * @brief Get a single scalar value from the `%Tensor` object data.
     * Equivalent to `%GetValue({})`.
     *
     */
    const T &GetScalar() { return data_[0]; }

    /**
     * @brief Randomly assign values to `%Tensor` object data. This method
     * will allow for reproducible random number generation with a given seed.
     *
     * @param seed Seed the RNG with a given value.
     */
    void FillRandom(size_t seed)
    {
        std::mt19937 mt_engine(seed);
        std::uniform_real_distribution<typename T::value_type> r_dist(-1, 1);

        for (size_t i = 0; i < GetSize(); i++) {
            data_[i] = {r_dist(mt_engine), r_dist(mt_engine)};
        }
    }

    /**
     * @brief Randomly assign values to `%Tensor` object data.
     *
     */
    void FillRandom()
    {
        static std::mt19937 mt_engine(std::random_device{}());
        static std::uniform_real_distribution<typename T::value_type> r_dist(-1,
                                                                             1);

        for (size_t i = 0; i < GetSize(); i++) {
            data_[i] = {r_dist(mt_engine), r_dist(mt_engine)};
        }
    }

    /**
     * @brief Inform whether the tensor is rank-0 (true) or otherwise (false).
     *
     */
    bool IsScalar() const noexcept { return GetSize() == 1; }
};

/**
 * @brief Contract given `%Tensor` objects over the intersection of the index
 * sets. The resulting tensor will be formed with indices given by the symmetric
 * difference of the index sets.
 *
 * Example: Given a 3x2x4 tensor A(i,j,k) and a 2x4x2 tensor B(j,k,l), the
 * common indices are (j,k), and the symmetric difference of the sets are (i,l).
 * The result of the contraction will be a tensor 3x2 tensor C(i,l).
 * \code{.cpp}
 *     Tensor A({"i", "j", "k"}, {3, 2, 4});
 *     Tensor B({"j", "k", "l"}, {2, 4, 2});
 *     A.FillRandom();
 *     B.FillRandom();
 *     C = ContractTensors(A, B);
 * \endcode
 *
 * @see TODO: Link to documentation
 *
 * @tparam T `%Tensor` data type.
 * @param A Left tensor to contract.
 * @param B Right tensor to contract.
 */
template <class T> Tensor<T> ContractTensors(Tensor<T> &A, Tensor<T> &B)
{

    using namespace Jet::Utilities;
    using namespace Jet::TensorHelpers;

    auto &&left_indices = VectorSubtraction(A.GetIndices(), B.GetIndices());
    auto &&right_indices = VectorSubtraction(B.GetIndices(), A.GetIndices());
    auto &&common_indices = VectorIntersection(A.GetIndices(), B.GetIndices());

    size_t left_dim = 1, right_dim = 1, common_dim = 1;
    for (size_t i = 0; i < left_indices.size(); ++i) {
        left_dim *= A.GetIndexToDimension().at(left_indices[i]);
    }
    for (size_t i = 0; i < right_indices.size(); ++i) {
        right_dim *= B.GetIndexToDimension().at(right_indices[i]);
    }
    for (size_t i = 0; i < common_indices.size(); ++i) {
        size_t a_dim = A.GetIndexToDimension().at(common_indices[i]);
        common_dim *= a_dim;
    }

    auto &&a_new_ordering = VectorUnion(left_indices, common_indices);
    auto &&b_new_ordering = VectorUnion(common_indices, right_indices);

    auto &&C_indices = VectorUnion(left_indices, right_indices);
    std::vector<size_t> C_dimensions(C_indices.size());
    for (size_t i = 0; i < left_indices.size(); ++i)
        C_dimensions[i] = A.GetIndexToDimension().at(left_indices[i]);
    for (size_t i = 0; i < right_indices.size(); ++i)
        C_dimensions[i + left_indices.size()] =
            B.GetIndexToDimension().at(right_indices[i]);

    Tensor<T> C(C_indices, C_dimensions);
    auto &&At = Transpose(A, a_new_ordering);
    auto &&Bt = Transpose(B, b_new_ordering);

    TensorHelpers::MultiplyTensorData<T>(
        At.GetData(), Bt.GetData(), C.GetData(), left_indices, right_indices,
        left_dim, right_dim, common_dim);

    return C;
}

/**
 * @brief Perform slicing of given tensor index. This will return a `%Tensor`
 * object whose given indices and data are a subset of the provided tensor
 * object, sliced along the given index argument.
 *
 * Example: Given a 2x3 tensor `%A(i,j)`. We can slice along any of the given
 * indices, and return a Tensor object of a given slice, indexed relative to
 * the ranges given by `%RavelIndex()`. The following example slices along each
 * index, with the resulting slices selected as required.
 * \code{.cpp}
 *     Tensor A({"i", "j"}, {2, 3});
 *     A.FillRandom();
 *
 *     SliceIndex(A, "i", 0);  // [1x3] tensor, slice 0
 *     SliceIndex(A, "i", 1);  // [1x3] tensor, slice 1

 *     SliceIndex(A, "j", 0);  // [2x1] tensor, slice 0
 *     SliceIndex(A, "j", 1);  // [2x1] tensor, slice 1
 *     SliceIndex(A, "j", 2);  // [2x1] tensor, slice 2
 * \endcode
 *
 * @tparam T `%Tensor` data type.
 * @param tensor `%Tensor` object to slice.
 * @param index_str Tensor index label on which to slice.
 * @param index_value Tensor slice to return from multidimensional indexing
 returned by `%RavelIndex()`.
 */
template <class T>
Tensor<T> SliceIndex(const Tensor<T> &tensor, const std::string &index_str,
                     size_t index_value)
{

    std::vector<std::string> new_ordering = tensor.GetIndices();
    auto it = find(new_ordering.begin(), new_ordering.end(), index_str);
    size_t index_num = std::distance(new_ordering.begin(), it);
    new_ordering.erase(new_ordering.begin() + index_num);
    new_ordering.insert(new_ordering.begin(), index_str);

    auto &&tensor_trans = Transpose(tensor, new_ordering);
    std::vector<std::string> sliced_indices(
        tensor_trans.GetIndices().begin() + 1, tensor_trans.GetIndices().end());
    std::vector<size_t> sliced_dimensions(tensor_trans.GetShape().begin() + 1,
                                          tensor_trans.GetShape().end());

    Tensor<T> tensor_sliced(sliced_indices, sliced_dimensions);
    size_t projection_size = tensor_sliced.GetSize();
    size_t projection_begin = projection_size * index_value;
    auto data_ptr = tensor_trans.GetData();

#if defined _OPENMP
    int max_right_dim = 1024;
#pragma omp parallel for schedule(static, max_right_dim)
#endif
    for (size_t p = 0; p < projection_size; ++p)
        tensor_sliced[p] = tensor_trans[projection_begin + p];

    return tensor_sliced;
}

/**
 * @brief Reshape `%Tensor` object to new given dimensions.
 *
 * @tparam T `%Tensor` data type.
 * @param old_tensor Original tensor object to reshape.
 * @param new_shape Index dimensionality for new tensor object.
 */
template <class T>
Tensor<T> Reshape(const Tensor<T> &old_tensor,
                  const std::vector<size_t> &new_shape)
{
    using namespace Utilities;

    JET_ABORT_IF_NOT(old_tensor.GetSize() ==
                         TensorHelpers::ShapeToSize(new_shape),
                     "Size is inconsistent between tensors.");
    Tensor<T> new_tensor(new_shape);
    Utilities::FastCopy(old_tensor.GetData(), new_tensor.GetData());
    return new_tensor;
}

/**
 * @brief Perform conjugation of complex data in `%Tensor`.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 */
template <class T> Tensor<T> Conj(const Tensor<T> &A)
{
    Tensor<T> A_conj(A.GetIndices(), A.GetShape());
    for (size_t i = 0; i < A.GetSize(); i++) {
        A_conj[i] = std::conj(A[i]);
    }
    return A_conj;
}

/**
 * @brief Transpose `%Tensor` indices to new ordering.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 * @param new_indices New `%Tensor` index label ordering.
 * @return Transposed tensor.
 */
template <class T>
Tensor<T> Transpose(const Tensor<T> &A,
                    const std::vector<std::string> &new_indices)
{
    using namespace Jet::Utilities;

    auto indices_ = A.GetIndices();
    auto shape_ = A.GetShape();

    if (new_indices == indices_)
        return A;

    std::vector<std::string> old_ordering(indices_);
    std::vector<size_t> old_dimensions(shape_);
    size_t num_indices = old_ordering.size();
    size_t total_dim = A.GetSize();

    if (num_indices == 0)
        JET_ABORT("Number of indices cannot be zero.");

    // Create map_old_to_new_idxpos from old to new indices, and new_dimensions.
    std::vector<size_t> map_old_to_new_idxpos(num_indices);
    std::vector<size_t> new_dimensions(num_indices);
    for (size_t i = 0; i < num_indices; ++i) {
        for (size_t j = 0; j < num_indices; ++j) {
            if (old_ordering[i] == new_indices[j]) {
                map_old_to_new_idxpos[i] = j;
                new_dimensions[j] = old_dimensions[i];
                break;
            }
        }
    }

    // Create super dimensions (combined dimension of all to the right of i).
    std::vector<size_t> old_super_dimensions(num_indices);
    std::vector<size_t> new_super_dimensions(num_indices);
    old_super_dimensions[num_indices - 1] = 1;
    new_super_dimensions[num_indices - 1] = 1;

    size_t old_dimensions_size = old_dimensions.size();
    if (old_dimensions_size >= 2)
        for (size_t i = old_dimensions_size; --i;) {
            old_super_dimensions[i - 1] =
                old_super_dimensions[i] * old_dimensions[i];
            new_super_dimensions[i - 1] =
                new_super_dimensions[i] * new_dimensions[i];
        }

    std::vector<unsigned short int> small_map_old_to_new_position(
        MAX_RIGHT_DIM);

    Tensor<T> At(new_indices, new_dimensions, A.GetData());
    // No combined efficient mapping from old to new positions with actual
    // copies in memory, all in small cache friendly (for old data, not new,
    // which could be very scattered) blocks.

    // Position old and new.
    size_t po = 0, pn;
    // Counter of the values of each indices in the iteration (old ordering).
    std::vector<size_t> old_counter(num_indices, 0);
    // offset is important when doing this in blocks, as it's indeed
    // implemented.
    size_t offset = 0;
    // internal_po keeps track of interations within a block.
    // Blocks have size MAX_RIGHT_DIM.
    size_t internal_po = 0;

    auto data = At.GetData().data();
    auto scratch = A.GetData().data();
    // External loop loops over blocks.
    while (true) {
        // If end of entire opration, break.
        if (po == total_dim - 1)
            break;

        internal_po = 0;
        // Each iteration of the while block goes through a new position.
        // Inside the while, j takes care of increasing indices properly.
        while (true) {
            po = 0;
            pn = 0;
            for (size_t i = 0; i < num_indices; ++i) {
                po += old_super_dimensions[i] * old_counter[i];
                pn += new_super_dimensions[map_old_to_new_idxpos[i]] *
                      old_counter[i];
            }
            small_map_old_to_new_position[po - offset] = pn;

            bool complete{true};
            for (size_t j = num_indices; j--;) {
                if (++old_counter[j] < old_dimensions[j]) {
                    complete = false;
                    break;
                }
                else
                    old_counter[j] = 0;
            }
            // If end of block or end of entire operation, break.
            if ((++internal_po == MAX_RIGHT_DIM) || (po == total_dim - 1))
                break;
            // If last index (0) was increased, then go back to fastest index.
            if (complete)
                break;
        }

        // Copy data for this block, taking into account offset of small_map...
        // The following line is to avoid casting MAX_RIGHT_DIM to size_t
        // every iteration. Note that it has to be size_t for min to work,
        // since total_dim is size_t.
        size_t effective_max = std::min((size_t)MAX_RIGHT_DIM, total_dim);
        for (size_t p = 0; p < effective_max; ++p)
            *(data + small_map_old_to_new_position[p]) =
                *(scratch + offset + p);

        offset += MAX_RIGHT_DIM;
    }

    return At;
}

/**
 * @brief Transpose `%Tensor` indices to new ordering.
 *
 * @warning The program is aborted if the number of elements in the new ordering
 *          does match the number of indices in the tensor.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 * @param new_ordering New ordering of the `%Tensor` index labels.
 * @return Transposed tensor.
 */
template <class T>
Tensor<T> Transpose(const Tensor<T> &A, const std::vector<size_t> &new_ordering)
{
    const size_t num_indices = A.GetIndices().size();
    JET_ABORT_IF_NOT(num_indices == new_ordering.size(),
                     "Size of ordering must match number of tensor indices.");

    std::vector<std::string> new_indices(num_indices);
    const auto &old_indices = A.GetIndices();

    for (size_t i = 0; i < num_indices; i++) {
        new_indices[i] = old_indices[new_ordering[i]];
    }

    return Transpose(A, new_indices);
}

/**
 * @brief Streams a tensor to an output stream.
 *
 * @param out Output stream to be modified.
 * @param tensor Tensor to be streamed.
 */
template <class T>
inline std::ostream &operator<<(std::ostream &out, const Tensor<T> &tensor)
{
    using namespace Jet::Utilities;

    out << "Size=" << tensor.GetSize() << std::endl;
    out << "Indices=" << tensor.GetIndices() << std::endl;
    out << "Data=" << tensor.GetData() << std::endl;

    return out;
}

}; // namespace Jet
