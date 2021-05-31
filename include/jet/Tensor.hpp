#pragma once

#include <complex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "TensorHelpers.hpp"
#include "Utilities.hpp"
#include "permute/PermuterIncludes.hpp"

namespace Jet {

/**
 * @brief `%Tensor` represents an \f$n\f$-rank data structure of complex-valued
 *        data for tensor operations.
 *
 * The following conventions are used:
 *
 *     - "Rank" and "order" are used interchangeably and refer to the number of
 *       tensor indices.
 *     - "Dimension" refers to the number of elements along a tensor index.
 *     - "Shape" refers to the dimensions of a tensor; the number of dimensions
 *       is the rank of the tensor.
 *
 * @tparam T Underlying complex tensor data type (`complex<float>` or
 *           `complex<double>`).
 */
template <class T = std::complex<float>> class Tensor;

/**
 * @brief Adds two `%Tensor` objects with the same index sets.
 *
 * The resulting tensor will have the same index set as the operand tensors. The
 * order of the indices follows that of the first argument (i.e., `A`).
 *
 * Example: Given a 2x3 tensor A(i,j) and a 2x3 tensor B(i,j), the addition of
 * A and B is a 2x3 tensor C(i,j):
 * \code{.cpp}
 *     Tensor A({"i", "j"}, {2, 3}, {0, 1, 2, 3, 4, 5});
 *     Tensor B({"i", "j}, {2, 3}, {5, 5, 5, 6, 6, 6});
 *     Tensor C = AddTensors(A, B);  // {5, 6, 7, 9, 10, 11}
 * \endcode
 *
 * @warning The program is aborted if the index sets of the given `%Tensor`
 *          objects to not match.
 *
 * @tparam T `%Tensor` data type.
 * @param A tensor on the LHS of the addition.
 * @param B tensor on the RHS of the addition.
 * @return `%Tensor` object representing the element-wise sum of the given
 *         tensors.
 */
template <class T> Tensor<T> AddTensors(const Tensor<T> &A, const Tensor<T> &B)
{
    const auto disjoint_indices =
        Jet::Utilities::VectorDisjunctiveUnion(A.GetIndices(), B.GetIndices());

    JET_ABORT_IF_NOT(disjoint_indices.empty(),
                     "Tensor addition with disjoint indices is not supported.");

    const auto &indices = A.GetIndices();
    const auto &shape = A.GetShape();

    // Align the underlying data vectors of `A` and `B`.
    const auto &&Bt = Transpose(B, indices);

    Tensor<T> C(indices, shape);
    const auto size = C.GetSize();

    auto c_ptr = C.GetData().data();
    auto a_ptr = A.GetData().data();
    auto bt_ptr = Bt.GetData().data();

#if defined _OPENMP
#pragma omp parallel for schedule(static, 1024)//MAX_RIGHT_DIM)
#endif
    for (size_t i = 0; i < size; i++) {
        c_ptr[i] = a_ptr[i] + bt_ptr[i];
    }

    return C;
}

/**
 * @brief Slices a `%Tensor` object index.
 *
 * The result is a `%Tensor` object whose given indices and data are a subset of
 * the provided tensor object, sliced along the given index argument.
 *
 * Example: Consider a 2x3 tensor `A(i,j)`. The following example slices along
 * each index with the resulting slices selected as required:
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
 * @param index `%Tensor` index label on which to slice.
 * @param value Value to slice the `%Tensor` index on.
 * @return Slice of the `%Tensor` object.
 */
template <class T>
Tensor<T> SliceIndex(const Tensor<T> &tensor, const std::string &index,
                     size_t value)
{

    std::vector<std::string> new_ordering = tensor.GetIndices();
    auto it = find(new_ordering.begin(), new_ordering.end(), index);
    size_t index_num = std::distance(new_ordering.begin(), it);
    new_ordering.erase(new_ordering.begin() + index_num);
    new_ordering.insert(new_ordering.begin(), index);

    auto &&tensor_trans = Transpose(tensor, new_ordering);
    std::vector<std::string> sliced_indices(
        tensor_trans.GetIndices().begin() + 1, tensor_trans.GetIndices().end());
    std::vector<size_t> sliced_dimensions(tensor_trans.GetShape().begin() + 1,
                                          tensor_trans.GetShape().end());

    Tensor<T> tensor_sliced(sliced_indices, sliced_dimensions);
    size_t projection_size = tensor_sliced.GetSize();
    size_t projection_begin = projection_size * value;

    auto data_ptr = tensor_trans.GetData().data();
    auto tensor_sliced_ptr = tensor_sliced.GetData().data();

#if defined _OPENMP
     int max_right_dim = 1024;
#pragma omp parallel for schedule(static, max_right_dim)
#endif
    for (size_t p = 0; p < projection_size; ++p)
        tensor_sliced_ptr[p] = data_ptr[projection_begin + p];

    return tensor_sliced;
}

/**
 * @brief Reshapes a `%Tensor` object to the given dimensions.
 *
 * @tparam T `%Tensor` data type.
 * @param old_tensor Original tensor object to reshape.
 * @param new_shape Index dimensionality for new tensor object.
 * @return Reshaped copy of the `%Tensor` object.
 */
template <class T>
Tensor<T> Reshape(const Tensor<T> &old_tensor,
                  const std::vector<size_t> &new_shape)
{
    using namespace Utilities;

    JET_ABORT_IF_NOT(old_tensor.GetSize() ==
                         Jet::Utilities::ShapeToSize(new_shape),
                     "Size is inconsistent between tensors.");
    Tensor<T> new_tensor(new_shape);
    Utilities::FastCopy(old_tensor.GetData(), new_tensor.GetData());
    return new_tensor;
}

/**
 * @brief Returns the conjugate of a `%Tensor` object.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 * @return `%Tensor` object representing the conjugate of `A`.
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
 * @brief Transposes the indices of a `%Tensor` object to a new ordering.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 * @param new_indices New `%Tensor` index label ordering.
 * @return Transposed `%Tensor` object.
 */
template <class T, size_t BLOCKSIZE = 1024, size_t MINSIZE = 32>
Tensor<T> Transpose(const Tensor<T> &A,
                    const std::vector<std::string> &new_indices)
{
    using namespace Jet::Utilities;

    if (A.GetIndices() == new_indices)
        return A;

    if (A.GetIndices().size() == 0)
        JET_ABORT("Number of indices cannot be zero.");

    std::vector<size_t> new_shape(A.GetShape().size());
    for (size_t i = 0; i < new_indices.size(); i++)
        new_shape[i] = A.GetIndexToDimension().at(new_indices[i]);

    if (is_pow_2(A.GetSize())) {
        auto permuter = Permuter<QFlexPermuter<BLOCKSIZE, MINSIZE>>();
        return Tensor<T>{new_indices, new_shape,
                         permuter.Transpose(A.GetData(), A.GetShape(),
                                            A.GetIndices(), new_indices)};
    }
    auto permuter = Permuter<DefaultPermuter<BLOCKSIZE>>();
    return Tensor<T>{
        new_indices, std::move(new_shape),
        std::move(permuter.Transpose(A.GetData(), A.GetShape(), A.GetIndices(),
                                     new_indices))};
}

/**
 * @brief Transposes the indices of a `%Tensor` to a new ordering.
 *
 * @warning The program is aborted if the number of elements in the new ordering
 *          does match the number of indices in the tensor.
 *
 * @tparam T `%Tensor` data type.
 * @param A Reference `%Tensor` object.
 * @param new_ordering New `%Tensor` index permutation.
 * @return Transposed `%Tensor` object.
 */
template <class T, size_t BLOCKSIZE = 1024, size_t MINSIZE = 32>
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

    return Transpose<T,BLOCKSIZE,MINSIZE>(A, new_indices);
}


/**
 * @brief Contracts two `%Tensor` objects over the intersection of their index
 *        sets.
 *
 * The resulting tensor will be formed with indices given by the symmetric
 * difference of the index sets.
 *
 * Example: Given a 3x2x4 tensor A(i,j,k) and a 2x4x2 tensor B(j,k,l), the
 * common indices are (j,k) and the symmetric difference of the sets is (i,l).
 * The result of the contraction is a 3x2 tensor C(i,l).
 * \code{.cpp}
 *     Tensor A({"i", "j", "k"}, {3, 2, 4});
 *     Tensor B({"j", "k", "l"}, {2, 4, 2});
 *     A.FillRandom();
 *     B.FillRandom();
 *     Tensor C = ContractTensors(A, B);
 * \endcode
 *
 * @see TODO: Link to documentation
 *
 * @tparam T `%Tensor` data type.
 * @param A tensor on the LHS of the contraction.
 * @param B tensor on the RHS of the contraction.
 * @return `%Tensor` object representing the contraction of the tensors.
 */
template <class T>
Tensor<T> ContractTensors(const Tensor<T> &A, const Tensor<T> &B)
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
    auto &&At = Transpose<T,1024UL,32UL>(A, a_new_ordering);
    auto &&Bt = Transpose<T,1024UL,32UL>(B, b_new_ordering);

    TensorHelpers::MultiplyTensorData<T>(
        At.GetData(), Bt.GetData(), C.GetData(), left_indices, right_indices,
        left_dim, right_dim, common_dim);

    return C;
}

/**
 * @brief Streams a tensor to an output stream.
 *
 * @param out Output stream to be modified.
 * @param tensor Tensor to be streamed.
 * @return Reference to the given output stream.
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

// Default template parameters belong in the forward declaration.
template <class T> class Tensor {

    static_assert(TensorHelpers::is_supported_data_type<T>,
                  "Tensor data type must be one of std::complex<float>, "
                  "std::complex<double>");

  public:
    /// Type of the real and imaginary components of the tensor data.
    using scalar_type_t = T;

    /**
     * @see Jet::AddTensors().
     */
    static constexpr auto &AddTensors = Jet::AddTensors<T>;

    /**
     * @see Jet::ContractTensors().
     */
    static constexpr auto &ContractTensors = Jet::ContractTensors<T>;

    /**
     * @see Jet::Reshape().
     */
    static constexpr auto &Reshape = Jet::Reshape<T>;

    /**
     * @see Jet::SliceIndex().
     */
    static constexpr auto &SliceIndex = Jet::SliceIndex<T>;

    /**
     * @brief Constructs a default `%Tensor` object.
     *
     * Default tensor objects have a single zero-initialized data value.
     *
     * @warning The shape and indices of a default `%Tensor` object are not set.
     */
    Tensor() : data_(1) {}

    /**
     * @brief Constructs a shaped `%Tensor` object.
     *
     * Shaped `%Tensor` objects have zero-initialized data values and a size of
     * (\f$\prod_i{\textrm{shape}_i}\f$). The indices of a shaped `%Tensor`
     * object default to the values from the set `?[a-zA-Z]`.
     *
     * @param shape Dimension of each `%Tensor` index.
     */
    Tensor(const std::vector<size_t> &shape)
        : data_(Jet::Utilities::ShapeToSize(shape))
    {
        using namespace Utilities;
        std::vector<std::string> indices(shape.size());
        for (size_t i = 0; i < indices.size(); i++) {
            indices[i] = "?" + GenerateStringIndex(i);
        }

        InitIndicesAndShape(indices, shape);
    }

    /**
     * @brief Constructs a shaped and labeled `%Tensor` object.
     *
     * Shaped and labeled `%Tensor` objects have zero-initialized data values
     * and a size of (\f$\prod_i{\textrm{shape}_i}\f$).
     *
     * @param indices Label of each `%Tensor` index.
     * @param shape Dimension of each `%Tensor` index.
     */
    Tensor(const std::vector<std::string> &indices,
           const std::vector<size_t> &shape)
        : data_(Jet::Utilities::ShapeToSize(shape))
    {
        InitIndicesAndShape(indices, shape);
    }

    /**
     * @brief Constructs a shaped, labeled, and populated `%Tensor` object.
     *
     * The size of a shaped, indexed, and populated `%Tensor` object is
     * (\f$\prod_i{\textrm{shape}_i}\f$).
     *
     * @param indices Label of each `%Tensor` index.
     * @param shape Dimension of each `%Tensor` index.
     * @param data Row-major encoded complex data representation of the
     *             `%Tensor` object.
     */
    Tensor(const std::vector<std::string> &indices,
           const std::vector<size_t> &shape, const std::vector<T> &data)
        : Tensor(indices, shape)
    {
        Utilities::FastCopy(data, data_);
    }

    /**
     * @brief Constructs a `%Tensor` object by copying another `%Tensor` object.
     *
     * @param other `%Tensor` object to be copied.
     */
    Tensor(const Tensor &other)
    {
        InitIndicesAndShape(other.GetIndices(), other.GetShape());
        Utilities::FastCopy(other.data_, data_);
    }

    /**
     * @brief Constructs a `%Tensor` object by moving another `%Tensor` object.
     *
     * @param other `%Tensor` object to be moved.
     */
    Tensor(Tensor &&other)
    {
        indices_ = std::move(other.indices_);
        shape_ = std::move(other.shape_);
        index_to_dimension_ = std::move(other.index_to_dimension_);
        data_ = std::move(other.data_);
    }

    /**
     * @brief Destructs this `%Tensor` object.
     */
    virtual ~Tensor() {}

    /**
     * @brief Initializes the indices and shape of a `%Tensor` object.
     *
     * The indices and shapes must be ordered to map directly such that
     * `indices[i]` has size `shape[i]`.
     *
     * @note This function updates the internal index-to-dimension map.
     *
     * @param indices Label of each `%Tensor` index.
     * @param shape Dimension of each `%Tensor` index.
     */
    void InitIndicesAndShape(const std::vector<std::string> &indices,
                             const std::vector<size_t> &shape) noexcept
    {
        indices_ = indices;
        shape_ = shape;

        index_to_dimension_.clear();
        for (size_t i = 0; i < shape_.size(); ++i)
            index_to_dimension_[indices[i]] = shape[i];
    }

    /**
     * @brief Sets the shape of a `%Tensor` object.
     *
     * The shape of a `%Tensor` defines the number of elements per rank (index).
     *
     * @param shape Number of elements in each `%Tensor` index.
     */
    void SetShape(const std::vector<size_t> &shape) noexcept { shape_ = shape; }

    /**
     * @brief Returns the shape of a `%Tensor` tensor object.
     *
     * @return Number of elements in each `%Tensor` index.
     */
    const std::vector<size_t> &GetShape() const noexcept { return shape_; }

    /**
     * @brief Returns a reference to a `%Tensor` object datum.
     *
     * @warning Supplying an index greater than or equal to `%GetSize()` is
     *          undefined behaviour.
     *
     * @param pos Position of the datum to retrieve, encoded as a 1D row-major
     *            index (lexicographic ordering).
     *
     * @return Reference to the complex data value at the specified position.
     */
    T &operator[](size_t pos) { return data_[pos]; }

    /**
     * @see `operator[](size_t pos)`.
     */
    const T &operator[](size_t pos) const { return data_[pos]; }

    /**
     * @brief Renames a `%Tensor` index label.
     *
     * @param pos Position of the `%Tensor` label.
     * @param new_label New `%Tensor` label.
     */
    void RenameIndex(size_t pos, std::string new_label) noexcept
    {
        const auto old_label = indices_[pos];
        const auto dimension = index_to_dimension_[old_label];
        index_to_dimension_.erase(old_label);

        indices_[pos] = new_label;
        index_to_dimension_.emplace(new_label, dimension);
    }

    /**
     * @brief Equality operator for `%Tensor` objects.
     *
     * @param other `%Tensor` object to be compared to this `%Tensor` object.
     * @return True if the two `%Tensor` objects are equivalent.
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
     * @param other `%Tensor` object to be compared to this `%Tensor` object.
     * @return True if the two `%Tensor` objects are not equivalent.
     */
    bool operator!=(const Tensor<T> &other) const { return !(*this == other); }

    /**
     * @brief Assignment operator for `%Tensor` objects.
     *
     * @param other `%Tensor` object to be assigned from.
     * @return Reference to this `%Tensor` object.
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
     * @param other `%Tensor` object to take ownership of.
     * @return Reference to this `%Tensor` object.
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
     * @brief Returns the index-to-dimension map.
     *
     * @returns Mapping from `%Tensor` index labels to dimension sizes.
     */
    const std::unordered_map<std::string, size_t> &GetIndexToDimension() const
    {
        return index_to_dimension_;
    }

    /**
     * @brief Sets the `%Tensor` data value at the given n-dimensional index.
     *
     * @param indices n-dimensional `%Tensor` data index in row-major order.
     * @param value Data value to set at given index.
     */
    void SetValue(const std::vector<size_t> &indices, const T &value)
    {
        data_[Jet::Utilities::RavelIndex(indices, shape_)] = value;
    }

    /**
     * @brief Returns the `%Tensor` data value at the given n-dimensional index.
     *
     * @param indices n-dimensional `%Tensor` data index in row-major order.
     *
     * @returns Complex data value.
     */
    T GetValue(const std::vector<size_t> &indices) const
    {
        return data_[Jet::Utilities::RavelIndex(indices, shape_)];
    }

    /**
     * @brief Returns the `%Tensor` data in row-major order.
     *
     * @return Vector of complex data values.
     */
    const std::vector<T> &GetData() const noexcept { return data_; }

    /**
     * @see GetData().
     */
    std::vector<T> &GetData() { return data_; }

    /**
     * @brief Returns the `%Tensor` index labels.
     *
     * @return Vector of index labels.
     */
    const std::vector<std::string> &GetIndices() const noexcept
    {
        return indices_;
    }

    /**
     * @brief Returns the size of a `%Tensor` object.
     *
     * @return Number of data elements.
     */
    size_t GetSize() const { return Jet::Utilities::ShapeToSize(shape_); }

    /**
     * @brief Returns a single scalar value from the `%Tensor` object.
     *
     * @note This is equivalent to calling `%GetValue({})`.
     *
     * @return Complex data value.
     */
    const T &GetScalar() const { return data_[0]; }

    /**
     * @brief Reports whether a `%Tensor` object is a scalar.
     *
     * @return True if this `%Tensor` object is rank-0 (and false otherwise).
     */
    bool IsScalar() const noexcept { return GetSize() == 1; }

    /**
     * @brief Assigns random values to the `%Tensor` object data.
     *
     * The real and imaginary components of each datum will be independently
     * sampled from a uniform distribution with support over [-1, 1].
     *
     * @note This overload enables reproducible random number generation for a
     *        given seed.
     *
     * @param seed Seed to supply to the RNG engine.
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
     * @brief Assigns random values to the `%Tensor` object data.
     *
     * The real and imaginary components of each datum will be independently
     * sampled from a uniform distribution with support over [-1, 1].
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

  private:
    /// Index labels.
    std::vector<std::string> indices_;

    /// Dimension along each index.
    std::vector<size_t> shape_;

    /// Mapping from index labels to dimensions.
    std::unordered_map<std::string, size_t> index_to_dimension_;

    /// Complex data values in row-major order.
    std::vector<T> data_;
};

}; // namespace Jet

