#pragma once

#include <complex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Abort.hpp"
#include "Tensor.hpp"
#include "Utilities.hpp"

namespace Jet {

//#define IS_POW_2(a) static_cast<bool>(v && !(v & (v - 1)));

/**
 * @brief Interface for tensor permutation backend.
 *
 * @tparam PermuteBackend
 */
template <class PermuteBackend> class PermuteBase {
  protected:
    using DataType = std::complex<float>;

    // Ensure derived type recognised as friend for CRTP
    friend PermuteBackend;

  public:
    PermuteBase() = default;

    void Transpose(const std::vector<DataType> &data_in,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        static_cast<PermuteBackend &>(*this).Transpose(
            data_in, data_out, current_order, new_order);
    }
};

class QFPermute : public PermuteBase<QFPermute> {
    using DataType = std::complex<float>;

    enum class PermuteType { PermuteLeft, PermuteRight };

  public:
    QFPermute() : PermuteBase<QFPermute>() {}

    /**
     * @brief Holds details and structures for planning the transpose.
     *
     */
    struct PermutePlan {
        size_t dim_left;
        size_t dim_right;
        size_t tensor_dim;
        size_t old_dimensions;
        size_t new_dimensions;

        PermuteType type;

        std::vector<std::string> new_ordering;
        std::vector<std::string> old_ordering;

        std::vector<size_t> map_old_to_new_position;

        PermutePlan(size_t dim_left, size_t dim_right, size_t tensor_dim,
                    size_t old_dimensions, size_t new_dimensions,
                    PermuteType type, std::vector<std::string> new_ordering,
                    std::vector<std::string> old_ordering,
                    std::vector<size_t> map_old_to_new_position)
            : dim_left(dim_left), dim_right(dim_right), tensor_dim(tensor_dim),
              old_dimensions(old_dimensions), new_dimensions(new_dimensions),
              type(type), new_ordering(new_ordering),
              old_ordering(old_ordering),
              map_old_to_new_position(map_old_to_new_position){};
    };

    using plan_type = PermutePlan;

    /**
     * @brief Tranpose the tensor data to the new index set.
     *
     * @note If input and output data arrays are the same, an in-place transpose
     * will be used.
     *
     * @param data_in Data of tensor object to tranpose.
     * @param data_out Output data object of tranposed tensor data.
     * @param current_order Existing index order of tensor.
     * @param new_order New index order of the `data_out` tensor data.
     */
    void Transpose(const std::vector<DataType> &data_in,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        if (&data_in == &data_out) {
            std::vector<size_t> indices_old(current_order.size());
            std::vector<size_t> indices_new(current_order.size());
            for (size_t i = 0; i < indices_old.size(); i++) {
                indices_old[i] = i;
                indices_new[i] = new_order.size() - (i + 1);
            }
            TransposeInPlace_(data_out, indices_old, indices_new);
        }
        else {
        }
    }

    struct PrecomputedQflexTransposeData {
        std::vector<PermutePlan> transplans;
        size_t total_dim;
        bool no_transpose;
        bool log_2;
    };

    void TransposeInPlace_(std::vector<DataType> &data,
                           const std::vector<size_t> &current_order,
                           const std::vector<size_t> &new_order)
    {
        data[0] *= 10;
        for (auto &a : current_order)
            std::cout << a << std::endl;
        for (auto &a : new_order)
            std::cout << a << std::endl;
    }

    /**
     * @brief Right-partition data permutation following
     * https://arxiv.org/pdf/1811.09599.pdf
     *
     * @param old_ordering
     * @param new_ordering
     * @param num_indices_right
     * @param tensor_size
     * @param precomputed_data
     */
    void PrecomputeRightTransposeData_(
        const std::vector<std::string> &old_ordering,
        const std::vector<std::string> &new_ordering,
        std::size_t num_indices_right, size_t tensor_size,
        PrecomputedQflexTransposeData &precomputed_data)
    {
        std::cout << num_indices_right << std::endl;

        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new
        // indices, old_dimensions, new_dimensions, and total_dim.
        std::size_t dim = 2;
        std::size_t num_indices = old_ordering.size();
        std::vector<std::size_t> map_old_to_new_idxpos(num_indices);
        std::vector<std::size_t> old_dimensions(num_indices, dim);
        std::vector<std::size_t> new_dimensions(num_indices, dim);

        std::size_t total_dim = 1;
        for (std::size_t i = 0; i < num_indices; ++i) {
            total_dim *= old_dimensions[i];
        }

        for (std::size_t i = 0; i < num_indices; ++i) {
            for (std::size_t j = 0; j < num_indices; ++j) {
                if (old_ordering[i] == new_ordering[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        std::vector<std::size_t> map_old_to_new_position(total_dim);
        GenerateBinaryReorderingMap(map_old_to_new_idxpos,
                                    map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to reorder within
        // small chuncks.
        std::size_t dim_right = total_dim;
        std::size_t dim_left =
            tensor_size / dim_right; // Remember, it's all powers of 2, so OK.

        PermutePlan plan{dim_left,     dim_right,    tensor_size,
                         dim,          dim,          PermuteType::PermuteRight,
                         new_ordering, old_ordering, map_old_to_new_position};
        precomputed_data.transplans.push_back(plan);
    }

    void PrecomputeLeftTransposeData_(
        const std::vector<std::string> &old_ordering,
        const std::vector<std::string> &new_ordering,
        std::size_t num_indices_right, size_t tensor_size,
        PrecomputedQflexTransposeData &precomputed_data)
    {
        std::cout << num_indices_right << std::endl;

        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new
        // indices, old_dimensions, new_dimensions, and total_dim.
        std::size_t dim = 2;
        std::size_t num_indices = old_ordering.size();
        std::vector<std::size_t> map_old_to_new_idxpos(num_indices);
        std::vector<std::size_t> old_dimensions(num_indices, dim);
        std::vector<std::size_t> new_dimensions(num_indices, dim);
        std::size_t total_dim = 1;
        for (std::size_t i = 0; i < num_indices; ++i)
            total_dim *= old_dimensions[i];
        for (std::size_t i = 0; i < num_indices; ++i) {
            for (std::size_t j = 0; j < num_indices; ++j) {
                if (old_ordering[i] == new_ordering[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        // on _REORDER_MAPS.
        std::vector<std::size_t> map_old_to_new_position(total_dim);
        GenerateBinaryReorderingMap(map_old_to_new_idxpos,
                                    map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to move small chunks.
        std::size_t dim_left = total_dim;
        std::size_t tensor_dim = tensor_size;
        std::size_t dim_right =
            tensor_dim / dim_left; // Remember, it's all powers
        // of 2, so OK.

        PermutePlan plan{dim_left,     dim_right,    tensor_size,
                         dim,          dim,          PermuteType::PermuteRight,
                         new_ordering, old_ordering, map_old_to_new_position};
        precomputed_data.transplans.push_back(plan);
    }

    /**
     * @brief TODO: Figure out what this does
     *
     * @param map_old_to_new_idxpos
     * @param map_old_to_new_position
     */
    void GenerateBinaryReorderingMap(
        const std::vector<std::size_t> &map_old_to_new_idxpos,
        std::vector<std::size_t> &map_old_to_new_position)
    {
        std::size_t dim = 2; // Hard coded!
        std::size_t num_indices = map_old_to_new_idxpos.size();

        // Check
        if (num_indices == 0)
            JET_ABORT("Number of indices cannot be zero.");

        // Check
        if ((std::size_t)std::pow(dim, num_indices) !=
            map_old_to_new_position.size()) {
            JET_ABORT("Size of map must be equal to 2^num_indices");
        }

        // Define super dimensions. See _naive_reorder().
        std::vector<std::size_t> old_dimensions(num_indices, dim);
        std::vector<std::size_t> new_dimensions(num_indices, dim);
        std::vector<std::size_t> old_super_dimensions(num_indices);
        std::vector<std::size_t> new_super_dimensions(num_indices);
        old_super_dimensions[num_indices - 1] = 1;
        new_super_dimensions[num_indices - 1] = 1;

        if (num_indices >= 2)
            for (std::size_t i = num_indices; --i;) {
                old_super_dimensions[i - 1] = old_super_dimensions[i] * dim;
                new_super_dimensions[i - 1] = new_super_dimensions[i] * dim;
            }

        // Iterate and generate map.
        std::vector<std::size_t> old_counter(num_indices, 0);

        /// This will not vectorise --- needs to be refactored
        while (true) {
            std::size_t po{0}, pn{0}; // Position of the data, old and new.

            for (std::size_t i = 0; i < num_indices; ++i) {
                po += old_super_dimensions[i] * old_counter[i];
                pn += new_super_dimensions[map_old_to_new_idxpos[i]] *
                      old_counter[i];
            }
            map_old_to_new_position[po] = pn;

            bool complete{true};
            for (std::size_t j = num_indices; j--;) {
                if (++old_counter[j] < old_dimensions[j]) {
                    complete = false;
                    break;
                }
                else
                    old_counter[j] = 0;
            }
            if (complete)
                break;
        }
    }

    // Can data_ and scratch_copy be declared as restrict?
    template <typename T>
    void PrecomputedLeftOrRightTranspose(const PermutePlan &p_plan,
                                         size_t tensor_dim, T &data_,
                                         T &scratch_copy)
    {
        if (p_plan.type == PermuteType::PermuteRight) {
            std::vector<T> temp_data(p_plan.dim_right);
            for (std::size_t pl = 0; pl < p_plan.dim_left; ++pl) {
                std::size_t offset = pl * p_plan.dim_right;
                for (std::size_t pr = 0; pr < p_plan.dim_right; ++pr)
                    temp_data[pr] = data_[offset + pr];
                for (std::size_t pr = 0; pr < p_plan.dim_right; ++pr)
                    data_[offset + p_plan.map_old_to_new_position[pr]] =
                        temp_data[pr];
            }
        }
        else {
            std::copy(data_, data_ + tensor_dim, scratch_copy);
            // Move back.
            for (std::size_t pl = 0; pl < p_plan.dim_left; ++pl) {
                std::size_t old_offset = pl * p_plan.dim_right;
                std::size_t new_offset =
                    p_plan.map_old_to_new_position[pl] * p_plan.dim_right;
                std::copy(scratch_copy + old_offset,
                          scratch_copy + old_offset + p_plan.dim_right,
                          data_ + new_offset);
            }
        }
    }

    /*******************************************************************************/

    // has to have dimensions that are multiples of 2
    
    template <typename T>
    PrecomputedQflexTransposeData
    PrecomputeFastTransposeData(Tensor<T> &tensor,
                                const std::vector<std::string> &new_ordering)
    {
        using namespace Jet::Utilities;
        PrecomputedQflexTransposeData precomputed_data;

        precomputed_data.log_2 = false;
        for (std::size_t i = 0; i < tensor.GetShape().size(); ++i) {
            if (!is_pow_2(tensor.GetShape()[i]) ) {
                precomputed_data.log_2 = false;
                break;
            }
        }

        // Create binary orderings.
        std::vector<std::string> old_ordering(tensor.GetIndices());
        std::vector<std::size_t> old_dimensions(tensor.GetShape());
        std::size_t num_indices = old_ordering.size();
        std::size_t total_dim = 1;
        for (std::size_t i = 0; i < num_indices; ++i)
            total_dim *= old_dimensions[i];
        // Create map_old_to_new_idxpos from old to new indices, and
        // new_dimensions.
        std::vector<std::size_t> map_old_to_new_idxpos(num_indices);
        std::vector<std::size_t> new_dimensions(num_indices);
        for (std::size_t i = 0; i < num_indices; ++i) {
            for (std::size_t j = 0; j < num_indices; ++j) {
                if (old_ordering[i] == new_ordering[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        precomputed_data.new_ordering = new_ordering;
        precomputed_data.new_dimensions = new_dimensions;
        precomputed_data.old_ordering = old_ordering;
        precomputed_data.old_dimensions = old_dimensions;
        precomputed_data.total_dim = total_dim;

        if (precomputed_data.log_2 == false) {
            return precomputed_data;
        }

        // Create binary orderings:
        std::vector<std::size_t> old_logs(num_indices);
        for (std::size_t i = 0; i < num_indices; ++i) {
            old_logs[i] = fast_log2(old_dimensions[i]);
        }
        std::size_t num_binary_indices = fast_log2(total_dim);
        // Create map from old letter to new group of letters.
        std::unordered_map<std::string, std::vector<std::string>> binary_groups;
        std::size_t alphabet_position = 0;
        for (std::size_t i = 0; i < num_indices; ++i) {
            std::vector<std::string> group(old_logs[i]);
            for (std::size_t j = 0; j < old_logs[i]; ++j) {
                group[j] = ALPHABET_[alphabet_position];
                ++alphabet_position;
            }
            binary_groups[old_ordering[i]] = group;
        }
        // Create old and new binary ordering in letters.
        std::vector<std::string> old_binary_ordering(num_binary_indices);
        std::vector<std::string> new_binary_ordering(num_binary_indices);
        std::size_t binary_position = 0;
        for (std::size_t i = 0; i < num_indices; ++i) {
            std::string old_index = old_ordering[i];
            for (std::size_t j = 0; j < binary_groups[old_index].size(); ++j) {
                old_binary_ordering[binary_position] =
                    binary_groups[old_index][j];
                ++binary_position;
            }
        }
        binary_position = 0;
        for (std::size_t i = 0; i < num_indices; ++i) {
            std::string new_index = new_ordering[i];
            for (std::size_t j = 0; j < binary_groups[new_index].size(); ++j) {
                new_binary_ordering[binary_position] =
                    binary_groups[new_index][j];
                ++binary_position;
            }
        }

        size_t tensor_size = tensor.GetSize();

        if (new_ordering == old_ordering) {
            precomputed_data.no_transpose = true;
            return precomputed_data;
        }
        // Up to here, I have created old_binary_ordering and
        // new_binary_ordering.

        // Change _indices and _dimensions, as well as _index_to_dimension.
        // This is common to all cases, special or default (worst case).

        // Now special cases, before the default L-R-L worst case.
        // Tensor doesn't have enough size to pass MAX_RIGHT_DIM => only one R.
        if (num_binary_indices <= fast_log2(MAX_RIGHT_DIM)) {
            // std::cout << "FIRST FUNCTION TRY " << std::endl;
            PrecomputeRightTransposeData(
                old_binary_ordering, new_binary_ordering, num_binary_indices,
                tensor_size, precomputed_data);
            return precomputed_data;
        }
        // Reordering needs only one right move or one left move.
        // Left moves might benefit a lot from being applied on shorter strings,
        // up to L10. Computation times are L4>L5>L6>...>L10. I'll consider
        // all of these cases.
        {
            if (new_binary_ordering.size() < fast_log2(MAX_RIGHT_DIM))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");

            std::size_t Lr = fast_log2(MAX_RIGHT_DIM);
            std::size_t Ll = new_binary_ordering.size() - Lr;
            std::size_t Rr = fast_log2(MIN_RIGHT_DIM);
            std::vector<std::string> Ll_old_indices(
                old_binary_ordering.begin(), old_binary_ordering.begin() + Ll);
            std::vector<std::string> Ll_new_indices(
                new_binary_ordering.begin(), new_binary_ordering.begin() + Ll);
            // Only one R10.
            if (Ll_old_indices == Ll_new_indices) {
                std::vector<std::string> Lr_old_indices(
                    old_binary_ordering.begin() + Ll,
                    old_binary_ordering.end());
                std::vector<std::string> Lr_new_indices(
                    new_binary_ordering.begin() + Ll,
                    new_binary_ordering.end());
                // std::cout << "SECOND FUNCTION TRY " << std::endl;
                PrecomputeRightTransposeData(Lr_old_indices, Lr_new_indices, Lr,
                                             tensor_size, precomputed_data);
                return precomputed_data;
            }

            if (Rr == 0)
                JET_ABORT("Rr move cannot be zero.");

            // TODO: This loop has been tested to make sure that extended_Rr is
            // consistent with its previous implementation. However, no
            // simulations so far seem to use this loop, so I cannot check it!
            //
            // Only one L\nu move.
            // for (long int i = 5; i >= -1; --i) {
            //  long int extended_Rr = Rr + i;
            for (std::size_t i = 7; i--;) {
                std::size_t extended_Rr = Rr + i - 1;
                std::vector<std::string> Rr_old_indices(
                    old_binary_ordering.end() - extended_Rr,
                    old_binary_ordering.end());
                std::vector<std::string> Rr_new_indices(
                    new_binary_ordering.end() - extended_Rr,
                    new_binary_ordering.end());
                if (Rr_old_indices == Rr_new_indices) {
                    std::vector<std::string> Rl_old_indices(
                        old_binary_ordering.begin(),
                        old_binary_ordering.end() - extended_Rr);
                    std::vector<std::string> Rl_new_indices(
                        new_binary_ordering.begin(),
                        new_binary_ordering.end() - extended_Rr);
                    PrecomputeLeftTransposeData(Rl_old_indices, Rl_new_indices,
                                                extended_Rr, tensor_size,
                                                precomputed_data);
                    return precomputed_data;
                }
            }
        }

        // Worst case.
        {
            // There are two boundaries, L and R.
            // The worst case is the following. It can be optimized, in order to
            // do work early and maybe save the later steps. Think about that,
            // but first let's have something that already works: 1) L5 All
            // indices that are to the left of R and need to end up to its
            //    right are placed in the bucket.
            // 2) R10 All indices to the right of R are placed in their final
            // ordering. 3) L5 All indices to the left of L are placed in their
            // final ordering. Then hardcode special cases. Add conditional to
            // _left_reorder and _right_reorder, so that they don't do anything
            // when not needed. Debug from here!

            if (new_binary_ordering.size() < fast_log2(MAX_RIGHT_DIM))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");
            if (new_binary_ordering.size() < fast_log2(MIN_RIGHT_DIM))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");

            std::size_t Lr = fast_log2(MAX_RIGHT_DIM);
            std::size_t Ll = new_binary_ordering.size() - Lr;
            std::size_t Rr = fast_log2(MIN_RIGHT_DIM);
            std::size_t Rl = new_binary_ordering.size() - Rr;
            // Helper vectors that can be reused.
            std::vector<std::string> Lr_indices(Lr), Ll_indices(Ll),
                Rr_indices(Rr), Rl_indices(Rl);
            for (std::size_t i = 0; i < Rr; ++i)
                Rr_indices[i] = new_binary_ordering[i + Rl];
            for (std::size_t i = 0; i < Rl; ++i)
                Rl_indices[i] = old_binary_ordering[i];
            std::vector<std::string> Rr_new_in_Rl_old =
                VectorIntersection(Rl_indices, Rr_indices);

            std::vector<std::string> Rl_old_not_in_Rr_new =
                VectorSubtraction(Rl_indices, Rr_new_in_Rl_old);

            std::vector<std::string> Rl_first_step =
                VectorConcatenation(Rl_old_not_in_Rr_new, Rr_new_in_Rl_old);

            std::vector<std::string> Rl_zeroth_step(Rl);
            for (std::size_t i = 0; i < Rl; ++i)
                Rl_zeroth_step[i] = old_binary_ordering[i];

            PrecomputeLeftTransposeData(Rl_zeroth_step, Rl_first_step, Rr,
                                        tensor_size, precomputed_data);

            std::vector<std::string> Lr_first_step = VectorConcatenation(
                std::vector<std::string>(Rl_first_step.begin() + Ll,
                                         Rl_first_step.end()),
                std::vector<std::string>(old_binary_ordering.begin() + Rl,
                                         old_binary_ordering.end()));
            Rr_indices = std::vector<std::string>(
                new_binary_ordering.begin() + Rl, new_binary_ordering.end());
            std::vector<std::string> Lr_second_step = VectorConcatenation(
                VectorSubtraction(Lr_first_step, Rr_indices),
                std::vector<std::string>(Rr_indices));

            PrecomputeRightTransposeData(Lr_first_step, Lr_second_step, Lr,
                                         tensor_size, precomputed_data);
            std::vector<std::string> Rl_second_step = VectorConcatenation(
                std::vector<std::string>(Rl_first_step.begin(),
                                         Rl_first_step.begin() + Ll),
                std::vector<std::string>(Lr_second_step.begin(),
                                         Lr_second_step.begin() + Lr - Rr));
            std::vector<std::string> Rl_third_step(
                new_binary_ordering.begin(), new_binary_ordering.begin() + Rl);
            PrecomputeLeftTransposeData(Rl_second_step, Rl_third_step, Rr,
                                        tensor_size, precomputed_data);
            // done with 3).
            return precomputed_data;
        }
    }

    /*******************************************************************************/

    template <class DataType>
    void FastTransposeWithPrecomputedDataOpenMPNoInit(
        DataType &data_, const PrecomputedQflexTransposeData &precomputed_data,
        DataType &scratch)
    {
        for (int plan_i = 0; plan_i < precomputed_data.transplans.size();
             plan_i++) {
            auto &dim_right = precomputed_data.transplans[plan_i].dim_right;
            auto &dim_left = precomputed_data.transplans[plan_i].dim_left;
            auto &tensor_dim = precomputed_data.transplans[plan_i].tensor_dim;
            auto &map_old_to_new_position =
                precomputed_data.transplans[plan_i].map_old_to_new_position;

            if (precomputed_data.transplans[plan_i].type ==
                PermuteType::PermuteRight) {
#pragma omp parallel
                {
                    std::vector<DataType> temp_data(dim_right);
#pragma omp for schedule(static)
                    for (std::size_t pl = 0; pl < dim_left; ++pl) {
                        std::size_t offset = pl * dim_right;
                        for (std::size_t pr = 0; pr < dim_right; ++pr)
                            temp_data[pr] = data_[offset + pr];
                        for (std::size_t pr = 0; pr < dim_right; ++pr)
                            data_[offset + map_old_to_new_position[pr]] =
                                temp_data[pr];
                    }
                }
            }
            else {
#pragma omp parallel for schedule(static, MAX_RIGHT_DIM)
                for (std::size_t p = 0; p < tensor_dim; ++p) {
                    scratch[p] = data_[p];
                }

#pragma omp parallel
                {
#pragma omp for schedule(static)
                    for (std::size_t pl = 0; pl < dim_left; ++pl) {
                        std::size_t old_offset = pl * dim_right;
                        std::size_t new_offset =
                            map_old_to_new_position[pl] * dim_right;
                        for (std::size_t pr = 0; pr < dim_right; ++pr) {
                            data_[new_offset + pr] = scratch[old_offset + pr];
                        }
                    }
                }
            }
        }
    }
};

} // namespace Jet