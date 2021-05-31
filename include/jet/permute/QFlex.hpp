#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "Permuter.hpp"

namespace Jet {

/**
 * @brief Power-of-2 permutation backend. Based on QFlex implementation.
 * 
 * @tparam BLOCKSIZE Controls the blocksize of the transpose to improve cache hits.
 * @tparam MIN_DIMS Controls the right-movement minimum dimension size.
 */
template <size_t BLOCKSIZE = 1024, size_t MIN_DIMS = 32> class QFlexPermuter {
  public:
    template <class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data_,
                                    const std::vector<size_t> &shape,
                                    const std::vector<std::string> &old_indices,
                                    const std::vector<std::string> &new_indices)

    {
        std::vector<DataType> data(data_);
        std::vector<DataType> scratch(data_);
        PrecomputedQflexTransposeData precomputed_data =
            PrecomputeFastTransposeData(data, shape, old_indices, new_indices);
        FastTranspose(data, precomputed_data, scratch);
        return data;
    }

    template <class DataType>
    void Transpose(const std::vector<DataType> &data_,
                   const std::vector<size_t> &shape,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &old_indices,
                   const std::vector<std::string> &new_indices)

    {
        data_out = data_;
        std::vector<DataType> scratch(data_);
        PrecomputedQflexTransposeData precomputed_data =
            PrecomputeFastTransposeData(data_out, shape, old_indices,
                                        new_indices);
        FastTranspose(data_out, precomputed_data, scratch);
    }

  private:
    static constexpr size_t blocksize_ = BLOCKSIZE;
    static constexpr size_t min_dims_ = MIN_DIMS;

    enum class PermuteType { PermuteLeft, PermuteRight, None };

    struct PrecomputedQflexTransposeData {

        std::vector<std::vector<size_t>> map_old_to_new_position;
        std::vector<size_t> dim_left;
        std::vector<size_t> dim_right;
        std::vector<size_t> tensor_dim;
        std::vector<PermuteType> types;

        std::vector<std::string> new_ordering;
        std::vector<size_t> new_dimensions;
        std::vector<std::string> old_ordering;
        std::vector<size_t> old_dimensions;
        bool no_transpose;
        size_t total_dim;
    };

    void GenerateBinaryReorderingMap(
        const std::vector<size_t> &map_old_to_new_idxpos,
        std::vector<size_t> &map_old_to_new_position)
    {
        size_t dim = 2; // Hard coded!
        size_t num_indices = map_old_to_new_idxpos.size();

        // Check
        if (num_indices == 0)
            JET_ABORT("Number of indices cannot be zero.");

        // Check
        if ((size_t)std::pow(dim, num_indices) !=
            map_old_to_new_position.size()) {
            JET_ABORT("Size of map must be equal to 2^num_indices");
        }

        // Define super dimensions. See _naive_reorder().
        std::vector<size_t> old_dimensions(num_indices, dim);
        std::vector<size_t> new_dimensions(num_indices, dim);
        std::vector<size_t> old_super_dimensions(num_indices);
        std::vector<size_t> new_super_dimensions(num_indices);
        old_super_dimensions[num_indices - 1] = 1;
        new_super_dimensions[num_indices - 1] = 1;

        if (num_indices >= 2)
            for (size_t i = num_indices; --i;) {
                old_super_dimensions[i - 1] = old_super_dimensions[i] * dim;
                new_super_dimensions[i - 1] = new_super_dimensions[i] * dim;
            }

        // Iterate and generate map.
        std::vector<size_t> old_counter(num_indices, 0);

        while (true) {
            size_t po{0}, pn{0}; // Position of the data, old and new.

            for (size_t i = 0; i < num_indices; ++i) {
                po += old_super_dimensions[i] * old_counter[i];
                pn += new_super_dimensions[map_old_to_new_idxpos[i]] *
                      old_counter[i];
            }
            map_old_to_new_position[po] = pn;

            bool complete{true};
            for (size_t j = num_indices; j--;) {
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

    template <typename DataType>
    void PrecomputedLeftOrRightTranspose(
        const std::vector<size_t> &map_old_to_new_position,
        size_t dim_left, size_t dim_right, size_t tensor_dim, PermuteType type,
        std::vector<DataType> &data_in, std::vector<DataType> &scratch)
    {
        auto data_ = data_in.data();
        auto scratch_copy = scratch.data();

        if (type == PermuteType::PermuteRight) {
            DataType *temp_data = new DataType[dim_right];
            for (size_t pl = 0; pl < dim_left; ++pl) {
                size_t offset = pl * dim_right;
                for (size_t pr = 0; pr < dim_right; ++pr)
                    *(temp_data + pr) = *(data_ + offset + pr);
                for (size_t pr = 0; pr < dim_right; ++pr)
                    *(data_ + offset + map_old_to_new_position[pr]) =
                        *(temp_data + pr);
            }
            delete[] temp_data;
            temp_data = nullptr;
        }
        else {
            std::copy(data_, data_ + tensor_dim, scratch_copy);
            // Move back.
            for (size_t pl = 0; pl < dim_left; ++pl) {
                size_t old_offset = pl * dim_right;
                size_t new_offset =
                    map_old_to_new_position[pl] * dim_right;
                std::copy(scratch_copy + old_offset,
                          scratch_copy + old_offset + dim_right,
                          data_ + new_offset);
            }
        }
    }

    template <typename DataType>
    void
    FastTranspose(std::vector<DataType> &data_in,
                  const PrecomputedQflexTransposeData &precomputed_data,
                  std::vector<DataType> &scratch_in)
    {
        auto data_ = data_in.data();
        auto scratch = scratch_in.data();

        for (size_t p_i = 0; p_i < precomputed_data.types.size();
             p_i++) { // Type is empty, hence no transpose
            auto &dim_right = precomputed_data.dim_right[p_i];
            auto &dim_left = precomputed_data.dim_left[p_i];
            auto &tensor_dim = precomputed_data.tensor_dim[p_i];
            auto &map_old_to_new_position =
                precomputed_data.map_old_to_new_position[p_i];

            if (precomputed_data.types[p_i] == PermuteType::PermuteRight) {
#if defined _OPENMP
#pragma omp parallel
#endif
                {
                    DataType *temp_data = new DataType[dim_right];
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
                    for (size_t pl = 0; pl < dim_left; ++pl) {
                        size_t offset = pl * dim_right;
                        for (size_t pr = 0; pr < dim_right; ++pr)
                            *(temp_data + pr) = *(data_ + offset + pr);
                        for (size_t pr = 0; pr < dim_right; ++pr)
                            *(data_ + offset + map_old_to_new_position[pr]) =
                                *(temp_data + pr);
                    }
                    delete[] temp_data;
                }
            }
            else {
#if defined _OPENMP
#pragma omp parallel for schedule(static, blocksize_)
#endif
                for (size_t p = 0; p < tensor_dim; ++p) {
                    *(scratch + p) = *(data_ + p);
                }
#if defined _OPENMP
#pragma omp parallel
#endif
                {
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
                    for (size_t pl = 0; pl < dim_left; ++pl) {
                        size_t old_offset = pl * dim_right;
                        size_t new_offset =
                            map_old_to_new_position[pl] * dim_right;
                        for (size_t pr = 0; pr < dim_right; ++pr) {
                            *(data_ + new_offset + pr) =
                                *(scratch + old_offset + pr);
                        }
                    }
                }
            }
        }
    }

    void PrecomputeRightTransposeData(
        const std::vector<std::string> &old_ordering,
        const std::vector<std::string> &new_ordering, size_t tensor_size,
        PrecomputedQflexTransposeData &precomputed_data)
    {

        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new
        // indices, old_dimensions, new_dimensions, and total_dim.
        size_t dim = 2;
        size_t num_indices = old_ordering.size();
        std::vector<size_t> map_old_to_new_idxpos(num_indices);
        std::vector<size_t> old_dimensions(num_indices, dim);
        std::vector<size_t> new_dimensions(num_indices, dim);

        size_t total_dim = 1;
        for (size_t i = 0; i < num_indices; ++i) {
            total_dim *= old_dimensions[i];
        }

        for (size_t i = 0; i < num_indices; ++i) {
            for (size_t j = 0; j < num_indices; ++j) {
                if (old_ordering[i] == new_ordering[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        std::vector<size_t> map_old_to_new_position(total_dim);
        GenerateBinaryReorderingMap(map_old_to_new_idxpos,
                                    map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to reorder within
        // small chuncks.
        size_t dim_right = total_dim;
        size_t dim_left =
            tensor_size / dim_right; // Remember, it's all powers of 2, so OK.

        precomputed_data.dim_left.push_back(dim_left);
        precomputed_data.dim_right.push_back(dim_right);
        precomputed_data.tensor_dim.push_back(tensor_size);
        precomputed_data.map_old_to_new_position.push_back(
            map_old_to_new_position);
        precomputed_data.types.push_back(PermuteType::PermuteRight);
    }

    void
    PrecomputeLeftTransposeData(const std::vector<std::string> &old_ordering,
                                const std::vector<std::string> &new_ordering,
                                size_t tensor_size,
                                PrecomputedQflexTransposeData &precomputed_data)
    {

        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new
        // indices, old_dimensions, new_dimensions, and total_dim.
        size_t dim = 2;
        size_t num_indices = old_ordering.size();
        std::vector<size_t> map_old_to_new_idxpos(num_indices);
        std::vector<size_t> old_dimensions(num_indices, dim);
        std::vector<size_t> new_dimensions(num_indices, dim);
        size_t total_dim = 1;
        for (size_t i = 0; i < num_indices; ++i)
            total_dim *= old_dimensions[i];
        for (size_t i = 0; i < num_indices; ++i) {
            for (size_t j = 0; j < num_indices; ++j) {
                if (old_ordering[i] == new_ordering[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        // on _REORDER_MAPS.
        std::vector<size_t> map_old_to_new_position(total_dim);
        GenerateBinaryReorderingMap(map_old_to_new_idxpos,
                                    map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to move small chunks.
        size_t dim_left = total_dim;
        size_t tensor_dim = tensor_size;
        size_t dim_right =
            tensor_dim / dim_left; // Remember, it's all powers
        // of 2, so OK.

        precomputed_data.dim_left.push_back(dim_left);
        precomputed_data.dim_right.push_back(dim_right);
        precomputed_data.tensor_dim.push_back(tensor_dim);
        precomputed_data.map_old_to_new_position.push_back(
            map_old_to_new_position);
        precomputed_data.types.push_back(PermuteType::PermuteLeft);
    }

    // has to have dimensions that are multiples of 2
    template <typename DataType>
    PrecomputedQflexTransposeData
    PrecomputeFastTransposeData(std::vector<DataType> &tensor_data,
                                const std::vector<size_t> &shape,
                                const std::vector<std::string> &old_indices,
                                const std::vector<std::string> &new_ordering)
    {
        using namespace Jet::Utilities;
        PrecomputedQflexTransposeData precomputed_data;

        for (size_t i = 0; i < shape.size(); ++i) {
            JET_ABORT_IF_NOT(is_pow_2(shape[i]),
                             "Fast transpose expects power-of-2 data.");
        }

        // Create binary orderings.
        std::vector<std::string> old_ordering(old_indices);
        std::vector<size_t> old_dimensions(shape);
        size_t num_indices = old_ordering.size();
        size_t total_dim = 1;
        for (size_t i = 0; i < num_indices; ++i)
            total_dim *= old_dimensions[i];
        // Create map_old_to_new_idxpos from old to new indices, and
        // new_dimensions.
        std::vector<size_t> map_old_to_new_idxpos(num_indices);
        std::vector<size_t> new_dimensions(num_indices);
        for (size_t i = 0; i < num_indices; ++i) {
            for (size_t j = 0; j < num_indices; ++j) {
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

        // Create binary orderings:
        std::vector<size_t> old_logs(num_indices);
        for (size_t i = 0; i < num_indices; ++i) {
            old_logs[i] = fast_log2(old_dimensions[i]);
        }
        size_t num_binary_indices = fast_log2(total_dim);
        // Create map from old letter to new group of letters.
        std::unordered_map<std::string, std::vector<std::string>> binary_groups;
        size_t alphabet_position = 0;
        for (size_t i = 0; i < num_indices; ++i) {
            std::vector<std::string> group(old_logs[i]);
            for (size_t j = 0; j < old_logs[i]; ++j) {
                group[j] = GenerateStringIndex(alphabet_position);
                ++alphabet_position;
            }
            binary_groups[old_ordering[i]] = group;
        }
        // Create old and new binary ordering in letters.
        std::vector<std::string> old_binary_ordering(num_binary_indices);
        std::vector<std::string> new_binary_ordering(num_binary_indices);
        size_t binary_position = 0;
        for (size_t i = 0; i < num_indices; ++i) {
            std::string old_index = old_ordering[i];
            for (size_t j = 0; j < binary_groups[old_index].size(); ++j) {
                old_binary_ordering[binary_position] =
                    binary_groups[old_index][j];
                ++binary_position;
            }
        }
        binary_position = 0;
        for (size_t i = 0; i < num_indices; ++i) {
            std::string new_index = new_ordering[i];
            for (size_t j = 0; j < binary_groups[new_index].size(); ++j) {
                new_binary_ordering[binary_position] =
                    binary_groups[new_index][j];
                ++binary_position;
            }
        }

        size_t tensor_size = tensor_data.size();

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
        if (num_binary_indices <= fast_log2(blocksize_)) {
            PrecomputeRightTransposeData(old_binary_ordering,
                                         new_binary_ordering, tensor_size,
                                         precomputed_data);
            return precomputed_data;
        }
        // Reordering needs only one right move or one left move.
        // Left moves might benefit a lot from being applied on shorter strings,
        // up to L10. Computation times are L4>L5>L6>...>L10. I'll consider
        // all of these cases.
        {
            if (new_binary_ordering.size() < fast_log2(blocksize_))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");

            constexpr size_t Lr = fast_log2(blocksize_);
            size_t Ll = new_binary_ordering.size() - Lr;
            constexpr size_t Rr = fast_log2(min_dims_);
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
                PrecomputeRightTransposeData(Lr_old_indices, Lr_new_indices,
                                             tensor_size, precomputed_data);
                return precomputed_data;
            }

            if (Rr == 0)
                JET_ABORT("Rr move cannot be zero.");

            for (size_t i = 7; i--;) {
                size_t extended_Rr = Rr + i - 1;
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
                                                tensor_size, precomputed_data);
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

            if (new_binary_ordering.size() < fast_log2(blocksize_))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");
            if (new_binary_ordering.size() < fast_log2(min_dims_))
                JET_ABORT(
                    "New ordering is too small to be used at this point.");

            constexpr size_t Lr = fast_log2(blocksize_);
            size_t Ll = new_binary_ordering.size() - Lr;
            constexpr size_t Rr = fast_log2(min_dims_);
            size_t Rl = new_binary_ordering.size() - Rr;
            // Helper vectors that can be reused.
            std::vector<std::string> Lr_indices(Lr), Ll_indices(Ll), Rr_indices(Rr), Rl_indices(Rl);
            for (size_t i = 0; i < Rr; ++i)
                Rr_indices[i] = new_binary_ordering[i + Rl];
            for (size_t i = 0; i < Rl; ++i)
                Rl_indices[i] = old_binary_ordering[i];
            std::vector<std::string> Rr_new_in_Rl_old = VectorIntersection(Rl_indices, Rr_indices);

            std::vector<std::string> Rl_old_not_in_Rr_new = VectorSubtraction(Rl_indices, Rr_new_in_Rl_old);

            std::vector<std::string> Rl_first_step =
                VectorConcatenation(Rl_old_not_in_Rr_new, Rr_new_in_Rl_old);

            std::vector<std::string> Rl_zeroth_step(Rl);
            for (size_t i = 0; i < Rl; ++i)
                Rl_zeroth_step[i] = old_binary_ordering[i];

            PrecomputeLeftTransposeData(Rl_zeroth_step, Rl_first_step, tensor_size, precomputed_data);

            std::vector<std::string> Lr_first_step = VectorConcatenation(
                std::vector<std::string>(Rl_first_step.begin() + Ll, Rl_first_step.end()),
                std::vector<std::string>(old_binary_ordering.begin() + Rl, old_binary_ordering.end()));

            Rr_indices = std::vector<std::string>( new_binary_ordering.begin() + Rl, new_binary_ordering.end());

            std::vector<std::string> Lr_second_step = VectorConcatenation(
                VectorSubtraction(Lr_first_step, Rr_indices),
                std::vector<std::string>(Rr_indices));

            PrecomputeRightTransposeData(Lr_first_step, Lr_second_step, tensor_size, precomputed_data);

            std::vector<std::string> Rl_second_step = VectorConcatenation(
                std::vector<std::string>(Rl_first_step.begin(), Rl_first_step.begin() + Ll),
                std::vector<std::string>(Lr_second_step.begin(), Lr_second_step.begin() + Lr - Rr));
            
            std::vector<std::string> Rl_third_step( new_binary_ordering.begin(), new_binary_ordering.begin() + Rl);
            PrecomputeLeftTransposeData(Rl_second_step, Rl_third_step, tensor_size, precomputed_data);
            // done with 3).
            return precomputed_data;
        }
    }
};

} // namespace Jet
