#pragma once

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "Permuter.hpp"

namespace Jet {

/**
 * @brief Power-of-2 permutation backend. Based on QFlex implementation.
 *
 * @tparam BLOCKSIZE Controls the blocksize of the transpose to improve cache
 * hits.
 * @tparam MIN_DIMS Controls the right-movement minimum dimension size.
 */
template <size_t BLOCKSIZE = 1024, size_t MIN_DIMS = 32> class QFlexPermuter {
    static_assert(BLOCKSIZE > MIN_DIMS,
                  "BLOCKSIZE must be greater than MIN_DIMS");

  public:
    template <class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data_,
                                    const std::vector<size_t> &shape,
                                    const std::vector<std::string> &old_indices,
                                    const std::vector<std::string> &new_indices)

    {
        std::vector<DataType> data_out(data_);
        Transpose(data_, shape, data_out, old_indices, new_indices);
        return data_out;
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
        PlanData precomputed_data = PrecomputeFastTransposeData(
            data_out, shape, old_indices, new_indices);
        FastTranspose(data_out, precomputed_data, scratch);
    }

  private:
    static constexpr size_t blocksize_ = BLOCKSIZE;
    static constexpr size_t min_dims_ = MIN_DIMS;

    enum class PermuteType { PermuteLeft, PermuteRight, None };

    struct DimData {
        const size_t dim_left;
        const size_t dim_right;
        const size_t tensor_dim;
        const std::vector<size_t> dim_map;

        DimData(size_t dim_left, size_t dim_right, size_t tensor_dim,
                const std::vector<size_t> &dim_map)
            : dim_left{dim_left}, dim_right{dim_right},
              tensor_dim{tensor_dim}, dim_map{dim_map}
        {
        }
        DimData(size_t dim_left, size_t dim_right, size_t tensor_dim,
                std::vector<size_t> &&dim_map)
            : dim_left{dim_left}, dim_right{dim_right},
              tensor_dim{tensor_dim}, dim_map{dim_map}
        {
        }
        DimData() : dim_left{}, dim_right{}, tensor_dim{}, dim_map{} {}
    };

    struct PlanData {
        std::vector<DimData> dim_data;
        std::vector<PermuteType> types;

        std::vector<std::string> new_ordering;
        std::vector<size_t> new_dimensions;
        std::vector<std::string> old_ordering;
        std::vector<size_t> old_dimensions;
        bool no_transpose;
        size_t total_dim;
        PlanData(size_t num_elements)
            : new_ordering(num_elements), new_dimensions(num_elements),
              old_ordering(num_elements), old_dimensions(num_elements)
        {
        }

        PlanData(const std::vector<std::string> &old_indices,
                 const std::vector<std::string> &new_ordering,
                 const std::vector<size_t> &shape)
            : new_ordering(new_ordering), new_dimensions(shape.size(), 0),
              old_ordering(old_indices), old_dimensions(shape)
        {
        }
    };

    void GenerateBinaryReorderingMap(
        const std::vector<size_t> &map_old_to_new_idxpos,
        std::vector<size_t> &map_old_to_new_position)
    {
        size_t dim = 2; // Hard coded!
        size_t num_indices = map_old_to_new_idxpos.size();

        JET_ABORT_IF(num_indices == 0, "Number of indices cannot be zero.");

        JET_ABORT_IF(static_cast<size_t>(std::pow(dim, num_indices)) !=
                         map_old_to_new_position.size(),
                     "Size of map must be equal to 2^num_indices");

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

            for (size_t i = 0; i < num_indices; i++) {
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
    void FastTranspose(std::vector<DataType> &data_in,
                       const PlanData &precomputed_data,
                       std::vector<DataType> &scratch_in)
    {

        for (size_t p_i = 0; p_i < precomputed_data.types.size();
             p_i++) { // Type is empty, hence no transpose
            auto &dim_right = precomputed_data.dim_data[p_i].dim_right;
            auto &dim_left = precomputed_data.dim_data[p_i].dim_left;
            auto &tensor_dim = precomputed_data.dim_data[p_i].tensor_dim;
            auto &map_old_to_new_position =
                precomputed_data.dim_data[p_i].dim_map;

            if (precomputed_data.types[p_i] == PermuteType::PermuteRight) {
#if defined _OPENMP
#pragma omp parallel
#endif
                {
                    std::vector<DataType> temp_data(dim_right);
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
                    for (size_t pl = 0; pl < dim_left; pl++) {
                        size_t offset = pl * dim_right;

                        for (size_t pr = 0; pr < dim_right; pr++)
                            temp_data[pr] = data_in[offset + pr];
                        for (size_t pr = 0; pr < dim_right; pr++) {
                            data_in[offset + map_old_to_new_position[pr]] =
                                temp_data[pr];
                        }
                    }
                }
            }
            else if (precomputed_data.types[p_i] == PermuteType::PermuteLeft) {
#if defined _OPENMP
#pragma omp parallel for schedule(static, blocksize_)
#endif
                for (size_t p = 0; p < tensor_dim; p++) {
                    scratch_in[p] = data_in[p];
                }
#if defined _OPENMP
#pragma omp parallel
#endif
                {
#if defined _OPENMP
#pragma omp for schedule(static)
#endif
                    for (size_t pl = 0; pl < dim_left; pl++) {
                        size_t old_offset = pl * dim_right;
                        size_t new_offset =
                            map_old_to_new_position[pl] * dim_right;
                        for (size_t pr = 0; pr < dim_right; pr++) {
                            data_in[new_offset + pr] =
                                scratch_in[old_offset + pr];
                        }
                    }
                }
            }
        }
    }

    void
    PrecomputeRightTransposeData(const std::vector<std::string> &old_ordering,
                                 const std::vector<std::string> &new_ordering,
                                 size_t tensor_size, PlanData &precomputed_data)
    {
        JET_ABORT_IF(old_ordering.size() != new_ordering.size(),
                     "The number of provided indices must be the same for old "
                     "and new orderings.")

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
        for (size_t i = 0; i < num_indices; i++) {
            total_dim *= old_dimensions[i];
        }

        for (size_t i = 0; i < num_indices; i++) {
            for (size_t j = 0; j < num_indices; j++) {
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
        // small chunks.
        size_t dim_right = total_dim;
        size_t dim_left = tensor_size >> Jet::Utilities::fast_log2(dim_right);

        precomputed_data.dim_data.push_back(
            DimData{dim_left, dim_right, tensor_size, map_old_to_new_position});

        precomputed_data.types.push_back(PermuteType::PermuteRight);
    }

    void
    PrecomputeLeftTransposeData(const std::vector<std::string> &old_ordering,
                                const std::vector<std::string> &new_ordering,
                                size_t tensor_size, PlanData &precomputed_data)
    {
        JET_ABORT_IF(old_ordering.size() != new_ordering.size(),
                     "The number of provided indices must be the same for old "
                     "and new orderings.")

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
        for (size_t i = 0; i < num_indices; i++)
            total_dim *= old_dimensions[i];
        for (size_t i = 0; i < num_indices; i++) {
            for (size_t j = 0; j < num_indices; j++) {
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
        size_t dim_right = tensor_dim >> Jet::Utilities::fast_log2(dim_left);

        precomputed_data.dim_data.push_back(
            DimData{dim_left, dim_right, tensor_dim, map_old_to_new_position});

        precomputed_data.types.push_back(PermuteType::PermuteLeft);
    }

    template <typename DataType>
    PlanData
    PrecomputeFastTransposeData(std::vector<DataType> &tensor_data,
                                const std::vector<size_t> &shape,
                                const std::vector<std::string> &old_indices,
                                const std::vector<std::string> &new_ordering)
    {
        using namespace Jet::Utilities;
        PlanData precomputed_data(old_indices, new_ordering, shape);

        for (size_t i = 0; i < shape.size(); i++) {
            JET_ABORT_IF_NOT(is_pow_2(shape[i]),
                             "Fast transpose expects power-of-2 data.");
        }

        // Create binary orderings.
        std::vector<std::string> &old_ordering = precomputed_data.old_ordering;
        std::vector<size_t> &old_dimensions = precomputed_data.old_dimensions;
        size_t num_indices = old_ordering.size();
        size_t total_dim = 1;
        for (size_t i = 0; i < num_indices; i++)
            total_dim *= old_dimensions[i];

        std::vector<size_t> &new_dimensions = precomputed_data.new_dimensions;
        for (size_t i = 0; i < num_indices; i++) {
            for (size_t j = 0; j < num_indices; j++) {
                if (old_ordering[i] == new_ordering[j]) {
                    new_dimensions[j] = old_dimensions[i];
                    break;
                }
            }
        }

        precomputed_data.total_dim = total_dim;

        if (new_ordering == old_ordering) {
            precomputed_data.no_transpose = true;
            return precomputed_data;
        }

        // Create binary orderings:
        std::vector<size_t> old_logs(num_indices);
        for (size_t i = 0; i < num_indices; i++) {
            old_logs[i] = fast_log2(old_dimensions[i]);
        }
        size_t num_binary_indices = fast_log2(total_dim);
        // Create map from old letter to new group of letters.
        std::unordered_map<std::string, std::vector<std::string>> binary_groups;
        size_t alphabet_position = 0;
        for (size_t i = 0; i < num_indices; i++) {
            std::vector<std::string> group(old_logs[i]);
            for (size_t j = 0; j < old_logs[i]; j++) {
                group[j] = GenerateStringIndex(alphabet_position++);
            }
            binary_groups.emplace(old_ordering[i], group);
        }
        // Create old and new binary ordering in letters.
        std::vector<std::string> old_binary_ordering(num_binary_indices);
        std::vector<std::string> new_binary_ordering(num_binary_indices);
        size_t binary_position = 0;
        for (size_t i = 0; i < num_indices; i++) {
            std::string old_index = old_ordering[i];
            for (std::string j : binary_groups[old_index]) {
                old_binary_ordering[binary_position++] = j;
            }
        }
        binary_position = 0;
        for (size_t i = 0; i < num_indices; i++) {
            std::string new_index = new_ordering[i];
            for (std::string j : binary_groups[new_index]) {
                new_binary_ordering[binary_position++] = j;
            }
        }

        size_t tensor_size = tensor_data.size();

        if (num_binary_indices <= fast_log2(blocksize_)) {
            PrecomputeRightTransposeData(old_binary_ordering,
                                         new_binary_ordering, tensor_size,
                                         precomputed_data);
            return precomputed_data;
        }

        {
            JET_ABORT_IF(new_binary_ordering.size() < fast_log2(blocksize_),
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

            JET_ABORT_IF(
                Rr == 0,
                "Number of minimum dimensions must be greater than one.");

            /// Search for matching substring sequences from end (right) to
            /// start. If the matched sequence is greater in length than the
            /// minimum support size defined by MIN_SIZE, we can form a
            /// Left-transpose.
            auto matched_it = std::mismatch(old_binary_ordering.rbegin(),
                                            old_binary_ordering.rend(),
                                            new_binary_ordering.rbegin());

            size_t num_elems =
                std::distance(old_binary_ordering.rbegin(), matched_it.first);

            if (matched_it.first != old_binary_ordering.rbegin() &&
                num_elems >= (Rr)) {
                std::vector<std::string> Rl_old_indices(
                    old_binary_ordering.begin(),
                    old_binary_ordering.end() - num_elems);
                std::vector<std::string> Rl_new_indices(
                    new_binary_ordering.begin(),
                    new_binary_ordering.end() - num_elems);
                PrecomputeLeftTransposeData(Rl_old_indices, Rl_new_indices,
                                            tensor_size, precomputed_data);
                return precomputed_data;
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

            JET_ABORT_IF(new_binary_ordering.size() < fast_log2(blocksize_),
                         "New ordering is too small to be used at this point.");
            JET_ABORT_IF(new_binary_ordering.size() < fast_log2(min_dims_),
                         "New ordering is too small to be used at this point.");

            constexpr size_t Lr = fast_log2(blocksize_);
            size_t Ll = new_binary_ordering.size() - Lr;
            constexpr size_t Rr = fast_log2(min_dims_);
            size_t Rl = new_binary_ordering.size() - Rr;
            // Helper vectors that can be reused.

            std::vector<std::string> Rr_indices(
                new_binary_ordering.begin() + Rl, new_binary_ordering.end());

            std::vector<std::string> Rl_indices(
                old_binary_ordering.begin(), old_binary_ordering.begin() + Rl);

            std::vector<std::string> Rr_new_in_Rl_old =
                VectorIntersection(Rl_indices, Rr_indices);

            std::vector<std::string> Rl_old_not_in_Rr_new =
                VectorSubtraction(Rl_indices, Rr_new_in_Rl_old);

            std::vector<std::string> Rl_first_step =
                VectorConcatenation(Rl_old_not_in_Rr_new, Rr_new_in_Rl_old);

            std::vector<std::string> &Rl_zeroth_step =
                Rl_indices; /// already created

            PrecomputeLeftTransposeData(Rl_zeroth_step, Rl_first_step,
                                        tensor_size, precomputed_data);

            std::vector<std::string> Lr_first_step(Rl_first_step.begin() + Ll,
                                                   Rl_first_step.end());
            Lr_first_step.insert(Lr_first_step.end(),
                                 old_binary_ordering.begin() + Rl,
                                 old_binary_ordering.end());

            std::vector<std::string> Lr_second_step = VectorConcatenation(
                VectorSubtraction(Lr_first_step, Rr_indices),
                std::vector<std::string>(Rr_indices));

            JET_ABORT_IF(Lr_second_step.size() != Lr_first_step.size(),
                         "New ordering is too small to be used at this point. "
                         "Consider increasing the BLOCKSIZE.");

            PrecomputeRightTransposeData(Lr_first_step, Lr_second_step,
                                         tensor_size, precomputed_data);

            std::vector<std::string> Rl_second_step = VectorConcatenation(
                std::vector<std::string>(Rl_first_step.begin(),
                                         Rl_first_step.begin() + Ll),
                std::vector<std::string>(Lr_second_step.begin(),
                                         Lr_second_step.begin() + Lr - Rr));

            std::vector<std::string> Rl_third_step(
                new_binary_ordering.begin(), new_binary_ordering.begin() + Rl);
            PrecomputeLeftTransposeData(Rl_second_step, Rl_third_step,
                                        tensor_size, precomputed_data);
            // done with 3).

            return precomputed_data;
        }
    }
};

} // namespace Jet
