#pragma once

#include<complex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Abort.hpp"
#include "Utilities.hpp"
#include "Tensor.hpp"

namespace Jet {


/**
 * @brief Interface for tensor permutation backend.
 * 
 * @tparam PermuteBackend 
 */
template <class DataType, class PermuteBackend>
class PermuteBase
{
    private:
    static const size_t uid;

    protected:

    PermuteBase() = default;

    // Ensure derived type recognised as friend for CRTP
    friend PermuteBackend;

    public:
    void Transpose(
        const std::vector<DataType>& data_in, 
        std::vector<DataType>& data_out, 
        const std::vector<std::string>& current_order, 
        const std::vector<std::string>& new_order)
    {
        static_cast<PermuteBackend>(*this).Transpose(data_in, data_out, current_order, new_order);
    }
};

class QFPermute : public PermuteBase<std::complex<float>, QFPermute>{
    using DataType = std::complex<float>;

    enum class PermuteType { 
        PermuteLeft, 
        PermuteRight 
    };

    public:

    QFPermute() : PermuteBase<DataType, QFPermute>()
    {

    }

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

        PermutePlan(
            size_t dim_left, 
            size_t dim_right, 
            size_t tensor_dim, 
            size_t old_dimensions, 
            size_t new_dimensions, 
            PermuteType type,
            std::vector<std::string> new_ordering,
            std::vector<std::string> old_ordering,
            std::vector<size_t> map_old_to_new_position
        ) : 
            dim_left(dim_left), 
            dim_right(dim_right), 
            tensor_dim(tensor_dim), 
            old_dimensions(old_dimensions), 
            new_dimensions(new_dimensions),
            type(type),
            new_ordering(new_ordering),
            old_ordering(old_ordering),
            map_old_to_new_position(map_old_to_new_position)
        {};
    };

    using plan_type = PermutePlan;


    /**
     * @brief Tranpose the tensor data to the new index set.
     * 
     * @note If input and output data arrays are the same, an in-place transpose will be used.
     * 
     * @param data_in Data of tensor object to tranpose.
     * @param data_out Output data object of tranposed tensor data.
     * @param current_order Existing index order of tensor.
     * @param new_order New index order of the `data_out` tensor data.
     */
    void Transpose(
        const DataType& data_in, 
        DataType& data_out, 
        const std::vector<size_t>& current_order, 
        const std::vector<size_t>& new_order)
    {
        if(&data_in == &data_out)
        {
            TransposeInPlace_(data_out, current_order, new_order);
        }
    }

    private:

    struct PrecomputedQflexTransposeData {
        std::vector<PermutePlan> transplans;
        size_t total_dim;
        bool no_transpose;
        bool log_2;
    };

    void TransposeInPlace_(DataType& data, const std::vector<size_t>& current_order, const std::vector<size_t>& new_order){
        data*=10;
        for (auto& a : current_order)
            std::cout << a << std::endl;
        for (auto& a : new_order)
            std::cout << a << std::endl;
    }

    /**
     * @brief Right-partition data permutation following https://arxiv.org/pdf/1811.09599.pdf
     * 
     * @param old_ordering 
     * @param new_ordering 
     * @param num_indices_right 
     * @param tensor_size 
     * @param precomputed_data 
     */
    void PrecomputeRightTransposeData_(
        const std::vector<std::string> &old_ordering,
        const std::vector<std::string> &new_ordering, std::size_t num_indices_right,
        size_t tensor_size, PrecomputedQflexTransposeData &precomputed_data)
    {
        std::cout << num_indices_right << std::endl;

        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new indices,
        // old_dimensions, new_dimensions, and total_dim.
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
        GenerateBinaryReorderingMap(map_old_to_new_idxpos, map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to reorder within
        // small chuncks.
        std::size_t dim_right = total_dim;
        std::size_t dim_left =
            tensor_size / dim_right; // Remember, it's all powers of 2, so OK.

        PermutePlan plan {dim_left, dim_right, tensor_size, dim, dim, PermuteType::PermuteRight, new_ordering, old_ordering, map_old_to_new_position};
        precomputed_data.transplans.push_back(plan);

/*
        precomputed_data.dim_left.push_back(dim_left);
        precomputed_data.dim_right.push_back(dim_right);
        precomputed_data.tensor_dim.push_back(tensor_size);
        precomputed_data.map_old_to_new_position.push_back(map_old_to_new_position);
        precomputed_data.types.push_back(PermuteType::PermuteRight);
*/
    }

    void PrecomputeLeftTransposeData_(
        const std::vector<std::string> &old_ordering,
        const std::vector<std::string> &new_ordering, std::size_t num_indices_right,
        size_t tensor_size, PrecomputedQflexTransposeData &precomputed_data)
    {
        std::cout << num_indices_right << std::endl;


        // Don't do anything if there is nothing to reorder.
        if (new_ordering == old_ordering) {
            return;
        }

        // Create dim, num_indices, map_old_to_new_idxpos from old to new indices,
        // old_dimensions, new_dimensions, and total_dim.
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
        GenerateBinaryReorderingMap(map_old_to_new_idxpos, map_old_to_new_position);

        // With the map_old_to_new_position, we are ready to move small chunks.
        std::size_t dim_left = total_dim;
        std::size_t tensor_dim = tensor_size;
        std::size_t dim_right = tensor_dim / dim_left; // Remember, it's all powers
        // of 2, so OK.
/*
        precomputed_data.dim_left.push_back(dim_left);
        precomputed_data.dim_right.push_back(dim_right);
        precomputed_data.tensor_dim.push_back(tensor_dim);
        precomputed_data.map_old_to_new_position.push_back(map_old_to_new_position);
        precomputed_data.types.push_back(PermuteType::PermuteLeft);*/

        PermutePlan plan {dim_left, dim_right, tensor_size, dim, dim, PermuteType::PermuteRight, new_ordering, old_ordering, map_old_to_new_position};
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

        while (true) {
            std::size_t po{0}, pn{0}; // Position of the data, old and new.

            for (std::size_t i = 0; i < num_indices; ++i) {
                po += old_super_dimensions[i] * old_counter[i];
                pn += new_super_dimensions[map_old_to_new_idxpos[i]] * old_counter[i];
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

    //Can data_ and scratch_copy be declared as restrict?
    template <typename T>
    void PrecomputedLeftOrRightTranspose(const PermutePlan &p_plan, 
        size_t tensor_dim, T *data_, T *scratch_copy)
    {
        if (p_plan.type == PermuteType::PermuteRight) {
            std::unique_ptr<T> temp_data(new T[p_plan.dim_right]);
            for (std::size_t pl = 0; pl < p_plan.dim_left; ++pl) {
                std::size_t offset = pl * p_plan.dim_right;
                for (std::size_t pr = 0; pr < p_plan.dim_right; ++pr)
                    temp_data[pr] = data_[offset + pr];
                for (std::size_t pr = 0; pr < p_plan.dim_right; ++pr)
                    data_[offset + p_plan.map_old_to_new_position[pr]] = temp_data[pr];
            }
        }
        else {
            std::copy(data_, data_ + tensor_dim, scratch_copy);
            // Move back.
            for (std::size_t pl = 0; pl < p_plan.dim_left; ++pl) {
                std::size_t old_offset = pl * p_plan.dim_right;
                std::size_t new_offset = p_plan.map_old_to_new_position[pl] * p_plan.dim_right;
                std::copy(scratch_copy + old_offset,
                        scratch_copy + old_offset + p_plan.dim_right,
                        data_ + new_offset);
            }
        }
    }
};


} //namespace Jet