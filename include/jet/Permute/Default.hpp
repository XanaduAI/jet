#pragma once

#include "Base.hpp"

namespace Jet {

class DefaultPermute : public PermuteBase<DefaultPermute> {

  public:
    DefaultPermute() : PermuteBase<DefaultPermute>() {}

    template <class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data_,
                                    const std::vector<size_t> &shape,
                                    const std::vector<std::string> &old_indices,
                                    const std::vector<std::string> &new_indices)

    {
        using namespace Jet::Utilities;

        if (new_indices == old_indices)
            return data_;

        const size_t num_indices = old_indices.size();
        const size_t total_dim = data_.size();

        if (num_indices == 0)
            JET_ABORT("Number of indices cannot be zero.");

        // Create map_old_to_new_idxpos from old to new indices, and
        // new_dimensions.
        std::vector<size_t> map_old_to_new_idxpos(num_indices);
        std::vector<size_t> new_dimensions(num_indices);
        for (size_t i = 0; i < num_indices; ++i) {
            for (size_t j = 0; j < num_indices; ++j) {
                if (old_indices[i] == new_indices[j]) {
                    map_old_to_new_idxpos[i] = j;
                    new_dimensions[j] = shape[i];
                    break;
                }
            }
        }

        // Create super dimensions (combined dimension of all to the right of
        // i).
        std::vector<size_t> old_super_dimensions(num_indices, 1);
        std::vector<size_t> new_super_dimensions(num_indices, 1);

        const size_t old_dimensions_size = shape.size();
        if (old_dimensions_size >= 2) {
            for (size_t i = old_dimensions_size; --i;) {
                old_super_dimensions[i - 1] =
                    old_super_dimensions[i] * shape[i];
                new_super_dimensions[i - 1] =
                    new_super_dimensions[i] * new_dimensions[i];
            }
        }

        std::vector<unsigned short int> small_map_old_to_new_position(
            MAX_RIGHT_DIM);

        std::vector<DataType> At(data_);
        // No combined efficient mapping from old to new positions with actual
        // copies in memory, all in small cache friendly (for old data, not new,
        // which could be very scattered) blocks.

        // Position old and new.
        size_t po = 0, pn;
        // Counter of the values of each indices in the iteration (old
        // ordering).
        std::vector<size_t> old_counter(num_indices, 0);
        // offset is important when doing this in blocks, as it's indeed
        // implemented.
        size_t offset = 0;
        // internal_po keeps track of interations within a block.
        // Blocks have size MAX_RIGHT_DIM.
        size_t internal_po = 0;

        DataType *__restrict data = At.data();
        const DataType *__restrict scratch =
            data_.data(); // internal pointer offers better performance than
                          // pointer from argument

        size_t effective_max;

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
                    if(++old_counter[j] < shape[j]){
                        complete=false;
                        break;
                    }
                    else{
                        old_counter[j] = 0;
                    }
                }
                // If end of block or end of entire operation, break.
                if ((++internal_po == MAX_RIGHT_DIM) || (po == total_dim - 1))
                    break;
                // If last index (0) was increased, then go back to fastest
                // index.
                if (complete)
                    break;
            }

            // Copy data for this block, taking into account offset of
            // small_map... The following line is to avoid casting MAX_RIGHT_DIM
            // to size_t every iteration. Note that it has to be size_t for min
            // to work, since total_dim is size_t.
            effective_max = std::min((size_t)MAX_RIGHT_DIM, total_dim);
            for (size_t p = 0; p < effective_max; ++p)
                data[small_map_old_to_new_position[p]] = scratch[offset + p];

            offset += MAX_RIGHT_DIM;
        }
        return At;
    }
};

} // namespace Jet
