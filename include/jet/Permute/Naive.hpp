#pragma once

#include "Base.hpp"

namespace Jet {

class NaivePermute : public PermuteBase<NaivePermute> {

    public:

    NaivePermute() : PermuteBase<NaivePermute>() {}

    template <class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data, const std::vector<size_t> &shape, const std::vector<std::string> &old_indices, const std::vector<std::string> &new_indices)
    {
        using namespace Jet::Utilities;

        if (new_indices == old_indices)
            return data;

        std::vector<std::string> old_ordering(old_indices);
        std::vector<size_t> old_dimensions(shape);
        size_t num_indices = old_ordering.size();
        size_t total_dim = data.size();

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

        std::vector<DataType> data_T(data);
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
            size_t effective_max = std::min(static_cast<size_t>(MAX_RIGHT_DIM), total_dim);
            for (size_t p = 0; p < effective_max; ++p)
                data_T[small_map_old_to_new_position[p]] = data[offset + p];

            offset += MAX_RIGHT_DIM;
        }

        return data_T;
    }
};

}