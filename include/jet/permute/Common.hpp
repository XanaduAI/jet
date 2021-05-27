#include <vector>

void BlockIterate(std::vector<std::size_t> old_strides){
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
                if ((++internal_po == blocksize) || (po == total_dim - 1))
                    break;
                // If last index (0) was increased, then go back to fastest
                // index.
                if (complete)
                    break;
            }

            // Copy data for this block, taking into account offset of
            // small_map.
            effective_max = std::min(blocksize, total_dim);
            for (size_t p = 0; p < effective_max; ++p)
                data[small_map_old_to_new_position[p]] = scratch[offset + p];

            offset += blocksize;
        }
}