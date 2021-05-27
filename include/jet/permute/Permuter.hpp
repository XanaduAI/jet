#pragma once

#include <complex>
#include <string>
#include <vector>

#include "../Abort.hpp"
#include "../Utilities.hpp"

namespace Jet {

/**
 * @brief Interface for tensor permutation backend.
 *
 * @tparam PermuteBackend
 */
template <class PermuterBackend> class Permuter {
  public:
    Permuter(size_t blocksize=1024) : blocksize(blocksize) {}

    template <class DataType>
    void Transpose(const std::vector<DataType> &data_in,
                   const std::vector<size_t> &shape,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        permuter_b_.Transpose(data_in, shape, data_out, current_order, new_order, blocksize);
    }

    template <class DataType>
    std::vector<DataType>
    Transpose(const std::vector<DataType> &data_in,
              const std::vector<size_t> &shape,
              const std::vector<std::string> &current_order,
              const std::vector<std::string> &new_order)
    {
        permuter_b_.Transpose(data_in, shape, current_order, new_order, blocksize);
    }

  protected:
    friend PermuterBackend;

  private:
    const size_t blocksize;
    PermuterBackend permuter_b_;
};

} // namespace Jet