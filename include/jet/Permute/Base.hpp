#pragma once

#include <complex>
#include <string>
#include <vector>

#include "../Abort.hpp"
#include "../Utilities.hpp"

namespace Jet {

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


/**
 * @brief Interface for tensor permutation backend.
 *
 * @tparam PermuteBackend
 */
template <class PermuteBackend> class PermuteBase {
  protected:

    // Ensure derived type recognised as friend for CRTP
    friend PermuteBackend;

  public:
    PermuteBase() = default;

    template<class DataType>
    void Transpose(const std::vector<DataType> &data_in,
                   const std::vector<size_t> &shape,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        static_cast<PermuteBackend &>(*this).Transpose(
            data_in, shape, data_out, current_order, new_order);
    }

    template<class DataType>
    std::vector<DataType> Transpose(const std::vector<DataType> &data_in,
                   const std::vector<size_t> &shape,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        static_cast<PermuteBackend &>(*this).Transpose(
            data_in, shape, current_order, new_order);
    }


};

}