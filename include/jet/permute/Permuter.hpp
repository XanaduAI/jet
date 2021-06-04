#pragma once

#include <complex>
#include <set>
#include <string>
#include <vector>

#include "../Abort.hpp"
#include "../Utilities.hpp"

namespace Jet {

/**
 * @brief Interface for tensor permutation backend.
 *
 * The Permuter class represents the front-end interface for calling
 * permutations, which are a generalization of transposition to high-rank
 * tensors. The class follows a composition-based approach, where we instantiate
 * with a given backend permuter, who makes available two `Transpose` methods,
 * one which returns the transform result, and another which modifies a
 * reference directly.
 *
 *   Example 1:
 *   const std::vector<size_t> data_in {0,1,2,3,4,5};
 *   std::vector<size_t> data_out(data_in.size(), 0);
 *   Permuter<DefaultPermuter<size_t>> p;
 *   p.Transpose(data_in, {2,3}, data_out, {"a","b"}, {"b","a"});
 *
 *   Example 2:
 *   const std::vector<size_t> data_in {0,1,2,3,4,5};
 *   Permuter<DefaultPermuter<size_t>> p;
 *   auto data_out = p.Transpose(data_in, {2,3}, {"a","b"}, {"b","a"});
 *
 * @tparam PermuteBackend
 */
template <class PermuterBackend> class Permuter {
  public:
    /**
     * @brief Reshape the given lexicographic data vector from old to new index
     * ordering.
     *
     * @tparam DataType Data participating in the permutation.
     * @param data_in Input data to be transposed.
     * @param shape Current shape of the tensor data in each dimension.
     * @param data_out Output data following the transpose.
     * @param current_order Current index ordering of the tensor.
     * @param new_order New index ordering of the tensor.
     */
    template <class DataType>
    void Transpose(const std::vector<DataType> &data_in,
                   const std::vector<size_t> &shape,
                   std::vector<DataType> &data_out,
                   const std::vector<std::string> &current_order,
                   const std::vector<std::string> &new_order)
    {
        const std::set<std::string> idx_old(current_order.begin(),
                                            current_order.end());
        const std::set<std::string> idx_new(new_order.begin(), new_order.end());
        JET_ABORT_IF_NOT(idx_old.size() == current_order.size(),
                         "Duplicate existing indices found. Please ensure "
                         "indices are unique.");
        JET_ABORT_IF_NOT(idx_new.size() == new_order.size(),
                         "Duplicate transpose indices found. Please ensure "
                         "indices are unique.");
        JET_ABORT_IF_NOT(shape.size() == new_order.size(),
                         "Tensor shape does not match number of indices.");
        JET_ABORT_IF_NOT(
            Jet::Utilities::ShapeToSize(shape) == data_in.size(),
            "Tensor shape does not match given input tensor data.");
        JET_ABORT_IF_NOT(
            Jet::Utilities::ShapeToSize(shape) == data_out.size(),
            "Tensor shape does not match given output tensor data.");
        JET_ABORT_IF_NOT(
            idx_old == idx_new,
            "New indices are an invalid permutation of the existing indices");

        permuter_b_.Transpose(data_in, shape, data_out, current_order,
                              new_order);
    }

    /**
     * @brief Reshape the given lexicographic data vector from old to new index
     * ordering.
     *
     * @tparam DataType Data participating in the permutation.
     * @param data_in Input data to be transposed.
     * @param shape Current shape of the tensor data in each dimension.
     * @param current_order Current index ordering of the tensor.
     * @param new_order New index ordering of the tensor.
     * @return std::vector<DataType> Output data following the transpose.
     */
    template <class DataType>
    std::vector<DataType>
    Transpose(const std::vector<DataType> &data_in,
              const std::vector<size_t> &shape,
              const std::vector<std::string> &current_order,
              const std::vector<std::string> &new_order)
    {
        const std::set<std::string> idx_old(current_order.begin(),
                                            current_order.end());
        const std::set<std::string> idx_new(new_order.begin(), new_order.end());
        JET_ABORT_IF_NOT(idx_old.size() == current_order.size(),
                         "Duplicate existing indices found. Please ensure "
                         "indices are unique.");
        JET_ABORT_IF_NOT(idx_new.size() == new_order.size(),
                         "Duplicate transpose indices found. Please ensure "
                         "indices are unique.");
        JET_ABORT_IF_NOT(shape.size() == new_order.size(),
                         "Tensor shape does not match number of indices.");
        JET_ABORT_IF_NOT(Jet::Utilities::ShapeToSize(shape) == data_in.size(),
                         "Tensor shape does not match given tensor data.");
        JET_ABORT_IF_NOT(
            idx_old == idx_new,
            "New indices are an invalid permutation of the existing indices");

        JET_ABORT_IF(shape.empty(), "Tensor shape cannot be empty.");
        JET_ABORT_IF(new_order.empty(), "Tensor indices cannot be empty.");
        return permuter_b_.Transpose(data_in, shape, current_order, new_order);
    }

  protected:
    friend PermuterBackend;

  private:
    PermuterBackend permuter_b_;
};

} // namespace Jet
