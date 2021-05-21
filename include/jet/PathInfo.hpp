#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "Abort.hpp"
#include "TensorNetwork.hpp"

namespace Jet {

/**
 * `%PathStepInfo` is a POD that represents the contraction metadata associated
 * with a node in a TensorNetwork.
 */
struct PathStepInfo {
    /// Unique ID for this path step.
    size_t id;

    /// ID of the path step immediately succeeding this path step.
    size_t parent;

    /// IDs of the path steps immediately preceding this path step.
    std::pair<size_t, size_t> children;

    /// Name of this path step.
    std::string name;

    /// Indices of the node associated with this step step.
    std::vector<std::string> node_indices;

    /// Indices of the tensor associated with this path step.
    std::vector<std::string> tensor_indices;

    /// Tags of the node associated with this path step.
    std::vector<std::string> tags;

    /// Indices that are contracted during this path step.
    std::vector<std::string> contracted_indices;

    /// ID of a missing path step.
    static constexpr size_t MISSING_ID = std::numeric_limits<size_t>::max();
};

/**
 * `%PathInfo` represents a contraction path in a `TensorNetwork`.
 */
class PathInfo {
  public:
    /// Type of a node ID.
    using node_id_t = size_t;

    /// Type of a contraction path.
    using path_t = std::vector<std::pair<node_id_t, node_id_t>>;

    /// Type of the index-to-size map.
    using index_to_size_map_t = std::unordered_map<std::string, size_t>;

    /// Type of a `PathStepInfo` sequence.
    using steps_t = std::vector<PathStepInfo>;

    /**
     * @brief Constructs an empty path.
     */
    PathInfo() : num_leaves_(0) {}

    /**
     * @brief Constructs a populated path.
     *
     * @warning The program is aborted if the given path references a node ID
     *          which does not exist in the given tensor network.
     *
     * @tparam Tensor Type of the tensor in the tensor network.
     * @param tn Tensor network associated with the contraction path.
     * @param path Pairs of node IDs representing a raw contraction path.
     */
    template <typename Tensor>
    PathInfo(const TensorNetwork<Tensor> &tn, const path_t &path) : path_(path)
    {
        const auto &nodes = tn.GetNodes();
        num_leaves_ = nodes.size();

        steps_.reserve(num_leaves_);
        for (const auto &node : nodes) {
            constexpr size_t missing_id = PathStepInfo::MISSING_ID;
            PathStepInfo step{
                node.id,                  // id
                missing_id,               // parent
                {missing_id, missing_id}, // children
                node.name,                // name
                node.indices,             // node_indices
                node.tensor.GetIndices(), // tensor_indices
                node.tags,                // tags
                {},                       // contracted_indices
            };
            steps_.emplace_back(step);
        }

        for (const auto &[index, edge] : tn.GetIndexToEdgeMap()) {
            index_to_size_map_.emplace(index, edge.dim);
        }

        for (const auto &[node_id_1, node_id_2] : path) {
            JET_ABORT_IF_NOT(node_id_1 < steps_.size(),
                             "Node ID 1 in contraction path pair is invalid.");
            JET_ABORT_IF_NOT(node_id_2 < steps_.size(),
                             "Node ID 2 in contraction path pair is invalid.");
            ContractSteps_(node_id_1, node_id_2);
        }
    }

    /**
     * @brief Returns the index-to-size map of this path.
     *
     * @return Map which associates each index with a dimension size.
     */
    const index_to_size_map_t &GetIndexSizes() const noexcept
    {
        return index_to_size_map_;
    }

    /**
     * @brief Returns the number of leaf steps in this path.
     *
     * @return Number of leaves.
     */
    size_t GetNumLeaves() const noexcept { return num_leaves_; }

    /**
     * @brief Returns the raw contraction path of this path.
     *
     * @return Pairs of node IDs representing the contraction path.
     */
    const path_t &GetPath() const noexcept { return path_; }

    /**
     * @brief Returns the steps of this path.
     *
     * @return Collection of path steps.
     */
    const steps_t &GetSteps() const noexcept { return steps_; }

    /**
     * @brief Computes the number of floating-point operations needed to execute
     *        a path step (excluding any dependencies or child steps).
     *
     * @warning The program is aborted if the given step ID does not exist.
     *
     * @param id ID of a path step.
     *
     * @return Number of floating-point multiplications and additions needed to
     *         compute the tensor associated with the path step.
     */
    double GetPathStepFlops(size_t id) const
    {
        JET_ABORT_IF_NOT(id < steps_.size(), "Step ID is invalid.");

        if (id < num_leaves_) {
            // Tensor network leaves are constructed for free.
            return 0;
        }

        const auto &step = steps_[id];

        // Calculate the number of FLOPs needed for each dot product.
        double muls = 1;
        for (const auto &index : step.contracted_indices) {
            const auto it = index_to_size_map_.find(index);
            muls *= it == index_to_size_map_.end() ? 1 : it->second;
        }
        double adds = muls - 1;

        // Find the number of elements in the tensor.
        double size = 1;
        for (const auto &index : step.tensor_indices) {
            const auto it = index_to_size_map_.find(index);
            size *= it == index_to_size_map_.end() ? 1 : it->second;
        }
        return size * (muls + adds);
    }

    /**
     * @brief Computes the total number of floating-point operations needed to
     *        contract the tensors represented by this path.
     *
     * @param id ID of a path step.
     *
     * @return Total number of floating-point multiplications and additions.
     */
    double GetTotalFlops() const noexcept
    {
        double flops = 0;
        for (size_t i = num_leaves_; i < steps_.size(); i++) {
            flops += GetPathStepFlops(i);
        }
        return flops;
    }

    /**
     * @brief Computes the size of the tensor in a path step.
     *
     * @warning The program is aborted if the given step ID does not exist.
     *
     * @param id ID of a path step.
     *
     * @return Number of elements in the tensor associated with the path step.
     */
    double GetPathStepMemory(size_t id) const
    {
        JET_ABORT_IF_NOT(id < steps_.size(), "Step ID is invalid.");

        const auto &step = steps_[id];
        const auto &indices = step.tensor_indices;

        double memory = 1;
        for (const auto &index : indices) {
            const auto it = index_to_size_map_.find(index);
            memory *= it == index_to_size_map_.end() ? 1 : it->second;
        }
        return memory;
    }

    /**
     * @brief Computes the total memory required (up to a constant `sizeof`
     *        factor) to hold the tensors represented by this path.
     *
     * @return Total number of elements in the tensors.
     */
    double GetTotalMemory() const noexcept
    {
        double memory = 0;
        for (size_t id = 0; id < steps_.size(); id++) {
            memory += GetPathStepMemory(id);
        }
        return memory;
    }

  private:
    /// Contraction path through the tensor network associated with this path.
    path_t path_;

    /// Contraction metadata of the tensor network associated with this path.
    steps_t steps_;

    /// Number of leaf nodes in the tensor network associated with this path.
    size_t num_leaves_;

    /// Map that associates each index with its corresponding dimension size.
    /// This information is used to estimate memory requirements and floating-
    /// point operation counts.
    index_to_size_map_t index_to_size_map_;

    /**
     * @brief Contracts two path steps.
     *
     * @param step_id_1 ID of the first step to be contracted.
     * @param step_id_2 ID of the second step to be contracted.
     */
    void ContractSteps_(size_t step_id_1, size_t step_id_2) noexcept
    {
        using namespace Jet::Utilities;
        auto &step_1 = steps_[step_id_1];
        auto &step_2 = steps_[step_id_2];

        const size_t step_3_id = steps_.size();
        const auto step_3_contracted_indices =
            VectorIntersection(step_1.tensor_indices, step_2.tensor_indices);
        const auto step_3_node_indices =
            VectorDisjunctiveUnion(step_1.node_indices, step_2.node_indices);
        const auto step_3_name = step_3_node_indices.size()
                                     ? JoinStringVector(step_3_node_indices)
                                     : "_";
        const auto step_3_tensor_indices = VectorDisjunctiveUnion(
            step_1.tensor_indices, step_2.tensor_indices);
        const auto step_3_tags = VectorUnion(step_1.tags, step_2.tags);

        // Assign the parent IDs before references to `steps_` elements are
        // invalidated by `std::vector::emplace_back()`.
        step_1.parent = step_3_id;
        step_2.parent = step_3_id;

        PathStepInfo step_3{
            step_3_id,                 // id
            PathStepInfo::MISSING_ID,  // parent
            {step_id_1, step_id_2},    // children
            step_3_name,               // name
            step_3_node_indices,       // node_indices
            step_3_tensor_indices,     // tensor_indices
            step_3_tags,               // tags
            step_3_contracted_indices, // contracted_indices
        };
        steps_.emplace_back(step_3);
    }
};

}; // namespace Jet
