#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Abort.hpp"
#include "Utilities.hpp"

#include "spdlog/spdlog.h"

namespace Jet {

/**
 * @brief `%TensorNetwork` represents a tensor network.
 *
 * Internally, the nodes (tensors) of the network are stored in `nodes_` and the
 * tags are stored in `tag_to_node_ids_map`.  Furthermore, when a contraction
 * path is not specified, the edges (contractions) are stored in
 * `index_to_edge_map_`.  The contraction path followed by a call to Contract()
 * is stored in the `path_` member.
 *
 * @tparam Tensor Type of the tensor in the tensor network.  If `tensor` is an
 *                instance of `%Tensor`, the following expressions should be
 *                valid:
 *                \code{.cpp}
 *                std::vector<std::string> indices = tensor.GetIndices();
 *                std::vector<std::size_t> shape = tensor.GetShape();
 *
 *                tensor.InitIndicesAndShape(new_indices, new_shape);
 *
 *                Tensor shaped_tensor = Reshape(tensor, new_shape);
 *                Tensor sliced_tensor = Slice(tensor, index, value);
 *                Tensor contracted_tensor = ContractTensors(tensor, tensor);
 *                \endcode
 */
template <class Tensor> class TensorNetwork {
  public:
    /// Type of a node ID.
    using node_id_t = size_t;

    /**
     * @brief `%Node` is a POD which wraps tensors in a `TensorNetwork`.
     */
    struct Node {
        /// Unique ID for this node.
        node_id_t id;

        /// Name of this node.
        std::string name;

        /// Indices of this node.
        std::vector<std::string> indices;

        /// Tags of this node.
        std::vector<std::string> tags;

        /// Reports whether this node has been contracted.
        bool contracted;

        /// Tensor of this node.
        Tensor tensor;
        
        const std::string to_string() const{
            using namespace Utilities;
            std::ostringstream out;
            out << "Node[id=" << id << ",name=" << name << ",indices=" << indices << ",tags=" << tags << ",contracted=" << contracted << ",tensor=" << tensor << "]";
            return out.str();
        }
    };

    /**
     * @brief `%Edge` is a POD which connects `%Nodes` in a `TensorNetwork`.
     */
    struct Edge {
        /// Dimensionality of this edge.
        size_t dim;

        /// IDs of the nodes connected by this edge.
        std::vector<node_id_t> node_ids;

        /**
         * @brief Reports whether two edges are the same.
         *
         * Two edges are the same if they have the same dimension and contain
         * the same set of node IDs.
         *
         * @param other %Edge to compare to this edge.
         * @return True if this edge and the given edge are the same.
         */
        bool operator==(const Edge &other) const noexcept
        {
            using set_t = std::unordered_set<size_t>;
            const set_t lhs_ids(node_ids.begin(), node_ids.end());
            const set_t rhs_ids(other.node_ids.begin(), other.node_ids.end());
            return dim == other.dim && lhs_ids == rhs_ids;
        }

        const std::string to_string() const{
            using namespace Utilities;
            std::ostringstream out;
            out << "Edge[dim=" << dim << ",node_ids=" << node_ids << "]";
            return out.str();
        }
    };

    /// Type of a `Node` collection.
    using nodes_t = std::vector<Node>;

    /// Type of the index-to-edge map.
    using index_to_edge_map_t = std::unordered_map<std::string, Edge>;

    /// Type of the tag-to-node-IDs map.
    using tag_to_node_ids_map_t =
        std::unordered_multimap<std::string, node_id_t>;

    /// Type of a contraction path.
    using path_t = std::vector<std::pair<node_id_t, node_id_t>>;

    /**
     * @brief Returns the nodes in this `%TensorNetwork`.
     *
     * @return Collection of nodes.
     */
    const nodes_t &GetNodes() const noexcept { return nodes_; }

    /**
     * @brief Returns the index-to-edge map of this `%TensorNetwork`.
     *
     * @return Map which associates indices with edges.
     */
    const index_to_edge_map_t &GetIndexToEdgeMap() const noexcept
    {
        return index_to_edge_map_;
    }

    /**
     * @brief Returns the tag-to-node-IDs map of this `%TensorNetwork`.
     *
     * @return Map which associates tags with node IDs.
     */
    const tag_to_node_ids_map_t &GetTagToNodesMap() const noexcept
    {
        return tag_to_nodes_map_;
    }

    /**
     * @brief Returns the contaction path of this `%TensorNetwork`.
     *
     * @note The contraction path is always empty before Contract() is called.
     *
     * @return Pairs of node IDs representing the contraction path.
     */
    const path_t &GetPath() noexcept { return path_; }

    /**
     * @brief Returns the number of indices in this `%TensorNetwork`.
     *
     * @return Number of indices.
     */
    size_t NumIndices() const noexcept { return index_to_edge_map_.size(); }

    /**
     * @brief Returns the number of tensors in this `%TensorNetwork`.
     *
     * @return Number of tensors.
     */
    size_t NumTensors() const noexcept { return nodes_.size(); }

    /**
     * @brief Adds a tensor with the specified tags.
     *
     * @warning This function is not safe for concurrent execution.
     *
     * @param tensor Tensor to be added to this tensor network.
     * @param tags Tags to be associated with the tensor.
     */
    void AddTensor(const Tensor &tensor,
                   const std::vector<std::string> &tags) noexcept
    {
        using namespace Utilities;
        SPDLOG_DEBUG("Entered AddTensor(tensor=" + to_string(tensor) + ",tags="+ to_string(tags) + ")");

        const auto &indices = tensor.GetIndices();
        const auto name = DeriveNodeName_(indices);

        const Node node{
            nodes_.size(), // id
            name,          // name
            indices,       // indices
            tags,          // tags
            false,         // contracted
            tensor,        // tensor
        };
        nodes_.emplace_back(node);

        AddNodeToIndexMap_(node);
        AddNodeToTagMap_(node);

        SPDLOG_DEBUG("Leaving AddTensor()");
    }

    /**
     * @brief Slices a set of indices.
     *
     * The value taken along each axis is derived from the provided linear
     * index.
     *
     * Example: Suppose this tensor network contains an index "A0" of dimension
     *          2 and another index "B1" of dimension 3.  To slice these indices
     *          at values 0 and 1 respectively, use
     *
     *          \code{.cpp}
     *          tensor_network.SliceIndices({"A0", "B1"}, 0 + 1 * 2);
     *          \endcode
     *
     * @warning The program is aborted if a sliced index does not exist in this
     *          tensor network.
     *
     * @see Jet::Utilities::UnravelIndex()
     *
     * @param indices Indices to be sliced.
     * @param value Raveled value representing the element to take along each
     *              of the given indices.  See `UnravelIndex()` for details on
     *              how raveled values are interpreted.
     */
    void SliceIndices(const std::vector<std::string> &indices,
                      unsigned long long value) noexcept
    {
        using namespace Utilities;
        SPDLOG_DEBUG("Entered SliceIndices(indices=" + to_string(indices) + ",value="+ std::to_string(value) + ")");

        std::unordered_map<size_t, std::vector<size_t>> node_to_index_map;
        std::vector<size_t> index_sizes(indices.size());

        // Map each node ID to the indexes in `indices` to be sliced.
        for (size_t i = 0; i < indices.size(); i++) {
            const auto it = index_to_edge_map_.find(indices[i]);
            JET_ABORT_IF(it == index_to_edge_map_.end(),
                         "Sliced index does not exist.");

            const auto &edge = it->second;
            index_sizes[i] = edge.dim;

            for (const auto node_id : edge.node_ids) {
                auto &indices_indexes = node_to_index_map[node_id];
                indices_indexes.emplace_back(i);
            }
        }

        const auto values = Jet::Utilities::UnravelIndex(value, index_sizes);

        // Slice the tensors while updating the node indices and names.
        for (const auto &[node_id, indices_indexes] : node_to_index_map) {
            auto &node = nodes_[node_id];
            auto &tensor = node.tensor;

            for (const int indices_index : indices_indexes) {
                const auto &sliced_index = indices[indices_index];
                const auto &sliced_value = values[indices_index];

                // Copy these members to avoid messing with the internal
                // representation of the tensor.
                auto tensor_indices = tensor.GetIndices();
                auto tensor_shape = tensor.GetShape();

                // Find the position of the sliced index in the tensor.
                const auto it = std::find(tensor_indices.begin(),
                                          tensor_indices.end(), sliced_index);
                const auto offset = std::distance(tensor_indices.begin(), it);

                tensor_shape.erase(tensor_shape.begin() + offset);
                tensor_indices.erase(tensor_indices.begin() + offset);

                tensor = SliceIndex(tensor, sliced_index, sliced_value);
                if (!tensor_indices.empty()) {
                    tensor = Reshape(tensor, tensor_shape);
                }

                // Erase the sliced index from the tensor.
                tensor.InitIndicesAndShape(tensor_indices, tensor_shape);

                // Do not erase the sliced index from the node. Instead,
                // annotate it with the sliced value.
                for (auto &node_index : node.indices) {
                    if (node_index == sliced_index) {
                        node_index += '(';
                        node_index += std::to_string(sliced_value);
                        node_index += ')';
                    }
                }
            }

            node.name = DeriveNodeName_(node.indices);
        }

        // Erase the sliced indices from the index-to-edge map.
        for (const auto &index : indices) {
            index_to_edge_map_.erase(index);
        }
        SPDLOG_DEBUG("Leaving SliceIndices()");
    }

    /**
     * @brief Contracts this tensor network.
     *
     * If the given contraction path is empty, the first two nodes belonging to
     * each edge of the tensor network are contracted.  Otherwise, the specified
     * contraction path is used to guide the contraction of nodes.
     *
     * @note After this function is invoked, the `path_` member of this tensor
     *       network will reflect the contractions performed in this function.
     *
     * @warning The program is aborted if the tensor network is empty.
     *
     * @param path Contraction path specified as a list of node ID pairs.
     * @return Tensor associated with the result of the final contraction.
     */
    const Tensor &Contract(const path_t &path = {}) noexcept
    {
        using namespace Utilities;
        SPDLOG_DEBUG("Entered Contract(path=" + to_string(path) + ")");

        JET_ABORT_IF(nodes_.empty(),
                     "An empty tensor network cannot be contracted.");

        if (!path.empty()) {
            for (const auto &[node_id_1, node_id_2] : path) {
                JET_ABORT_IF_NOT(node_id_1 < nodes_.size(),
                                 "Node ID 1 in contraction pair is invalid.");
                JET_ABORT_IF_NOT(node_id_2 < nodes_.size(),
                                 "Node ID 2 in contraction pair is invalid.");

                const size_t node_id_3 = ContractNodes_(node_id_1, node_id_2);

                const auto &node_1 = nodes_[node_id_1];
                const auto &node_2 = nodes_[node_id_2];
                const auto &node_3 = nodes_[node_id_3];
                UpdateIndexMapAfterContraction_(node_1, node_2, node_3);
            }
            path_ = path;
        }
        else {
            ContractEdges_();
            ContractScalars_();
        }
        SPDLOG_DEBUG("Leaving Contract(return=" + to_string(nodes_.back().tensor) + ")");
        return nodes_.back().tensor;
    }

  private:
    /// Nodes inside this tensor network.
    nodes_t nodes_;

    /// Map that associates each index with its corresponding edge.
    /// Not used when a contraction path is specified.
    index_to_edge_map_t index_to_edge_map_;

    /// Map that associates each tag with a list of nodes that contain that tag.
    /// Not used when a contraction path is specified.
    tag_to_node_ids_map_t tag_to_nodes_map_;

    /// Contraction path representing pairs of nodes to be contracted.
    std::vector<std::pair<size_t, size_t>> path_;

    /**
     * @brief Updates the index-to-edge map with a newly-constructed node.
     *
     * @param node %Node to be used to update the map.
     */
    void AddNodeToIndexMap_(const Node &node) noexcept
    {
        using namespace Utilities;
        auto s = node.to_string();
        SPDLOG_DEBUG("Entered AddNodeToIndexMap_(node=" + s + ")");

        const auto &indices = node.indices;
        const auto &shape = node.tensor.GetShape();

        for (size_t i = 0; i < indices.size(); i++) {
            if (shape[i] < 2) {
                continue;
            }

            const auto it = index_to_edge_map_.find(indices[i]);
            if (it != index_to_edge_map_.end()) {
                auto &edge = it->second;
                edge.node_ids.emplace_back(node.id);
            }
            else {
                const Edge edge{
                    shape[i],  // dim
                    {node.id}, // node_ids
                };
                index_to_edge_map_.emplace(indices[i], edge);
            }
        }
        SPDLOG_DEBUG("Leaving AddNodeToIndexMap_()");
    }

    /**
     * @brief Updates the tag-to-node-IDs map with a newly-constructed node.
     *
     * @param node %Node to be used to update the map.
     */
    void AddNodeToTagMap_(const Node &node) noexcept
    {
        SPDLOG_DEBUG("Entered AddNodeToTagMap_(node=" + node.to_string() + ")");

        for (const auto &tag : node.tags) {
            tag_to_nodes_map_.emplace(tag, node.id);
        }
        SPDLOG_DEBUG("Leaving AddNodeToTagMap_()");
    }

    /**
     * @brief Contracts two nodes.
     *
     * @param node_id_1 ID of the first node to be contracted.
     * @param node_id_2 ID of the second node to be contracted.
     * @return ID of the new node representing the contraction result.
     */
    size_t ContractNodes_(size_t node_id_1, size_t node_id_2) noexcept
    {
        SPDLOG_DEBUG("Entered ContractNodes_(node_id_1=" + std::to_string(node_id_1) + ",node_id_2=" + std::to_string(node_id_2) + ")");

        // Make sure node 1 has at least as many indices as node 2.
        const size_t node_1_size = nodes_[node_id_1].tensor.GetIndices().size();
        const size_t node_2_size = nodes_[node_id_2].tensor.GetIndices().size();
        if (node_1_size <= node_2_size) {
            std::swap(node_id_1, node_id_2);
        }

        auto &node_1 = nodes_[node_id_1];
        auto &node_2 = nodes_[node_id_2];
        const auto tensor_3 = ContractTensors(node_1.tensor, node_2.tensor);

        node_1.contracted = true;
        node_2.contracted = true;

        using namespace Jet::Utilities;
        const auto node_3_tags = VectorUnion(node_1.tags, node_2.tags);
        const auto node_3_indices =
            VectorDisjunctiveUnion(node_1.indices, node_2.indices);
        const auto node_3_name = DeriveNodeName_(node_3_indices);

        Node node_3{
            nodes_.size(),  // id
            node_3_name,    // name
            node_3_indices, // indices
            node_3_tags,    // tags
            false,          // contracted
            tensor_3,       // tensor
        };
        nodes_.emplace_back(node_3);
        SPDLOG_DEBUG("Leaving ContractNodes_(return=" + std::to_string(node_3.id) + ")");
        return node_3.id;
    }

    /**
     * @brief Updates the index-to-edge map following a contraction.
     *
     * @param node_1 First node that was contracted.
     * @param node_2 Second node that was contracted.
     * @param node_3 Node that was created from the contraction.
     */
    void UpdateIndexMapAfterContraction_(const Node &node_1, const Node &node_2,
                                         const Node &node_3) noexcept
    {
        using namespace Utilities;
        SPDLOG_DEBUG("Entered UpdateIndexMapAfterContraction_(node_1=" + node_1.to_string() + ",node_2=" + node_2.to_string() + ",node_3=" + node_3.to_string() + ")");

        // Replace the IDs of the contracted nodes with the ID of the new node
        // in the index-to-edge map.
        for (auto &index : node_3.indices) {
            const auto it = index_to_edge_map_.find(index);
            if (it == index_to_edge_map_.end()) {
                continue;
            }

            Edge &edge = it->second;
            for (auto &node_id : edge.node_ids) {
                if (node_id == node_1.id || node_id == node_2.id) {
                    node_id = node_3.id;
                }
            }
        }

        // Delete the contracted indices in the index-to-edge map.
        const auto contracted_indices = Jet::Utilities::VectorIntersection(
            node_1.tensor.GetIndices(), node_2.tensor.GetIndices());

        for (const auto &index : contracted_indices) {
            index_to_edge_map_.erase(index);
        }
        SPDLOG_DEBUG("Leaving UpdateIndexMapAfterContraction_()");
    }

    /**
     * @brief Contracts all the edges (shared indices) in this `%TensorNetwork`.
     */
    void ContractEdges_() noexcept
    {
        SPDLOG_DEBUG("Entered ContractEdges_()");

        // Create a copy of the indices from the index-to-edge map since this
        // map will be modified in the next loop.
        std::vector<std::string> indices;
        for (const auto &[index, _] : index_to_edge_map_) {
            indices.emplace_back(index);
        }

        for (const auto &index : indices) {
            const auto it = index_to_edge_map_.find(index);
            if (it == index_to_edge_map_.end()) {
                continue;
            }

            const auto &node_ids = it->second.node_ids;
            if (node_ids.size() != 2) {
                continue;
            }

            const size_t node_id_0 = node_ids[0];
            const size_t node_id_1 = node_ids[1];
            const size_t node_id_2 = ContractNodes_(node_id_0, node_id_1);
            UpdateIndexMapAfterContraction_(
                nodes_[node_id_0], nodes_[node_id_1], nodes_[node_id_2]);

            path_.emplace_back(node_id_0, node_id_1);
        }
        SPDLOG_DEBUG("Leaving ContractEdges_()");
    }

    /**
     * @brief Contracts all the scalars in this `%TensorNetwork`.
     */
    void ContractScalars_() noexcept
    {
        SPDLOG_DEBUG("Entered ContractScalars_()");

        std::vector<size_t> node_ids;
        for (size_t i = 0; i < nodes_.size(); i++) {
            const bool scalar = nodes_[i].tensor.GetIndices().empty();
            if (scalar) {
                node_ids.emplace_back(i);
            }
        }

        if (node_ids.size() >= 2) {
            // Use `node_id` to track the cumulative contracted tensor.
            size_t node_id = node_ids[0];
            for (size_t i = 1; i < node_ids.size(); i++) {
                const size_t node_id_1 = node_id;
                const size_t node_id_2 = node_ids[i];
                path_.emplace_back(node_id_1, node_id_2);
                node_id = ContractNodes_(node_id_1, node_id_2);
            }
        }
        SPDLOG_DEBUG("Leaving ContractScalars_()");
    }

    /**
     * @brief Derives the name of a node from a sequence of indices.
     *
     * @param indices Indices of the node.
     * @return Name of the node.
     */
    std::string
    DeriveNodeName_(const std::vector<std::string> &indices) const noexcept
    {
        SPDLOG_DEBUG("Entered DeriveNodeName_(indices=" + Utilities::to_string(indices) + ")");
        auto node_name = indices.size() ? Jet::Utilities::JoinStringVector(indices) : "_";
        SPDLOG_DEBUG("Leaving DeriveNodeName_(return=" + node_name + ")");
        return node_name;
    }
};

/**
 * @brief Streams a `TensorNetwork` to an output stream.
 *
 * @tparam Tensor Type of the tensor in the tensor network.
 * @param out Output stream to be modified.
 * @param tn Tensor network to be streamed.
 * @return Reference to the given output stream.
 */
template <class Tensor>
inline std::ostream &operator<<(std::ostream &out,
                                const TensorNetwork<Tensor> &tn)
{
    // Overloads the "<<" operator between a std::ostream and std::vector.
    using namespace Jet::Utilities;

    out << "TensorNetwork[Nodes[" << std::endl;

    for (size_t n_idx = 0; n_idx < tn.GetNodes().size(); n_idx++) {
        out << tn.GetNodes()[n_idx];
        if(n_idx != tn.GetNodes().size()-1)
            out << ",";
    }
    out << "],Edges[";

    typename TensorNetwork<Tensor>::index_to_edge_map_t::iterator it = tn.GetIndexToEdgeMap().begin();
    while(it != tn.GetIndexToEdgeMap().end())
    {
        out << "index=" << it.first << "," << it.second << ",";
        if(std::distance(tn.GetIndexToEdgeMap().begin(), it) != tn.GetIndexToEdgeMap().size()-1)
            out << ",";
        it++;
    }
    out << "]]";

    return out;
}

template <class Tensor>
inline std::string to_string( const TensorNetwork<Tensor> &tn )
{
    std::ostringstream oss;
    oss << tn;
    return oss.str();
}

}; // namespace Jet
