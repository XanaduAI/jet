#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Abort.hpp"
#include "Utilities.hpp"

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
 * @tparam Tensor Type of the tensor in the tensor network.  The only
 *                requirement for this type is that the following member
 *                functions exist:
 *                \code{.cpp}
 *     std::vector<std::string> GetIndices();
 *     std::vector<std::size_t> GetShape();
 *
 *     void InitIndicesAndShape(const std::vector<std::string>&,
 *                              const std::vector<std::size_t>&);
 *
 *     static Tensor Reshape(const Tensor&, const std::vector<std::string>&);
 *     static Tensor SliceIndex(const Tensor&, const std::string&, size_t);
 *     static Tensor ContractTensors(const Tensor&, const Tensor&);
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
     * @brief Adds a tensor with the specified tags and returns its
     *        assigned ID.
     *
     * @warning This function is not safe for concurrent execution.
     *
     * @param tensor Tensor to be added to this tensor network.
     * @param tags Tags to be associated with the tensor.
     *
     * @return Node ID assigned to the tensor.
     */
    node_id_t AddTensor(const Tensor &tensor,
                        const std::vector<std::string> &tags) noexcept
    {
        node_id_t id = nodes_.size();
        nodes_.emplace_back(Node{
            id,                                   // id
            DeriveNodeName_(tensor.GetIndices()), // name
            tensor.GetIndices(),                  // indices
            tags,                                 // tags
            false,                                // contracted
            tensor                                // tensor
        });

        AddNodeToIndexMap_(nodes_[id]);
        AddNodeToTagMap_(nodes_[id]);

        return id;
    }

  // void FillRandom(size_t num_tensors,
  // 		  size_t num_edges,
  // 		  int seed = 0,
  // 		  size_t min_dimension = 2,
  // 		  size_t max_dimension = 2)
  // {

  //   //We demand that two times the indices is greater than the number of tensors, so we can
  //   //have atleast one index per tensor.
  //   JET_ABORT_IF(num_tensors >= 2*num_edges, "num_tensors >= 2*num_edges);

  //   std::vector<std::string> index_database(num_edges);
  // 		 std::vector<std::vector<std::string>> 
  // 		 std::vector<std::vector<std::size_t>>
    
    
    
  // }
  
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
                      unsigned long long value)
    {
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

                tensor = Tensor::SliceIndex(tensor, sliced_index, sliced_value);
                if (!tensor_indices.empty()) {
                    tensor = Tensor::Reshape(tensor, tensor_shape);
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
    const Tensor &Contract(const path_t &path = {})
    {
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

        return nodes_.back().tensor;
    }



  size_t RankSimplify
  (
   const std::vector<std::string>
   &ignore_tags
  )
  {
    size_t num_nodes = NumTensors();
    using namespace Jet::Utilities;
    size_t counter_prev = -1;
    size_t counter = 0;
    // no tensor contractions can be done that reduce rank
    // after counter_prev catches up to counter

    std::unordered_set<size_t> ignore_node;
    for (auto i : ignore_tags) {
      if (tag_to_nodes_map_.count(i)) {
	auto its = tag_to_nodes_map_.equal_range(i);
	for (auto it = its.first; it != its.second; ++it) {
	  ignore_node.insert(it->second);
	}	
      }
    }

    while (counter_prev != counter) {
      std::pair<size_t, size_t> node_pair;
      int diff = 0;
      bool found = false;
      for (auto &i : index_to_edge_map_) {
	Edge &e = i.second;
	size_t a = e.node_ids[0];
	if (ignore_node.count(a))
	  continue;
	size_t b = e.node_ids[1];
	if (ignore_node.count(b))
	  continue;
	Node node_a = nodes_[a];
	Node node_b = nodes_[b];
	const std::vector<std::string> &contracted_indices =
	  VectorIntersection(node_a.tensor.GetIndices(),
			     node_b.tensor.GetIndices());
	size_t node_a_rank = node_a.tensor.GetShape().size();
	size_t node_b_rank = node_b.tensor.GetShape().size();
	int node_c_rank =
	  node_a_rank + node_b_rank - 2 * contracted_indices.size();
	int max_rank =
	  (node_a_rank > node_b_rank) ? node_a_rank : node_b_rank;
	int diff_temp = node_c_rank - max_rank;
	if (diff_temp <= 0 && (found == false || diff_temp < diff)) {
	  found = true;
	  diff = diff_temp;
	  node_pair = {a, b};
	}
      }
      if (found) {
	size_t new_node = ContractNodes_(node_pair.first,node_pair.second);
	UpdateIndexMapAfterContraction_(nodes_[node_pair.first],
					nodes_[node_pair.second],
					nodes_[new_node]);
	counter++;
      }
      counter_prev++;
    }
    TensorNetwork<Tensor> tn;
    for (int i = 0; i < nodes_.size(); i++) {
      if (nodes_[i].contracted == false)
	tn.AddTensor(nodes_[i].tensor, nodes_[i].tags);
    }
    *(this) = tn;

    return counter;
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
    }

    /**
     * @brief Updates the tag-to-node-IDs map with a newly-constructed node.
     *
     * @param node %Node to be used to update the map.
     */
    void AddNodeToTagMap_(const Node &node) noexcept
    {
        for (const auto &tag : node.tags) {
            tag_to_nodes_map_.emplace(tag, node.id);
        }
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
        auto &node_1 = nodes_[node_id_1];
        auto &node_2 = nodes_[node_id_2];
        const auto tensor_3 =
            Tensor::ContractTensors(node_1.tensor, node_2.tensor);

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
    }

    /**
     * @brief Contracts all the edges (shared indices) in this `%TensorNetwork`.
     */
    void ContractEdges_() noexcept
    {
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
    }

    /**
     * @brief Contracts all the scalars in this `%TensorNetwork`.
     */
    void ContractScalars_() noexcept
    {
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
        return indices.size() ? Jet::Utilities::JoinStringVector(indices) : "_";
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

    out << "Printing Nodes" << std::endl;
    for (const auto &node : tn.GetNodes()) {
        out << node.id << ' ' << node.name << ' ' << node.tags << std::endl;
    }
    out << "Printing Edges" << std::endl;
    for (const auto &[index, edge] : tn.GetIndexToEdgeMap()) {
        out << index << ' ' << edge.dim << ' ' << edge.node_ids << std::endl;
    }
    return out;
}

}; // namespace Jet
