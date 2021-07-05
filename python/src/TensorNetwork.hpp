#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Type.hpp"
#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the `TensorNetwork` class.
 *
 * @tparam T Template parameter of the `Tensor` class.
 * @param m Jet pybind11 module.
 */
template <class T> void AddBindingsForTensorNetwork(py::module_ &m)
{
    using TensorNetwork = Jet::TensorNetwork<Jet::Tensor<T>>;
    using NodeID_t = typename Jet::TensorNetwork<Jet::Tensor<T>>::NodeID_t;
    using Node = typename Jet::TensorNetwork<Jet::Tensor<T>>::Node;
    using Edge = typename Jet::TensorNetwork<Jet::Tensor<T>>::Edge;

    const std::string class_name = "TensorNetwork" + Type<T>::suffix;
    const std::string class_name_node = "TensorNetworkNode" + Type<T>::suffix;
    const std::string class_name_edge = "TensorNetworkEdge" + Type<T>::suffix;

    auto cls =
        py::class_<TensorNetwork>(m, class_name.c_str(),
                                  "TensorNetwork represents a tensor network.")

            // Static properties
            // -----------------------------------------------------------------

            .def_property_readonly_static(
                "dtype", [](const py::object &) { return Type<T>::dtype; },
                "Data type of this tensor network.")

            // Constructors
            // -----------------------------------------------------------------

            .def(py::init<>(), "Constructs an empty tensor network.")

            // Magic methods
            // -----------------------------------------------------------------

            .def("__str__",
                 [](const TensorNetwork &tn) {
                     std::stringstream ss;
                     ss << tn;
                     return ss.str();
                 })

            // Properties
            // -----------------------------------------------------------------

            .def_property_readonly("index_to_edge_map",
                                   &TensorNetwork::GetIndexToEdgeMap,
                                   "Mapping from indices to edges.")

            .def_property_readonly(
                "tag_to_node_id_map",
                [](const TensorNetwork &tn) {
                    std::unordered_map<std::string, std::vector<NodeID_t>> map;

                    for (const auto &[tag, node_id] : tn.GetTagToNodesMap()) {
                        map[tag].emplace_back(node_id);
                    }

                    return map;
                },
                "Mapping from tags to node IDs.")

            .def_property_readonly("path", &TensorNetwork::GetPath, R"(
Path by which this tensor network was contracted. Is empty if ``Contract()`` has
not been called.
                )")

            .def_property_readonly("nodes", &TensorNetwork::GetNodes,
                                   "List of nodes in this tensor network.")

            .def_property_readonly("num_tensors", &TensorNetwork::NumTensors,
                                   "Number of tensors in this tensor network.")

            .def_property_readonly(
                "num_indices", &TensorNetwork::NumIndices,
                "Number of unique indices in this tensor network.")

            // Other
            // -----------------------------------------------------------------

            .def("add_tensor", &TensorNetwork::AddTensor, py::arg("tensor"),
                 py::arg("tags") = std::vector<std::string>(), R"(
Adds a tensor to a tensor network.

Args:
    tensor (Tensor): Tensor to add.
    tags (List[str]): List of tags to associate with the tensor.

Returns:
    int: Node ID assigned to the tensor.
                )")

            .def("slice_indices", &TensorNetwork::SliceIndices, R"(
Slices a set of indices. The value taken along each axis is derived from the
provided linear index.

Args:
    indices (List[str]): List of string indices to be sliced.
    value (int): Raveled index value representing the element to take along each
        of the indices.
                )")

            .def("contract", &TensorNetwork::Contract,
                 py::arg("path") = std::vector<std::pair<size_t, size_t>>(), R"(
Contracts a tensor network along an optionally-provided path.

Args:
    path (Optional[Tuple[str, str]]): Optional contraction path specified as a
        list of node ID pairs.

Returns:
    Tensor: Tensor associated with the result of the final contraction.
                )");

    py::class_<Node>(cls, class_name_node.c_str())
        .def_readonly("id", &Node::id)
        .def_readonly("name", &Node::name)
        .def_readonly("indices", &Node::indices)
        .def_readonly("tags", &Node::tags)
        .def_readonly("contracted", &Node::contracted)
        .def_readonly("tensor", &Node::tensor);

    py::class_<Edge>(cls, class_name_edge.c_str())
        .def_readonly("dim", &Edge::dim)
        .def_readonly("node_ids", &Edge::node_ids)
        .def("__eq__", &Edge::operator==);
}
