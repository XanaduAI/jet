#pragma once

#include <Jet.hpp>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <vector>

template <class MultiMap> class multimap_proxy {

  public:
    using key_type = typename MultiMap::key_type;
    using mapped_type = typename MultiMap::mapped_type;

    multimap_proxy(const MultiMap &map) : map(map){};

    multimap_proxy(MultiMap &&) = delete;

    std::vector<mapped_type> operator[](const key_type &key)
    {
        const auto &[begin, end] = map.equal_range(key);

        return std::vector<mapped_type>(begin, end);
    }

  private:
    const MultiMap &map;
};

namespace py = pybind11;

template <class Tensor>
void AddBindingsForTensorNetwork(py::module_ &m, const char *name)
{

    using TensorNetwork = Jet::TensorNetwork<Tensor>;
    using node_id_t = typename Jet::TensorNetwork<Tensor>::node_id_t;
    using Node = typename Jet::TensorNetwork<Tensor>::Node;
    using Edge = typename Jet::TensorNetwork<Tensor>::Edge;

    auto cls =
        py::class_<TensorNetwork>(m, name, R"(
        TensorNetwork represents a tensor network)")

            // Constructors
            // --------------------------------------------------------------------

            .def(py::init<>(), R"(
            Constructs an empty tensor network)")

            // Properties
            // --------------------------------------------------------------------

            .def_property_readonly("index_to_edge_map",
                                   &TensorNetwork::GetIndexToEdgeMap, R"(
                A map of indices to edges.)")

            .def_property_readonly(
                "tag_to_node_id_map",
                [](const TensorNetwork &tn) {
                    std::unordered_map<std::string, std::vector<node_id_t>> map;

                    for (const auto &[tag, node_id] : tn.GetTagToNodesMap()) {
                        map[tag].emplace_back(node_id);
                    }

                    return map;
                },
                R"(Returns a list of node ids for Tensors associated with the given tag)")

            .def_property_readonly("path", &TensorNetwork::GetPath, R"(
                The path by which this tensor network was contracted.
                Is empty if Contract() has not been called.)")

            .def_property_readonly("nodes", &TensorNetwork::GetNodes, R"(
                A list of the nodes in this tensor network.)")

            .def_property_readonly("num_tensors", &TensorNetwork::NumTensors,
                                   R"(
                The number of tensors in this tensor network)")

            .def_property_readonly("num_indices", &TensorNetwork::NumIndices,
                                   R"(
                The number of unique indices in this tensor network)")

            // Other
            // --------------------------------------------------------------------

            .def("add_tensor", &TensorNetwork::AddTensor, R"(
                Adds a tensor to a tensor network.
                
                Args:
                    tensor: Tensor to add
                    tags: List of string tags to associate to tensor

                Returns:
                    Node ID assigned to tensor)")

            .def("slice_indices", &TensorNetwork::SliceIndices, R"(
                Slices a set of indices. 
                
                The value taken along each axis is derived from the provided
                linear index.

                Args:
                    indices: list of string indices to be sliced
                    value: Raveled index value representing the element
                    to take along each of the indices.
                )")

            .def("contract", &TensorNetwork::Contract, R"(
                Contracts a tensor network along an optionally-provided path)");

    py::class_<Node>(cls, (std::string(name) + "Node").c_str())
        .def_readonly("id", &Node::id)
        .def_readonly("name", &Node::name)
        .def_readonly("indices", &Node::indices)
        .def_readonly("tags", &Node::tags)
        .def_readonly("contracted", &Node::contracted)
        .def_readonly("tensor", &Node::tensor);

    py::class_<Edge>(cls, (std::string(name) + "Edge").c_str())
        .def_readonly("dim", &Edge::dim)
        .def_readonly("node_ids", &Edge::node_ids)
        .def("__eq__", &Edge::operator==);
}
