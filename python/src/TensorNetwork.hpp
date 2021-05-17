#pragma once

#include <Jet.hpp>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <class Tensor>
void AddBindingsForTensorNetwork(py::module_ &m, const char *name)
{
    using TensorNetwork = Jet::TensorNetwork<Tensor>;
    using node_id_t = typename Jet::TensorNetwork<Tensor>::node_id_t;

    auto cls =
        py::class_<TensorNetwork>(m, name, R"(
        TensorNetwork represents a tensor network)")

            // Constructors
            // --------------------------------------------------------------------

            .def(py::init<>(), R"(
            Constructs an empty tensor network)")

            // Properties
            // --------------------------------------------------------------------

            .def_property_readonly(
                "nodes", &TensorNetwork::GetNodes,
                R"(A list of the nodes in this tensor network)")

            .def_property_readonly(
                "path", &TensorNetwork::GetPath,
                R"(The path by which this tensor network was contracted.
                Is empty if Contract() has not been called.)")

            .def_property_readonly(
                "num_tensors", &TensorNetwork::NumTensors,
                R"(The number of tensors in this tensor network)")

            .def_property_readonly(
                "num_indices", &TensorNetwork::NumIndices,
                R"(The number of unique indices in this tensor network)")

            // Magic methods
            // --------------------------------------------------------------------

            .def(
                "__getitem__",
                [](const TensorNetwork &tn, node_id_t node_id) {
                    return tn.GetNodes().at(node_id).tensor;
                },
                py::arg("node_id"), R"(
            Returns Tensor in the network with the given node id. 

            Params:
                node_id: ID of the node

            Returns:
                Tensor network node)")

            // Other
            // --------------------------------------------------------------------

            .def(
                "get_node_ids_by_tag",
                [](const TensorNetwork &tn, const std::string &tag) {
                    auto [begin, end] = tn.GetTagToNodesMap().equal_range(tag);
                    std::vector<size_t> nodes;

                    for (auto it = begin; it != end; ++it) {
                        nodes.push_back(it->second);
                    }

                    return nodes;
                },
                R"(Returns a list of node ids for Tensors associated with the given tag.)")

            .def("add_tensor", &TensorNetwork::AddTensor,
                 R"(Add tensor to network.
                
                Args:
                    tensor: Tensor to add
                    tags: List of string tags to associate to tensor)")

            .def("slice_indices", &TensorNetwork::SliceIndices,
                 R"(Slices a set of indices. 
                
                The value taken along each axis is derived from the provided
                linear index.

                Args:
                    indices: list of string indices to be sliced
                    value: Raveled index value representing the element
                    to take along each of the indices.
                )")

            .def(
                "contract", &TensorNetwork::Contract,
                R"(Contract tensor network along an optionally provided path)");
}
