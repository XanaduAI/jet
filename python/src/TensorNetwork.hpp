#pragma once

#include <string>

#include <Jet.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <class Tensor>
void AddBindingsForTensorNetwork(py::module_ &m, const char *name)
{
    using TensorNetwork = Jet::TensorNetwork<Tensor>;

    py::class_<TensorNetwork>(m, name, R"(
        TensorNetwork represents a tensor network)")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<>(), R"(
            Constructs an empty tensor network)")

        // Properties
        // ---------------------------------------------------------------------

        .def_property_readonly("nodes", &TensorNetwork::GetNodes,
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
        // ---------------------------------------------------------------------

        .def(
            "__getitem__",
            [](const TensorNetwork &tn, size_t node_id) {
                return tn.GetNodes()[node_id];
            },
            py::arg("node_id"), R"(
            Returns the tensor network node with the given ID. 

            Params:
                node_id: ID of the node

            Returns:
                Tensor network node)")

        // Other
        // ---------------------------------------------------------------------

        .def(
            "get_node_ids_by_tag",
            [](const TensorNetwork &tn, const std::string &tag) {
                auto [begin, end] = tn.GetTagToNodesMap().equal_range(tag);
                std::vector<size_t> ids;

                for (auto it = begin; it != end; ++it) {
                    ids.push_back(it->second);
                }

                return ids;
            },
            R"(Returns a list of IDs of nodes with the given tag.)")

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

        .def("contract", &TensorNetwork::Contract,
             R"(Contract tensor network along an optionally provided path)");

    std::string node_class_name = std::string(name) + "Node";
    py::class_<typename TensorNetwork::Node>(m, node_class_name.c_str());
}
