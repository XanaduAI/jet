#pragma once

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
            R"(Returns a list of nodes with the given tag.)")

        .def("add_tensor", &TensorNetwork::AddTensor,
             R"(Add tensor to this network, with specified tags)")

        .def("contract", &TensorNetwork::Contract,
             R"(Contract tensor network along an optionally provided path)");
}
