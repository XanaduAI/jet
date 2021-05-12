#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Jet.hpp>

template<class Tensor> AddBindingsForTensorNewtork(py::_module &m, const char *name)
{
    using TensorNetwork = Jet::TensorNetwork<Tensor>;

    py::class_<TensorNetwork>(m, name, R"(
        TensorNetwork represents a tensor network)")

        .def_property_readonly("nodes", &TensorNetwork::GetNodes,
                R"(A list of the nodes in this tensor network)")

        .def_property_readonly("path", &TensorNetwork::GetPath,
                R"(The path by which this tensor network was contracted.
                Is empty if Contract() has not been called.)")

        .def_property_readonly("num_tensors", &TensorNetwork::NumTensors(),
                R"(The number of tensors in this tensor network)")

        .def_property_readonly("num_indices", &TensorNetwork::NumIndices(),
                R"(The number of unique indices in this tensor network)")

        .def("get_nodes_by_tag", 
                [](const TensorNetwork &tn, const std::string &tag) {
                    auto [begin, end] = tn.GetTagsToNodesMap().equal_range(tag);
                    return std::vector<TensorNetwork::node_id_it>(begin, end);
                }
                R"(Returns a list of nodes with the given tag.)")

        .def("add_tensor", &TensorNetwork::AddTensor(),
                R"(Add tensor to this network, with specified tags)")

        .def("contract", &TensorNewtork::Contract(),
                R"(Contract tensor network along an optionally provided path)");
    
}
