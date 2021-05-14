#pragma once

#include <pybind11/pybind11.h>

#include <Jet.hpp>

namespace py = pybind11;

template <class Tensor, class... Tensors>
void bind_constructors(py::class_<Jet::PathInfo> &c)
{
    c.def(py::init<const Jet::TensorNetwork<Tensor> &,
                   const Jet::PathInfo::path_t &>(),
          py::arg("tn"), py::arg("path"), R"(
            Constructs a populated PathInfo for the given path
            through a tensor network

            Args:
                tn: Tensor network
                path: Pairs of integers representing contraction path through
                tensor network
            )");

    if constexpr (sizeof...(Tensors) == 0) {
        return;
    }
    else {
        bind_constructors<Tensors...>(c);
    }
}

template <class... Tensors> void AddBindingsForPathInfo(py::module_ &m)
{

    auto cls = py::class_<Jet::PathInfo>(m, "PathInfo", R"(
        PathInfo represents a contraction path in a Tensor Network)")

                   .def(py::init<>(), R"(Constructs an empty path)")

                   .def_property_readonly("index_to_size_map",
                                          &Jet::PathInfo::GetIndexSizes, R"(
            A map of indices to their dimension)")

                   .def_property_readonly("num_leaves",
                                          &Jet::PathInfo::GetNumLeaves, R"(
            Number of leaf steps in this path)")

                   .def_property_readonly("path", &Jet::PathInfo::GetPath, R"(
            The contraction path)")

                   .def_property_readonly("steps", &Jet::PathInfo::GetSteps, R"(
            The steps of this path)")

                   .def("get_total_flops", &Jet::PathInfo::GetTotalFlops, R"(
            Computes total number of floating-point operations needed
            contract the tensor network along this path)")

                   .def("get_total_memory", &Jet::PathInfo::GetTotalMemory, R"(
            Computes total memory required to contract the tensor
            network along this path)");

    // Create bindings for each Tensor type
    bind_constructors<Tensors...>(cls);
}
