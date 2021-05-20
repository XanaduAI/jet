#pragma once

#include <pybind11/pybind11.h>

#include <Jet.hpp>

namespace py = pybind11;

template <class Tensor, class... Tensors>
void bind_constructors(py::class_<Jet::PathInfo> &c)
{
    c.def(py::init<const Jet::TensorNetwork<Tensor> &,
                   const Jet::PathInfo::path_t &>(),
          py::arg("tn"), py::arg("path") = Jet::PathInfo::path_t(), R"(
            Constructs a populated PathInfo for the given path
            through a tensor network

            Args:
                tn: Tensor network
                path: Pairs of integers representing a contraction path through
                      the tensor network
            )");

    if constexpr (sizeof...(Tensors) > 0) {
        bind_constructors<Tensors...>(c);
    }
}

template <class... Tensors> void AddBindingsForPathInfo(py::module_ &m)
{

    py::class_<Jet::PathStepInfo>(m, "PathStepInfo", R"(
        PathStepInfo represents the contraction metadata associated
        with a node in a TensorNetwork)")

        // Constants
        // ----------------------------------------------------------------
        .def_property_readonly("MISSING_ID",
                               []() { return Jet::PathStepInfo::MISSING_ID; })

        // Instance variables
        // ----------------------------------------------------------------
        .def_readonly("id", &Jet::PathStepInfo::id)
        .def_readonly("parent", &Jet::PathStepInfo::parent)
        .def_readonly("children", &Jet::PathStepInfo::children)
        .def_readonly("name", &Jet::PathStepInfo::name)
        .def_readonly("node_indices", &Jet::PathStepInfo::node_indices)
        .def_readonly("tensor_indices", &Jet::PathStepInfo::tensor_indices)
        .def_readonly("tags", &Jet::PathStepInfo::tags)
        .def_readonly("contracted_indices",
                      &Jet::PathStepInfo::contracted_indices);

    auto cls =
        py::class_<Jet::PathInfo>(
            m, "PathInfo",
            "PathInfo represents a contraction path in a tensor network.")

            // Constructors
            //-----------------------------------------------------------------

            .def(py::init<>(), R"(Constructs an empty path)")

            // Properties
            // ----------------------------------------------------------------
            .def_property_readonly("index_to_size_map",
                                   &Jet::PathInfo::GetIndexSizes, R"(
            A map of indices to their dimension)")

            .def_property_readonly("num_leaves", &Jet::PathInfo::GetNumLeaves,
                                   R"(
            Number of leaf steps in this path)")

            .def_property_readonly("path", &Jet::PathInfo::GetPath, R"(
            The contraction path)")

            .def_property_readonly("steps", &Jet::PathInfo::GetSteps, R"(
            The steps of this path)")

            // Other
            // -----------------------------------------------------------------
            .def("total_flops", &Jet::PathInfo::GetTotalFlops, R"(
            Computes the total number of floating-point operations needed
            to contract the tensor network along this path)")

            .def("total_memory", &Jet::PathInfo::GetTotalMemory, R"(
            Computes the total memory required to contract the tensor
            network along this path.)");

    // Add constructor bindings for each Tensor type
    bind_constructors<Tensors...>(cls);
}