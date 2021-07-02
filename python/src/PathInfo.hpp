#pragma once

#include <string>

#include <pybind11/pybind11.h>

#include <Jet.hpp>

namespace py = pybind11;

template <class T, class... Ts>
void bind_constructors(py::class_<Jet::PathInfo> &c)
{
    c.def(py::init<const Jet::TensorNetwork<Jet::Tensor<T>> &,
                   const Jet::PathInfo::Path &>(),
          py::arg("tn"), py::arg("path") = Jet::PathInfo::Path(),
          R"(
Constructs a populated PathInfo for the given path through a tensor network.
          )");

    if constexpr (sizeof...(Ts) > 0) {
        bind_constructors<Ts...>(c);
    }
}

/**
 * @brief Adds Python bindings for the include/jet/PathInfo.hpp file.
 *
 * @tparam T Template parameter of the `Tensor` class.
 * @param m Jet pybind11 module.
 */
template <class... Ts> void AddBindingsForPathInfo(py::module_ &m)
{
    py::class_<Jet::PathStepInfo>(m, "PathStepInfo", R"(
PathStepInfo represents the contraction metadata associated with a node in a
``TensorNetwork``.
        )")

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
        py::class_<Jet::PathInfo>(m, "PathInfo",
                                  R"(
PathInfo represents a contraction path in a tensor network.

Args:
    tn (TensorNetwork): Tensor network associated with the contraction path.
    path (List[Tuple[int, int]]): Pairs of integers representing a contraction
        path through the tensor network.
            )")

            // Constructors
            //-----------------------------------------------------------------

            .def(py::init<>(), "Constructs an empty path")

            // Properties
            // ----------------------------------------------------------------

            .def_property_readonly("index_to_size_map",
                                   &Jet::PathInfo::GetIndexSizes,
                                   "Mapping from indices to their dimension.")

            .def_property_readonly("num_leaves", &Jet::PathInfo::GetNumLeaves,
                                   "Number of leaf steps in this path")

            .def_property_readonly(
                "path", &Jet::PathInfo::GetPath,
                "List of node ID pairs representing the contraction path.")

            .def_property_readonly("steps", &Jet::PathInfo::GetSteps,
                                   "Steps of this path.")

            // Other
            // -----------------------------------------------------------------

            .def("total_flops", &Jet::PathInfo::GetTotalFlops, R"(
Computes the total number of floating-point operations needed to contract the
tensor network along this path

Returns:
    float: Total number of floating-point operations.
            )")

            .def("total_memory", &Jet::PathInfo::GetTotalMemory, R"(
Computes the total memory required to contract the tensor network along this path.

Returns:
    float: Total number of elements in the contracted tensors.
            )");

    // Add constructor bindings for each Tensor data type.
    bind_constructors<Ts...>(cls);
}
