#include <sstream>
#include <string>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Type.hpp"
#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the `TaskBasedContractor` class.
 *
 * @note Functions that return objects from the taskflow library are not bound.
 *
 * @tparam T Template parameter of the `Tensor` class.
 * @param m Jet pybind11 module.
 */
template <class T> void AddBindingsForTaskBasedContractor(py::module_ &m)
{
    using TaskBasedContractor = Jet::TaskBasedContractor<Jet::Tensor<T>>;

    const std::string class_name = "TaskBasedContractor" + Type<T>::suffix;

    py::class_<TaskBasedContractor>(m, class_name.c_str(), R"(
TaskBasedContractor represents a tensor network contractor that contracts
tensors concurrently using a task-based scheduler.
    )")

        // Static properties
        // ---------------------------------------------------------------------

        .def_property_readonly_static(
            "dtype", [](const py::object &) { return Type<T>::dtype; },
            "Data type of this task-based contractor.")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<>(), "Constructs a new task-based contractor.")

        // Properties
        // ---------------------------------------------------------------------

        .def_property_readonly(
            "name_to_tensor_map",
            [](const TaskBasedContractor &tbc) {
                std::unordered_map<std::string, Jet::Tensor<T> *>
                    name_to_tensor_map;
                for (const auto &[name, ptr] : tbc.GetNameToTensorMap()) {
                    name_to_tensor_map.emplace(name, ptr.get());
                }
                return name_to_tensor_map;
            },
            "Mapping from names to tensors.")

        .def_property_readonly(
            "name_to_parents_map", &TaskBasedContractor::GetNameToParentsMap,
            "Mapping from names to lists of parent node IDs.")

        .def_property_readonly("results", &TaskBasedContractor::GetResults,
                               "List of tensor results.")

        .def_property_readonly("reduction_result",
                               &TaskBasedContractor::GetReductionResult,
                               "Tensor at the end of the reduction task.")

        .def_property_readonly("flops", &TaskBasedContractor::GetFlops, R"(
Number of floating-point additions and multiplications required to implement the
contraction tasks.
        )")

        .def_property_readonly("memory", &TaskBasedContractor::GetMemory,
                               "Number of elements in the non-leaf tensors.")

        // Other
        // ---------------------------------------------------------------------

        .def("add_contraction_tasks", &TaskBasedContractor::AddContractionTasks,
             py::arg("tn"), py::arg("path_info"), R"(
Adds contraction tasks for a tensor network.

Args:
    tn (TensorNetwork): Tensor network to be contracted.
    path_info (.PathInfo): Contraction path through the tensor network.

Returns:
    int: Number of contraction tasks shared with previous calls to this method.
        )")

        .def("add_reduction_task", &TaskBasedContractor::AddReductionTask, R"(
Adds a reduction task to sum the result tensors.

Returns:
    int: Number of created reduction tasks.
        )")

        .def("add_deletion_tasks", &TaskBasedContractor::AddDeletionTasks, R"(
Adds deletion tasks for intermediate tensors, deallocating each one when it is
no longer needed.

Returns:
    int: Number of created deletion tasks.
        )")

        .def(
            "contract", [](TaskBasedContractor &tbc) { tbc.Contract().wait(); },
            R"(
Executes the tasks in this task-based contractor.

.. warning::

    This is a blocking call.
        )");
}