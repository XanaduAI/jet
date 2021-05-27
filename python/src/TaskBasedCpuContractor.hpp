#include <sstream>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the `TaskBasedCpuContractor` class.
 *
 * @note Functions that return objects from the taskflow library are not bound.
 *
 * @tparam Tensor Template parameter of the `TaskBasedCpuContractor` class.
 * @param m Jet pybind11 module.
 * @param name Name of the `TaskBasedCpuContractor` class binding.
 */
template <class T>
void AddBindingsForTaskBasedCpuContractor(py::module_ &m, const char *name)
{
    using TaskBasedCpuContractor = Jet::TaskBasedCpuContractor<T>;

    py::class_<TaskBasedCpuContractor>(m, name, R"(
        This class is a tensor network contractor that contracts tensors
        concurrently on the CPU using a task-based scheduler.
    )")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<>(), "Constructs a new task-based CPU contractor.")

        // Properties
        // ---------------------------------------------------------------------

        .def_property_readonly(
            "name_to_tensor_map",
            [](const TaskBasedCpuContractor &tbcc) {
                std::unordered_map<std::string, T *> name_to_tensor_map;
                for (const auto &[name, ptr] : tbcc.GetNameToTensorMap()) {
                    name_to_tensor_map.emplace(name, ptr.get());
                }
                return name_to_tensor_map;
            },
            "Dictionary which maps names to tensors.")

        .def_property_readonly(
            "name_to_parents_map", &TaskBasedCpuContractor::GetNameToParentsMap,
            "Dictionary which maps names to lists of parent node IDs.")

        .def_property_readonly("results", &TaskBasedCpuContractor::GetResults,
                               "List of tensor results.")

        .def_property_readonly("reduction_result",
                               &TaskBasedCpuContractor::GetReductionResult,
                               "Tensor at the end of the reduction task.")

        .def_property_readonly("flops", &TaskBasedCpuContractor::GetFlops, R"(
            Number of floating-point additions and multiplications required
            to implement the contraction tasks.
        )")

        .def_property_readonly("memory", &TaskBasedCpuContractor::GetMemory,
                               "Number of elements in the non-leaf tensors.")

        // Other
        // ---------------------------------------------------------------------

        .def("add_contraction_tasks",
             &TaskBasedCpuContractor::AddContractionTasks, py::arg("tn"),
             py::arg("path_info"), R"(
            Adds contraction tasks for a tensor network.

            Args:
                tn: tensor network to be contracted.
                path_info: contraction path through the tensor network.

            Returns:
                Number of contraction tasks shared with previous calls to this
                method.
        )")

        .def("add_reduction_task", &TaskBasedCpuContractor::AddReductionTask,
             R"(
            Adds a reduction task to sum the result tensors.

            Returns:
                Number of created reduction tasks.
        )")

        .def("add_deletion_tasks", &TaskBasedCpuContractor::AddDeletionTasks,
             R"(
            Adds deletion tasks for intermediate tensors, deallocating each one
            when it is no longer needed.

            Returns:
                Number of created deletion tasks.
        )")

        .def(
            "contract",
            [](TaskBasedCpuContractor &tbcc) { tbcc.Contract().wait(); }, R"(
            Executes the tasks in this task-based CPU contractor.

            Warning:
                This is a blocking call.
        )");
}