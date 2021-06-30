#include <pybind11/pybind11.h>

#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the include/jet/Version.hpp file.
 *
 * @param m Jet pybind11 module.
 */
void AddBindingsForVersion(py::module_ &m)
{
    m.attr("__version__") = Jet::Version();

    m.def("version", Jet::Version, R"(
        Returns the current Jet version.

        Returns:
            str: The Jet version.
    )");
}