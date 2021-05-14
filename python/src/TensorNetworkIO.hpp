#include <optional>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Jet.hpp>

namespace py = pybind11;

namespace {

/**
 * @brief Adds Python bindings for the TensorNetworkFile class.
 *
 * @tparam Tensor Type of the tensor in the tensor network file.
 * @param m Jet pybind11 module.
 * @param name Name of the TensorNetworkFile class binding.
 */
template <class Tensor>
void AddBindingsForTensorNetworkFile(py::module_ &m, const char *name)
{
    using TensorNetworkFile = Jet::TensorNetworkFile<Tensor>;

    py::class_<TensorNetworkFile>(
        m, name, "This class represents the contents of a tensor network file.")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<const std::optional<Jet::PathInfo> &,
                      const Jet::TensorNetwork<Tensor> &>(),
             py::arg("path") = std::nullopt,
             py::arg("tensors") = Jet::TensorNetwork<Tensor>())

        // Properties
        // ---------------------------------------------------------------------

        .def_readwrite("path", &TensorNetworkFile::path, "Contraction path.")

        .def_readwrite("tensors", &TensorNetworkFile::tensors,
                       "Tensor network.");
}

/**
 * @brief Adds Python bindings for the TensorNetworkSerializer class.
 *
 * @tparam Tensor Type of the tensor in the tensor network file.
 * @param m Jet pybind11 module.
 * @param name Name of the TensorNetworkSerializer class binding.
 */
template <class Tensor>
void AddBindingsForTensorNetworkSerializer(py::module_ &m, const char *name)
{
    using TensorNetworkSerializer = Jet::TensorNetworkSerializer<Tensor>;

    py::class_<TensorNetworkSerializer>(m, name, R"(
        This class is a functor for serializing and deserializing a tensor
        network (and, optionally, a path) to/from a JSON document.

        If called with a tensor network (and, optionally, a path), the tensor
        network serializer will return a JSON string representing the given
        arguments.

        If called with a string, the tensor network serializer will parse the
        string as a JSON object and return a tensor network file instance. The
        object must contain a "tensors" key containing an array of tensors. It
        may optionally contain a "path" key describing a contraction path.

        Each element of the "tensors" array is an array with four elements
        containing the tags, ordered indices, shape, and values of each
        tensor, respectively.  Each value is an array of size two, representing
        the real and imaginary components of a complex number.  For example,

        ```
        [
            ["A", "hermitian"],
            ["a", "b"],
            [2, 2],
            [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]]
        ]
        ```

        corresponds to the 2x2 matrix

        ```
        [[1, i], [-i, 1]]
        ```

        where "a" is the row index and "b" is the column index.

        The "path" value is a list of integer pairs such as `[i, j]` where `i`
        and `j` are the indexes of two tensors in the `"tensors"` array (or the
        index of an intermediate tensor). 
    )")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<int>(), py::arg("indent") = -1, R"(
            Constructs a tensor network serializer with an indentation level.

            Args:
                indent: indentation level to use when serializing to JSON.  The
                        default value (-1) uses the most compact representation.
        )")

        // Special methods
        // ---------------------------------------------------------------------

        .def(
            "__call__",
            [](TensorNetworkSerializer &serializer,
               const Jet::TensorNetwork<Tensor> &tn) { return serializer(tn); },
            py::arg("tn"),
            R"(
            Dumps the given tensor network to a JSON string.

            Args:
                tn: tensor network to serialize.

            Returns:
                Serialized tensor network file representing the given tensor
                network.
        )")

        .def(
            "__call__",
            [](TensorNetworkSerializer &serializer,
               const Jet::TensorNetwork<Tensor> &tn,
               const Jet::PathInfo &path) { return serializer(tn, path); },
            py::arg("tn"), py::arg("path"),
            R"(
            Dumps the given tensor network and path to a JSON string.

            Args:
                tn: tensor network to serialize.
                path: contraction path to serialize.

            Returns:
                Serialized tensor network file representing the given tensor
                network and contraction path.
        )")

        .def(
            "__call__",
            [](TensorNetworkSerializer &serializer, const std::string &str) {
                return serializer(str);
            },
            py::arg("str"),
            R"(
            Loads a tensor network file from a JSON string.

            Args:
                str: tensor network file serialized as a JSON string.

            Returns:
                Deserialized tensor network file representing a tensor network
                (and contraction path, if specified).
        )");
}

} // namespace

/**
 * @brief Adds Python bindings for the include/jet/TensorNetworkIO.hpp file.
 *
 * @tparam Tensor Type of the tensor in the TensorNetwork.
 * @param m Jet pybind11 module.
 * @param tnf_name Name of the TensorNetworkFile class binding.
 * @param tns_name Name of the TensorNetworkSerializer class binding.
 */
template <class Tensor>
void AddBindingsForTensorNetworkIO(py::module_ &m, const char *tnf_name,
                                   const char *tns_name)
{
    AddBindingsForTensorNetworkFile<Tensor>(m, tnf_name);
    AddBindingsForTensorNetworkSerializer<Tensor>(m, tns_name);
}