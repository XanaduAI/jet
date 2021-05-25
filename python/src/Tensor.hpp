#include <complex>
#include <sstream>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the include/jet/Tensor.hpp file.
 *
 * @tparam T Template parameter for Jet::Tensor.
 * @param m Jet pybind11 module.
 * @param name Name of the Tensor class binding.
 */
template <class T> void AddBindingsForTensor(py::module_ &m, const char *name)
{
    using tensor_t = Jet::Tensor<T>;

    py::class_<tensor_t>(m, name, R"(
        This class represents an n-rank data structure of complex-valued data
        for tensor operations. We use the following conventions:

            - "Rank" and "order" are used interchangeably and refer to the
              number of tensor indices.  In general, "rank" will be preferred.
            - "Dimension" refers to the number of elements along a tensor index.
            - "Shape" refers to the dimensions of a tensor; the number of
              dimensions is the rank of the tensor.
    )")
        // Static functions
        // ---------------------------------------------------------------------

        .def_static("add_tensors", &tensor_t::AddTensors, py::arg("A"),
                    py::arg("B"), "Alias for add_tensors().")

        .def_static("contract_tensors", &tensor_t::ContractTensors,
                    py::arg("A"), py::arg("B"), "Alias for contract_tensors().")

        .def_static("reshape", &tensor_t::Reshape, py::arg("tensor"),
                    py::arg("shape"), "Alias for reshape().")

        .def_static("slice_index", &tensor_t::SliceIndex, py::arg("tensor"),
                    py::arg("index"), py::arg("value"),
                    "Alias for slice_index().")

        // Constructors
        // ---------------------------------------------------------------------

        .def(py::init<>(), R"(
            Constructs a tensor object with a single zero-initialized data
            value. The shape and indices of this tensor object are not set.
        )")

        .def(py::init<const std::vector<size_t> &>(), py::arg("shape"), R"(
            Constructs a tensor object with the given shape, zero-initialized
            data values, and a size equal to the product of the shape
            dimensions. The indices of this tensor object default to values from
            the set `?[a-zA-Z]`.

            Args:
                shape: dimension of each tensor index.
        )")

        .def(py::init<const std::vector<std::string> &,
                      const std::vector<size_t> &>(),
             py::arg("indices"), py::arg("shape"), R"(
            Constructs a tensor object with the given shape, index labels, zero-
            initialized data values, and a size equal to the product of the
            shape dimensions.

            Args:
                indices: label of each tensor index.
                shape: dimension of each tensor index.
        )")

        .def(py::init<const std::vector<std::string> &,
                      const std::vector<size_t> &, const std::vector<T> &>(),
             py::arg("indices"), py::arg("shape"), py::arg("data"), R"(
            Constructs a tensor object with the given shape, index labels,
            data values, and a size equal to the product of the shape
            dimensions.

            Args:
                indices: label of each tensor index.
                shape: dimension of each tensor index.
                data: row-major encoded complex data representation.
        )")

        .def(py::init<const tensor_t &>(), py::arg("other"), R"(
            Constructs a copy of a tensor object.

            Args:
                other: tensor object to copy.
        )")

        // Properties
        // ---------------------------------------------------------------------

        .def_property("shape", &tensor_t::GetShape, &tensor_t::SetShape,
                      "List containing the dimension of each tensor index.")

        .def_property_readonly("index_to_dimension_map",
                               &tensor_t::GetIndexToDimension,
                               "Mapping from index labels to dimension sizes.")

        .def_property_readonly("data", &tensor_t::GetData,
                               "Complex data values in row-major order.")

        .def_property_readonly("indices", &tensor_t::GetIndices,
                               "List of index labels.")

        .def_property_readonly("scalar", &tensor_t::GetScalar,
                               "First data value of the tensor.")

        // Special methods
        // ---------------------------------------------------------------------

        .def(
            "__getitem__",
            [](const tensor_t &tensor, size_t pos) { return tensor[pos]; },
            py::arg("pos"), R"(
            Returns the tensor object data at the given local index. Supplying
            an index greater than or equal to the size of the tensor is
            undefined behaviour.

            Args:
                pos: position of the datum to retrieve, encoded as a 1D row-
                     major index (lexicographic ordering).

            Returns:
                Complex data value at the given index.
        )")

        .def("__len__", &tensor_t::GetSize,
             "Returns the number of data elements in the tensor.")

        .def(
            "__repr__",
            [](const tensor_t &tensor) {
                std::stringstream stream;
                stream << tensor;
                return stream.str();
            },
            "Returns a string representation of the tensor.")

        .def(py::self == py::self, py::arg("other"),
             "Reports whether two tensor objects are equivalent.")

        .def(py::self != py::self, py::arg("other"),
             "Reports whether two tensor objects are different.")

        // Other
        // ---------------------------------------------------------------------

        .def("fill_random", py::overload_cast<>(&tensor_t::FillRandom), R"(
            Assigns random values to the tensor data.  The real and imaginary
            components of each datum will be independently sampled from a
            uniform distribution with support over [-1, 1].
        )")

        .def("fill_random", py::overload_cast<size_t>(&tensor_t::FillRandom),
             py::arg("seed"), R"(
            Assigns random values to the tensor data.  The real and imaginary
            components of each datum will be independently sampled from a
            uniform distribution with support over [-1, 1].  This overload
            enables reproducible random number generation for a given seed.

            Args:
                seed: seed to supply to the RNG engine.
        )")

        .def("init_indices_and_shape", &tensor_t::InitIndicesAndShape,
             py::arg("indices"), py::arg("shape"), R"(
            Initializes the indices and shape of a tensor object. The indices
            and shape must be ordered to map directly such that `indices[i]` has
            size `shape[i]`.

            Args:
                indices: label of each tensor index.
                shape: dimension of each tensor index.
        )")

        .def("get_value", &tensor_t::GetValue, py::arg("indices"), R"(
            Returns the tensor data value at the given n-dimensional index.

            Args:
                indices: n-dimensional tensor data index in row-major order.

            Returns:
                Complex data value at the given index.
        )")

        .def("is_scalar", &tensor_t::IsScalar, R"(
            Reports whether the tensor is a scalar.

            Returns:
                True if the tensor is of rank 0.  Otherwise, False is returned.
        )")

        .def("rename_index", &tensor_t::RenameIndex, py::arg("pos"),
             py::arg("new_label"), R"(
            Renames the index label at the given position.

            Args:
                pos: Position of the label.
                new_label: New label.
        )")

        .def("set_value", &tensor_t::SetValue, py::arg("indices"),
             py::arg("value"), R"(
            Sets the tensor data value at the given n-dimensional index.

            Args:
                indices: n-dimensional tensor data index in row-major order.
                value: value to set at the data index.
        )");

    // Free functions
    // -------------------------------------------------------------------------

    m.def("conj", Jet::Conj<T>, py::arg("tensor"), R"(
            Returns the conjugate of the given tensor object.

            Args:
                tensor: reference tensor object.

            Returns:
                Conjugate of the given tensor object.
          )");

    m.def("add_tensors", Jet::AddTensors<T>, py::arg("A"), py::arg("B"),
          R"(
            Adds two tensor objects with the same index sets. The resulting
            tensor will have the same indices as the first argument (i.e., `A`).

            Example:
                Given a 2x3 tensor A(i,j) and a 2x3 tensor B(i,j), the addition
                of A and B is another 2x3 tensor C(i,j):

                    import jet

                    A = jet.Tensor64(["i", "j"], [2, 3])
                    B = jet.Tensor64(["i", "j"], [2, 3])

                    A.fill_random()
                    B.fill_random()

                    C = jet.add_tensors(A, B);

            Args:
                A: tensor on the LHS of the addition.
                B: tensor on the RHS of the addition.

            Returns:
                Tensor object representing the element-wise sum of the tensors.
          )");

    m.def("contract_tensors", Jet::ContractTensors<T>, py::arg("A"),
          py::arg("B"), R"(
            Contracts two tensor objects over the intersection of their index
            sets. The resulting tensor will be formed with indices given by the
            symmetric difference of the index sets.

            Example:
                Given a 3x2x4 tensor A(i,j,k) and a 2x4x2 tensor B(j,k,l), the
                common indices are {j,k} and the symmetric difference of the
                sets is {i,l}. The result of the contraction will be a tensor
                3x2 tensor C(i,l).

                    import jet

                    A = jet.Tensor64(["i", "j", "k"], [3, 2, 4])
                    B = jet.Tensor64(["j", "k", "l"], [2, 4, 2])

                    A.fill_random()
                    B.fill_random()

                    C = jet.contract_tensors(A, B);

            Args:
                A: tensor on the LHS of the contraction.
                B: tensor on the RHS of the contraction.

            Returns:
                Tensor object representing the contraction of the tensors.
          )");

    m.def("reshape", Jet::Reshape<T>, py::arg("tensor"), py::arg("shape"), R"(
            Reshapes a tensor object to the given dimensions.

            Args:
                tensor: tensor object to reshape.
                shape: index dimensionality of the reshaped tensor object.
            
            Returns:
                Reshaped copy of the given tensor object.
          )");

    m.def("slice_index", Jet::SliceIndex<T>, py::arg("tensor"),
          py::arg("index"), py::arg("value"), R"(
            Slices a tensor object index. The result is a tensor object whose
            indices and data are a subset of the provided tensor object, sliced
            along the given index argument.

            Example:
                Suppose that A(i,j) is a 2x3 tensor.  Then,

                    import jet

                    A = jet.Tensor64({"i", "j"}, {2, 3})
                    A.fill_random()

                    jet.slice_index(A, "i", 0) # Result is a 1x3 tensor
                    jet.slice_index(A, "i", 1) # Result is a 1x3 tensor

                    jet.slice_index(A, "j", 0) # Result is a 2x1 tensor
                    jet.slice_index(A, "j", 1) # Result is a 2x1 tensor
                    jet.slice_index(A, "j", 2) # Result is a 2x1 tensor

            Args:
                tensor: tensor object to slice.
                index: index label on which to slice.
                value: value to slice the index on.
            
            Returns:
                Slice of the tensor object.
          )");

    m.def("transpose",
          py::overload_cast<const tensor_t &, const std::vector<std::string> &>(
              Jet::Transpose<T>),
          py::arg("tensor"), py::arg("indices"), R"(
            Transposes the indices of a tensor object.

            Args:
                tensor: reference tensor object.
                indices: desired index ordering, specified as a list of labels.

            Returns:
                Transposed tensor object.
          )");

    m.def("transpose",
          py::overload_cast<const tensor_t &, const std::vector<size_t> &>(
              Jet::Transpose<T>),
          py::arg("tensor"), py::arg("ordering"), R"(
            Transposes the indices of a tensor object.

            Args:
                tensor: reference tensor object.
                ordering: desired index ordering, specified as a permutation.

            Returns:
                Transposed tensor object.
          )");
}