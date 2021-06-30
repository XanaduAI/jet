#include <complex>
#include <sstream>
#include <string>

#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Type.hpp"
#include <Jet.hpp>

namespace py = pybind11;

/**
 * @brief Adds Python bindings for the include/jet/Tensor.hpp file.
 *
 * @tparam T Template parameter of the `Tensor` class.
 * @param m Jet pybind11 module.
 */
template <class T> void AddBindingsForTensor(py::module_ &m)
{
    using tensor_t = Jet::Tensor<T>;

    const std::string class_name = "Tensor" + Type<T>::suffix;

    py::class_<tensor_t>(m, class_name.c_str(), R"(
        Tensor represents an :math:`n`-rank data structure of complex-valued data
        for tensor operations. We use the following conventions:

        - "Rank" and "order" are used interchangeably and refer to the number of
          tensor indices. In general, "rank" will be preferred.
        - "Dimension" refers to the number of elements along a tensor index.
        - "Shape" refers to the dimensions of a tensor; the number of dimensions
          is the rank of the tensor.

        Args:
            indices (Sequence[str]): Label of each tensor index.
            shape (Sequence[int]): Dimension of each tensor index.
            data (Sequence[complex]): Row-major encoded complex data representation.
        )")

        // Static properties
        // ---------------------------------------------------------------------

        .def_property_readonly_static(
            "dtype", [](const py::object &) { return Type<T>::dtype; },
            "Data type of this tensor.")

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
            the set ``?[a-zA-Z]``.

            Args:
                shape (Sequence[int]): Dimension of each tensor index.
        )")

        .def(py::init<const std::vector<std::string> &,
                      const std::vector<size_t> &>(),
             py::arg("indices"), py::arg("shape"), R"(
            Constructs a tensor object with the given shape, index labels, zero-
            initialized data values, and a size equal to the product of the
            shape dimensions.

            Args:
                indices (Sequence[str]): Label of each tensor index.
                shape (Sequence[int]): Dimension of each tensor index.
        )")

        .def(py::init<const std::vector<std::string> &,
                      const std::vector<size_t> &, const std::vector<T> &>(),
             py::arg("indices"), py::arg("shape"), py::arg("data"), R"(
            Constructs a tensor object with the given shape, index labels,
            data values, and a size equal to the product of the shape
            dimensions.

            Args:
                indices (Sequence[str]): Label of each tensor index.
                shape (Sequence[int]): Dimension of each tensor index.
                data (Sequence[complex]): Row-major encoded complex data representation.
        )")

        .def(py::init<const tensor_t &>(), py::arg("other"), R"(
            Constructs a copy of a tensor object.

            Args:
                other (Tensor): Tensor object to copy.
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
            py::arg("pos"),
            R"(
            Returns the tensor object data at the given local index. Supplying
            an index greater than or equal to the size of the tensor is
            undefined behaviour.

            Args:
                pos (int): Position of the datum to retrieve, encoded as a 1D
                    row-major index (lexicographic ordering).

            Returns:
                complex: Complex data value at the given index.
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
            Assigns random values to the tensor data. The real and imaginary
            components of each datum will be independently sampled from a
            uniform distribution with support over ``[-1, 1]``.
        )")

        .def("fill_random", py::overload_cast<size_t>(&tensor_t::FillRandom),
             py::arg("seed"), R"(
            Assigns random values to the tensor data. The real and imaginary
            components of each datum will be independently sampled from a
            uniform distribution with support over ``[-1, 1]``. This overload
            enables reproducible random number generation for a given seed.

            Args:
                seed (int): Seed to supply to the RNG engine.
        )")

        .def("init_indices_and_shape", &tensor_t::InitIndicesAndShape,
             py::arg("indices"), py::arg("shape"), R"(
            Initializes the indices and shape of a tensor object. The indices
            and shape must be ordered to map directly such that ``indices[i]``
            has size ``shape[i]``.

            Args:
                indices (Sequence[str]): Label of each tensor index.
                shape (Sequence[int]): Dimension of each tensor index.
        )")

        .def("get_value", &tensor_t::GetValue, py::arg("indices"), R"(
            Returns the tensor data value at the given :math:`n`-dimensional index.

            Args:
                indices (Sequence[str]): :math:`n`-dimensional tensor data index in row-major order.

            Returns:
                complex: Complex data value at the given index.
        )")

        .def("is_scalar", &tensor_t::IsScalar, R"(
            Reports whether the tensor is a scalar.

            Returns:
                bool: ``True`` if the tensor is of rank 0. Otherwise, ``False`` is returned.
        )")

        .def("rename_index", &tensor_t::RenameIndex, py::arg("pos"),
             py::arg("new_label"), R"(
            Renames the index label at the given position.

            Args:
                pos (int): Position of the label.
                new_label (str): New label.
        )")

        .def("set_value", &tensor_t::SetValue, py::arg("indices"),
             py::arg("value"), R"(
            Sets the tensor data value at the given :math:`n`-dimensional index.

            Args:
                indices (Sequence[str]): :math:`n`-dimensional tensor data index in row-major order.
                value (complex): Value to set at the data index.
        )")

        .def(
            "conj",
            [](const tensor_t &self) -> tensor_t { return self.Conj(); },
            R"(
            Returns the conjugate of the given tensor object.

            Returns:
                Tensor: Conjugate tensor object.
          )")

        .def(
            "add_tensor",
            [](const tensor_t &self, const tensor_t &other) -> tensor_t {
                return self.AddTensor(other);
            },
            py::arg("other"),
            R"(
            Adds two tensor objects with the same index sets. The resulting
            tensor will have the same indices as the first argument (i.e., ``A``).

            Args:
                A (Tensor): Tensor on the LHS of the addition.
                B (Tensor): Tensor on the RHS of the addition.

            Returns:
                Tensor: Element-wise sum of the tensors.

            **Example**

            Given a 2x3 tensor :math:`A(i,j)` and a 2x3 tensor :math:`B(i,j)`,
            the addition of :math:`A` and :math:`B` is another 2x3 tensor
            :math:`C(i,j)`:

            .. code-block:: python

                import jet

                A = jet.Tensor(["i", "j"], [2, 3])
                B = jet.Tensor(["i", "j"], [2, 3])

                A.fill_random()
                B.fill_random()

                C = A.add_tensor(B);
            )")

        .def(
            "contract_with_tensor",
            [](const tensor_t &self, const tensor_t &other) -> tensor_t {
                return self.ContractWithTensor(other);
            },
            py::arg("other"), R"(
            Contracts two tensor objects over the intersection of their index
            sets. The resulting tensor will be formed with indices given by the
            symmetric difference of the index sets.

            Args:
                A (Tensor): Tensor on the LHS of the contraction.
                B (Tensor): Tensor on the RHS of the contraction.

            Returns:
                Tensor: Contraction of the tensors.

            **Example**

            Given a 3x2x4 tensor :math:`A(i,j,k)` and a 2x4x2 tensor :math:`B(j,k,l)`,
            the common indices are :math:`\{j,k\}` and the symmetric difference of
            the sets is :math:`\{i,l\}`. The result of the contraction will be a
            tensor 3x2 tensor :math:`C(i,l)`.

            .. code-block:: python

                import jet

                A = jet.Tensor(["i", "j", "k"], [3, 2, 4])
                B = jet.Tensor(["j", "k", "l"], [2, 4, 2])

                A.fill_random()
                B.fill_random()

                C = A.contract_with_tensor(B);
          )")

        .def(
            "reshape",
            [](const tensor_t &self, const std::vector<size_t> &shape)
                -> tensor_t { return self.Reshape(shape); },
            py::arg("shape"), R"(
            Reshapes a tensor object to the given dimensions.

            Args:
                shape (Sequence[int]): Index dimensionality of the reshaped tensor object.
            
            Returns:
                Tensor: Reshaped copy of the given tensor object.
          )")

        .def(
            "slice_index",
            [](const tensor_t &self, const std::string &index,
               const size_t value) -> tensor_t {
                return self.SliceIndex(index, value);
            },
            py::arg("index"), py::arg("value"), R"(
            Slices a tensor object index. The result is a tensor object whose
            indices and data are a subset of the provided tensor object, sliced
            along the given index argument.

            Args:
                index (str): Index label on which to slice.
                value (int): Value to slice the index on.
            
            Returns:
                Tensor: Slice of the tensor object.

            **Example**

            Suppose that :math:`A(i,j)` is a 2x3 tensor. Then,

            .. code-block:: python

                import jet

                A = jet.Tensor({"i", "j"}, {2, 3})
                A.fill_random()

                A.slice_index("i", 0) # Result is a 1x3 tensor
                A.slice_index("i", 1) # Result is a 1x3 tensor

                A.slice_index("j", 0) # Result is a 2x1 tensor
                A.slice_index("j", 1) # Result is a 2x1 tensor
                A.slice_index("j", 2) # Result is a 2x1 tensor
          )")

        .def(
            "transpose",
            [](const tensor_t &self,
               const std::vector<std::string> &new_indices) -> tensor_t {
                return self.Transpose(new_indices);
            },
            py::arg("new_indices"), R"(
            Transposes the indices of a tensor object.

            Args:
                indices (Sequence[str]): Desired index ordering, specified as a list of labels.

            Returns:
                Tensor: Transposed tensor object.
          )")

        .def(
            "transpose",
            [](const tensor_t &self, const std::vector<size_t> &new_ordering)
                -> tensor_t { return self.Transpose(new_ordering); },
            py::arg("new_ordering"), R"(
            Transposes the indices of a tensor object.

            Args:
                ordering (Sequence[int]): Desired index ordering, specified as a permutation.

            Returns:
                Tensor: Transposed tensor object.
          )");

    // Static methods as module free-functions
    // ---------------------------------------------------------------------

    m.def(
        "conj",
        [](const tensor_t &tensor) -> tensor_t {
            return Jet::Tensor<>::Conj<T>(tensor);
        },
        py::arg("tensor"), R"(
        Returns the conjugate of the given tensor object.

        Args:
            tensor (Tensor): Reference tensor object.

        Returns:
            Tensor: Conjugate tensor object.
        )");

    m.def(
        "add_tensors",
        [](const tensor_t &tensor_a, const tensor_t &tensor_b) -> tensor_t {
            return Jet::Tensor<>::AddTensors<T>(tensor_a, tensor_b);
        },
        py::arg("tensor_a"), py::arg("tensor_b"),
        R"(
        Adds two tensor objects with the same index sets. The resulting
        tensor will have the same indices as the first argument (i.e., ``A``).

        Args:
            A (Tensor): Tensor on the LHS of the addition.
            B (Tensor): Tensor on the RHS of the addition.

        Returns:
            Tensor: Element-wise sum of the tensors.

        **Example**

        Given a 2x3 tensor :math:`A(i,j)` and a 2x3 tensor :math:`B(i,j)`, the
        addition of :math:`A` and :math:`B` is another 2x3 tensor :math:`C(i,j)`:

        .. code-block:: python

            import jet

            A = jet.Tensor(["i", "j"], [2, 3])
            B = jet.Tensor(["i", "j"], [2, 3])

            A.fill_random()
            B.fill_random()

            C = jet.add_tensors(A, B);
        )");

    m.def(
        "contract_tensors",
        [](const tensor_t &tensor_l, const tensor_t &tensor_r) -> tensor_t {
            return Jet::Tensor<>::ContractTensors<T>(tensor_l, tensor_r);
        },
        py::arg("tensor_l"), py::arg("tensor_r"), R"(
        Contracts two tensor objects over the intersection of their index
        sets. The resulting tensor will be formed with indices given by the
        symmetric difference of the index sets.

        Args:
            A (Tensor): Tensor on the LHS of the contraction.
            B (Tensor): Tensor on the RHS of the contraction.

        Returns:
            Tensor: Contraction of the tensors.

        **Example**

        Given a 3x2x4 tensor :math:`A(i,j,k)` and a 2x4x2 tensor :math:`B(j,k,l)`,
        the common indices are :math:`\{j,k\}` and the symmetric difference of the
        sets is :math:`\{i,l\}`. The result of the contraction will be a tensor 3x2
        tensor :math:`C(i,l)`.

        .. code-block python:

            import jet

            A = jet.Tensor(["i", "j", "k"], [3, 2, 4])
            B = jet.Tensor(["j", "k", "l"], [2, 4, 2])

            A.fill_random()
            B.fill_random()

            C = jet.contract_tensors(A, B);
        )");

    m.def(
        "reshape",
        [](const tensor_t &tensor, const std::vector<size_t> &shape)
            -> tensor_t { return Jet::Tensor<>::Reshape<T>(tensor, shape); },
        py::arg("tensor"), py::arg("shape"), R"(
            Reshapes a tensor object to the given dimensions.

            Args:
                tensor (Tensor): Tensor object to reshape.
                shape (Sequence[int]): Index dimensionality of the reshaped tensor object.
            
            Returns:
                Tensor: Reshaped copy of the given tensor object.
          )");

    m.def(
        "slice_index",
        [](const tensor_t &tensor, const std::string &index,
           const size_t value) -> tensor_t {
            return Jet::Tensor<>::SliceIndex<T>(tensor, index, value);
        },
        py::arg("tensor"), py::arg("index"), py::arg("value"), R"(
            Slices a tensor object index. The result is a tensor object whose
            indices and data are a subset of the provided tensor object, sliced
            along the given index argument.

            Args:
                tensor (Tensor): Reference tensor object.
                index (str): Index label on which to slice.
                value (int): Value to slice the index on.
            
            Returns:
                Tensor: Slice of the tensor object.

            **Example**

            Suppose that :math:`A(i,j)` is a 2x3 tensor. Then,

            .. code-block:: python

                import jet

                A = jet.Tensor({"i", "j"}, {2, 3})
                A.fill_random()

                jet.slice_index(A, "i", 0) # Result is a 1x3 tensor
                jet.slice_index(A, "i", 1) # Result is a 1x3 tensor

                jet.slice_index(A, "j", 0) # Result is a 2x1 tensor
                jet.slice_index(A, "j", 1) # Result is a 2x1 tensor
                jet.slice_index(A, "j", 2) # Result is a 2x1 tensor
          )");

    m.def(
        "transpose",
        [](const tensor_t &tensor,
           const std::vector<std::string> &new_indices) -> tensor_t {
            return Jet::Tensor<>::Transpose<T>(tensor, new_indices);
        },
        py::arg("tensor"), py::arg("new_indices"), R"(
        Transposes the indices of a tensor object.

            Args:
                tensor (Tensor): Reference tensor object.
                indices (Sequence[str]): Desired index ordering, specified as a list of labels.

            Returns:
                Tensor: Transposed tensor object.
          )");

    m.def(
        "transpose",
        [](const tensor_t &tensor,
           const std::vector<size_t> &new_ordering) -> tensor_t {
            return Jet::Tensor<>::Transpose<T>(tensor, new_ordering);
        },
        py::arg("tensor"), py::arg("new_ordering"), R"(
            Transposes the indices of a tensor object.

            Args:
                tensor (Tensor): Reference tensor object.
                ordering (Sequence[int]): Desired index ordering, specified as a permutation.

            Returns:
                Tensor: Transposed tensor object.
          )");
}
