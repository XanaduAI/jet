import numpy as np
import pytest

from jet import Gate


class TestGate:
    @pytest.fixture
    def gate(self):
        """Returns a generic gate with two wires and one parameter."""
        return Gate(name="G", num_wires=2, params=[1])

    def test_tensor_indices_not_set(self, gate):
        """Tests that the correct tensor is returned for a gate with unset indices."""
        gate._data = lambda: np.arange(16)
        tensor = gate.tensor()

        assert tensor.indices == ["0", "1", "2", "3"]
        assert tensor.shape == [2, 2, 2, 2]
        assert tensor.data == list(range(16))

    def test_tensor_indices_are_set(self, gate):
        """Tests that the correct tensor is returned for a gate with specified indices."""
        gate._data = lambda: np.arange(3 ** 4)
        gate.indices = ["r", "g", "b", "a"]
        tensor = gate.tensor()

        assert tensor.indices == ["r", "g", "b", "a"]
        assert tensor.shape == [3, 3, 3, 3]
        assert tensor.data == list(range(3 ** 4))

    def test_tensor_adjoint(self, gate):
        """Tests that the correct adjoint tensor is returned for a gate."""
        gate._data = lambda: np.array([1 + 2j])
        tensor = gate.tensor(adjoint=True)

        assert tensor.indices == ["0", "1", "2", "3"]
        assert tensor.shape == [1, 1, 1, 1]
        assert tensor.data == [1 - 2j]

    def test_data(self, gate):
        """Tests that a NotImplementedError is raised when the data of a generic
        gate is retrieved.
        """
        with pytest.raises(NotImplementedError):
            gate._data()

    def test_validate_wrong_number_of_parameters(self, gate):
        """Tests that a ValueError is raised when a gate with the wrong number
        of parameters is validated.
        """
        with pytest.raises(ValueError):
            gate._validate(want_num_params=2)

    def test_validate_correct_number_of_parameters(self, gate):
        """Tests that no exceptions are raised when a gate with the correct
        number of parameters is validated.
        """
        gate._validate(want_num_params=1)

    @pytest.mark.parametrize("indices", [1, ["i", 2, "k"], ["x", "x"], []])
    def test_indices_are_invalid(self, gate, indices):
        """Tests that a ValueError is raised when the indices of a gate are set
        to an invalid value.
        """
        with pytest.raises(ValueError):
            gate.indices = indices

    @pytest.mark.parametrize("indices", [None, ["1", "2", "3", "4"]])
    def test_indices_are_valid(self, gate, indices):
        """Tests that the indices of a gate can be set and retrieved."""
        assert gate.indices is None
        gate.indices = indices
        assert gate.indices == indices
