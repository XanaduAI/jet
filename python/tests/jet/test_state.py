import numpy as np
import pytest

import jet


class MockState(jet.State):
    """MockState represents an imaginary qutrit state."""

    def __init__(self):
        super().__init__(name="MockState", num_wires=1)

    def _data(self) -> np.ndarray:
        return np.array([1j, 0, 0])


class TestState:
    @pytest.fixture
    def state(self) -> MockState:
        """Returns a mock state instance."""
        return MockState()

    def test_tensor_indices_not_set(self, state):
        """Tests that the correct tensor is returned for a state with unset indices."""
        tensor = state.tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [3]
        assert tensor.data == [1j, 0, 0]

    def test_tensor_indices_are_set(self, state):
        """Tests that the correct tensor is returned for a state with specified indices."""
        state.indices = ["i"]
        tensor = state.tensor()
        assert tensor.indices == ["i"]
        assert tensor.shape == [3]
        assert tensor.data == [1j, 0, 0]

    def test_tensor_adjoint(self, state):
        """Tests that the correct adjoint tensor is returned for a state."""
        tensor = state.tensor(adjoint=True)
        assert tensor.indices == ["0"]
        assert tensor.shape == [3]
        assert tensor.data == [-1j, 0, 0]

    @pytest.mark.parametrize("indices", [1, [2], ["x", "x"], []])
    def test_indices_are_invalid(self, state, indices):
        """Tests that a ValueError is raised when the indices of a state are set
        to an invalid value.
        """
        with pytest.raises(ValueError):
            state.indices = indices

    @pytest.mark.parametrize("indices", [None, ["i"]])
    def test_indices_are_valid(self, state, indices):
        """Tests that the indices of a state can be set and retrieved."""
        assert state.indices is None
        state.indices = indices
        assert state.indices == indices

    def test_equal_states(self):
        """Tests that two equivalent states are equal."""
        assert MockState() == MockState()

    def test_unequal_states(self):
        """Tests that two different states are not equal."""
        state0 = MockState()
        state1 = MockState()
        state1._data = lambda: np.array([0, 0, 1])
        assert state0 != state1


class TestQubit:
    def test_default_constructor(self):
        """Tests that the default state vector of a qubit is the vacuum state."""
        tensor = jet.Qubit().tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [1, 0]

    def test_data_contructor(self):
        """Tests that the state vector of a qubit can be manually specified."""
        tensor = jet.Qubit(data=np.array([0, 1j])).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [0, 1j]


class TestQudit:
    def test_default_constructor(self):
        """Tests that the default state vector of a qudit is the vacuum state."""
        tensor = jet.Qudit().tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [1, 0]

    def test_data_contructor(self):
        """Tests that the state vector of a qudit can be manually specified."""
        tensor = jet.Qudit(data=np.arange(4)).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [4]
        assert tensor.data == [0, 1, 2, 3]

    def test_dim_contructor(self):
        """Tests that the dimension of a qudit can be manually specified."""
        tensor = jet.Qudit(dim=3).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [3]
        assert tensor.data == [1, 0, 0]


class TestQuditRegister:
    def test_default_constructor(self):
        """Tests that the default state vector of a qudit register is the vacuum state."""
        tensor = jet.QuditRegister(size=1).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [1, 0]

    def test_data_contructor(self):
        """Tests that the state vector of a qudit register can be manually specified."""
        tensor = jet.QuditRegister(size=3, data=np.arange(8)).tensor()
        assert tensor.indices == ["0", "1", "2"]
        assert tensor.shape == [2, 2, 2]
        assert tensor.data == list(range(8))

    def test_shape_contructor(self):
        """Tests that the shape of a qudit register can be manually specified."""
        tensor = jet.QuditRegister(size=2, dim=3).tensor()
        assert tensor.indices == ["0", "1"]
        assert tensor.shape == [3, 3]
        assert tensor.data == [1, 0, 0, 0, 0, 0, 0, 0, 0]
