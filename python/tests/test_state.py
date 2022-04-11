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
        state1._data = lambda: np.array([0, 0, 1])  # pylint: disable=protected-access
        assert state0 != state1


class TestQudit:
    def test_init(self):
        """Tests that the default state vector of a qudit is the vacuum state."""
        qudit = jet.Qudit(dim=3)
        assert qudit.name == "Qudit(d=3)"

        tensor = qudit.tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [3]
        assert tensor.data == [1, 0, 0]

    def test_init_with_data(self):
        """Tests that the state vector of a qudit can be manually specified."""
        tensor = jet.Qudit(dim=4, data=np.arange(4)).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [4]
        assert tensor.data == [0, 1, 2, 3]


class TestQuditRegister:
    def test_init(self):
        """Tests that the default state vector of a qudit register is the vacuum state."""
        qudits = jet.QuditRegister(dim=3, size=2)
        assert qudits.name == "Qudit(d=3)[2]"

        tensor = qudits.tensor()
        assert tensor.indices == ["0", "1"]
        assert tensor.shape == [3, 3]
        assert tensor.data == [1, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_init_with_data(self):
        """Tests that the state vector of a qudit register can be manually specified."""
        tensor = jet.QuditRegister(dim=4, size=3, data=np.arange(4**3)).tensor()
        assert tensor.indices == ["0", "1", "2"]
        assert tensor.shape == [4, 4, 4]
        assert tensor.data == list(range(4**3))


class TestQubit:
    def test_init(self):
        """Tests that the default state vector of a qubit is the vacuum state."""
        qubit = jet.Qubit()
        assert qubit.name == "Qubit"

        tensor = qubit.tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [1, 0]

    def test_init_with_data(self):
        """Tests that the state vector of a qubit can be manually specified."""
        tensor = jet.Qubit(data=np.arange(2)).tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [0, 1]


class TestQubitRegister:
    def test_init(self):
        """Tests that the default state vector of a qubit register is the vacuum state."""
        qubits = jet.QubitRegister(size=1)
        assert qubits.name == "Qubit[1]"

        tensor = qubits.tensor()
        assert tensor.indices == ["0"]
        assert tensor.shape == [2]
        assert tensor.data == [1, 0]

    def test_init_with_data(self):
        """Tests that the state vector of a qubit register can be manually specified."""
        tensor = jet.QubitRegister(size=3, data=np.arange(8)).tensor()
        assert tensor.indices == ["0", "1", "2"]
        assert tensor.shape == [2, 2, 2]
        assert tensor.data == list(range(8))
