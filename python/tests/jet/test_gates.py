from itertools import chain
from math import pi, sqrt

import numpy as np
import pytest

import jet

INV_SQRT2 = 1 / sqrt(2)


class MockGate(jet.Gate):
    """MockGate represents a fictional, unnormalized gate which can be applied to pairs of qutrits."""

    def __init__(self):
        super().__init__(name="MockGate", num_params=0, num_wires=2)

    def _data(self) -> np.ndarray:
        return np.eye(3 ** 2) * (1 + 1j)


class TestGate:
    @pytest.fixture
    def gate(self):
        """Returns a mock gate instance."""
        return MockGate()

    def test_tensor_indices_not_set(self, gate):
        """Tests that the correct tensor is returned for a gate with unset indices."""
        tensor = gate.tensor()

        assert tensor.indices == ["0", "1", "2", "3"]
        assert tensor.shape == [3, 3, 3, 3]
        assert tensor.data == np.ravel(np.eye(9) * (1 + 1j)).tolist()

    def test_tensor_indices_are_set(self, gate):
        """Tests that the correct tensor is returned for a gate with specified indices."""
        gate.indices = ["r", "g", "b", "a"]
        tensor = gate.tensor()

        assert tensor.indices == ["r", "g", "b", "a"]
        assert tensor.shape == [3, 3, 3, 3]
        assert tensor.data == np.ravel(np.eye(9) * (1 + 1j)).tolist()

    def test_tensor_adjoint(self, gate):
        """Tests that the correct adjoint tensor is returned for a gate."""
        tensor = gate.tensor(adjoint=True)

        assert tensor.indices == ["0", "1", "2", "3"]
        assert tensor.shape == [3, 3, 3, 3]
        assert tensor.data == np.ravel(np.eye(9) * (1 - 1j) / 2).tolist()

    @pytest.mark.parametrize("indices", [1, ["i", "j", "k", 4], ["x", "x", "x", "x"], []])
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


@pytest.mark.parametrize(
    ["gate", "state", "want_tensor"],
    [
        ############################################################################################
        # Continuous variable Fock gates
        ############################################################################################
        pytest.param(
            jet.Displacement(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[1, 0, 0]),
            jet.Tensor(
                indices=["0"], shape=[3], data=[0.135335283237, 0.270670566473j, -0.382785986042]
            ),
            id="Displacement(2,pi/2,3)|1>",
        ),
        pytest.param(
            jet.Displacement(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[0, 1, 0]),
            jet.Tensor(
                indices=["0"], shape=[3], data=[0.270670566473j, -0.40600584971, -0.382785986042j]
            ),
            id="Displacement(2,pi/2,3)|2>",
        ),
        pytest.param(
            jet.Displacement(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[0, 0, 1]),
            jet.Tensor(
                indices=["0"], shape=[3], data=[-0.382785986042, -0.382785986042j, 0.135335283237]
            ),
            id="Displacement(2,pi/2,3)|3>",
        ),
        pytest.param(
            jet.Squeezing(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[1, 0, 0]),
            jet.Tensor(indices=["0"], shape=[3], data=[0.515560111756, 0, -0.351442087775j]),
            id="Squeezing(2,pi/2,3)|1>",
        ),
        pytest.param(
            jet.Squeezing(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[0, 1, 0]),
            jet.Tensor(indices=["0"], shape=[3], data=[0, 0.137037026803, 0]),
            id="Squeezing(2,pi/2,3)|2>",
        ),
        pytest.param(
            jet.Squeezing(2, pi / 2, 3),
            jet.Tensor(indices=["1"], shape=[3], data=[0, 0, 1]),
            jet.Tensor(indices=["0"], shape=[3], data=[-0.351442087775j, 0, -0.203142935143]),
            id="Squeezing(2,pi/2,3)|3>",
        ),
        pytest.param(
            jet.TwoModeSqueezing(3, pi / 4, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0.099327927419, 0, 0, 0.069888119434 + 0.069888119434j],
            ),
            id="TwoModeSqueezing(3,pi/4,2)|00>",
        ),
        pytest.param(
            jet.TwoModeSqueezing(3, pi / 4, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0, 0.009866037165, 0, 0],
            ),
            id="TwoModeSqueezing(3,pi/4,2)|01>",
        ),
        pytest.param(
            jet.TwoModeSqueezing(3, pi / 4, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0, 0, 0.009866037165, 0],
            ),
            id="TwoModeSqueezing(3,pi/4,2)|10>",
        ),
        pytest.param(
            jet.TwoModeSqueezing(3, pi / 4, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[-0.069888119434 + 0.069888119434j, 0, 0, -0.097367981372],
            ),
            id="TwoModeSqueezing(3,pi/4,2)|11>",
        ),
        pytest.param(
            jet.Beamsplitter(pi / 4, pi / 2, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[1, 0, 0, 0],
            ),
            id="Beamsplitter(pi/4,pi/2,2)|00>",
        ),
        pytest.param(
            jet.Beamsplitter(pi / 4, pi / 2, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0, INV_SQRT2, INV_SQRT2 * 1j, 0],
            ),
            id="Beamsplitter(pi/4,pi/2,2)|01>",
        ),
        pytest.param(
            jet.Beamsplitter(pi / 4, pi / 2, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0, INV_SQRT2 * 1j, INV_SQRT2, 0],
            ),
            id="Beamsplitter(pi/4,pi/2,2)|10>",
        ),
        pytest.param(
            jet.Beamsplitter(pi / 4, pi / 2, 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(
                indices=["0", "1"],
                shape=[2, 2],
                data=[0, 0, 0, 0],
            ),
            id="Beamsplitter(pi/4,pi/2,2)|11>",
        ),
        ############################################################################################
        # Hadamard gate
        ############################################################################################
        pytest.param(
            jet.Hadamard(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, INV_SQRT2]),
            id="H|0>",
        ),
        pytest.param(
            jet.Hadamard(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, -INV_SQRT2]),
            id="H|1>",
        ),
        ############################################################################################
        # Pauli gates
        ############################################################################################
        pytest.param(
            jet.PauliX(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1]),
            id="X|0>",
        ),
        pytest.param(
            jet.PauliX(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="X|1>",
        ),
        pytest.param(
            jet.PauliY(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="Y|0>",
        ),
        pytest.param(
            jet.PauliY(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
            id="Y|1>",
        ),
        pytest.param(
            jet.PauliZ(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="Z|0>",
        ),
        pytest.param(
            jet.PauliZ(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -1]),
            id="Z|1>",
        ),
        ############################################################################################
        # Unparametrized phase shift gates
        ############################################################################################
        pytest.param(
            jet.S(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="S|0>",
        ),
        pytest.param(
            jet.S(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="S|1>",
        ),
        pytest.param(
            jet.T(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="T|0>",
        ),
        pytest.param(
            jet.T(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, INV_SQRT2 + INV_SQRT2 * 1j]),
            id="T|1>",
        ),
        pytest.param(
            jet.SX(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1 / 2 + 1j / 2, 1 / 2 - 1j / 2]),
            id="SX|0>",
        ),
        pytest.param(
            jet.SX(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[1 / 2 - 1j / 2, 1 / 2 + 1j / 2]),
            id="SX|1>",
        ),
        ############################################################################################
        # Parametrized phase shift gates
        ############################################################################################
        pytest.param(
            jet.PhaseShift(pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="P|0>",
        ),
        pytest.param(
            jet.PhaseShift(pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="P|1>",
        ),
        pytest.param(
            jet.CPhaseShift(pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CP|00>",
        ),
        pytest.param(
            jet.CPhaseShift(pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CP|01>",
        ),
        pytest.param(
            jet.CPhaseShift(pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
            id="CP|10>",
        ),
        pytest.param(
            jet.CPhaseShift(pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
            id="CP|11>",
        ),
        ############################################################################################
        # Control gates
        ############################################################################################
        pytest.param(
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CNOT|00>",
        ),
        pytest.param(
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CNOT|01>",
        ),
        pytest.param(
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
            id="CNOT|10>",
        ),
        pytest.param(
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
            id="CNOT|11>",
        ),
        pytest.param(
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CY|00>",
        ),
        pytest.param(
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CY|01>",
        ),
        pytest.param(
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
            id="CY|10>",
        ),
        pytest.param(
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
            id="CY|11>",
        ),
        pytest.param(
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CZ|00>",
        ),
        pytest.param(
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CZ|01>",
        ),
        pytest.param(
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
            id="CZ|10>",
        ),
        pytest.param(
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -1]),
            id="CZ|11>",
        ),
        ############################################################################################
        # SWAP gates
        ############################################################################################
        pytest.param(
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="SWAP|00>",
        ),
        pytest.param(
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
            id="SWAP|01>",
        ),
        pytest.param(
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="SWAP|10>",
        ),
        pytest.param(
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
            id="SWAP|11>",
        ),
        pytest.param(
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="ISWAP|00>",
        ),
        pytest.param(
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1j, 0]),
            id="ISWAP|01>",
        ),
        pytest.param(
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1j, 0, 0]),
            id="ISWAP|10>",
        ),
        pytest.param(
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
            id="ISWAP|11>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            id="CSWAP|000>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            id="CSWAP|001>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            id="CSWAP|010>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            id="CSWAP|011>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            id="CSWAP|100>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            id="CSWAP|101>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            id="CSWAP|110>",
        ),
        pytest.param(
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            id="CSWAP|111>",
        ),
        ############################################################################################
        # Toffoli gate
        ############################################################################################
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            id="T|000>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            id="T|001>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            id="T|010>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            id="T|011>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            id="T|100>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            id="T|101>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            id="T|110>",
        ),
        pytest.param(
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            id="T|111>",
        ),
        ############################################################################################
        # Rotation gates
        ############################################################################################
        pytest.param(
            jet.RX(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -1j]),
            id="RX(pi)|0>",
        ),
        pytest.param(
            jet.RX(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
            id="RX(pi)|1>",
        ),
        pytest.param(
            jet.RY(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1]),
            id="RY(pi)|0>",
        ),
        pytest.param(
            jet.RY(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
            id="RY(pi)|1>",
        ),
        pytest.param(
            jet.RZ(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
            id="RZ(pi)|0>",
        ),
        pytest.param(
            jet.RZ(pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="RY(pi)|1>",
        ),
        pytest.param(
            jet.Rot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -INV_SQRT2 + INV_SQRT2 * 1j]),
            id="Rot(pi/2,pi,2*pi)|0>",
        ),
        pytest.param(
            jet.Rot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2 + INV_SQRT2 * 1j, 0]),
            id="Rot(pi/2,pi,2*pi)|1>",
        ),
        ############################################################################################
        # Controlled rotation gates
        ############################################################################################
        pytest.param(
            jet.CRX(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRX|00>",
        ),
        pytest.param(
            jet.CRX(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRX|01>",
        ),
        pytest.param(
            jet.CRX(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -1j]),
            id="CRX|10>",
        ),
        pytest.param(
            jet.CRX(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
            id="CRX|11>",
        ),
        pytest.param(
            jet.CRY(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRY|00>",
        ),
        pytest.param(
            jet.CRY(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRY|01>",
        ),
        pytest.param(
            jet.CRY(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
            id="CRY|10>",
        ),
        pytest.param(
            jet.CRY(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1, 0]),
            id="CRY|11>",
        ),
        pytest.param(
            jet.CRZ(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRZ|00>",
        ),
        pytest.param(
            jet.CRZ(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRZ|01>",
        ),
        pytest.param(
            jet.CRZ(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
            id="CRZ|10>",
        ),
        pytest.param(
            jet.CRZ(pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
            id="CRZ|11>",
        ),
        pytest.param(
            jet.CRot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRot|00>",
        ),
        pytest.param(
            jet.CRot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRot|01>",
        ),
        pytest.param(
            jet.CRot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(
                indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -INV_SQRT2 + INV_SQRT2 * 1j]
            ),
            id="CRot|10>",
        ),
        pytest.param(
            jet.CRot(pi / 2, pi, 2 * pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(
                indices=["0", "1"], shape=[2, 2], data=[0, 0, INV_SQRT2 + INV_SQRT2 * 1j, 0]
            ),
            id="CRot|11>",
        ),
        ############################################################################################
        # U gates
        ############################################################################################
        pytest.param(
            jet.U1(pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="U1|0>",
        ),
        pytest.param(
            jet.U1(pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="U1|1>",
        ),
        pytest.param(
            jet.U2(pi / 2, pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, INV_SQRT2 * 1j]),
            id="U2|0>",
        ),
        pytest.param(
            jet.U2(pi / 2, pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, -INV_SQRT2 * 1j]),
            id="U2|1>",
        ),
        pytest.param(
            jet.U3(2 * pi, pi, pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
            id="U3(2*pi,pi,pi/2)|0>",
        ),
        pytest.param(
            jet.U3(2 * pi, pi, pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="U3(2*pi,pi,pi/2)|1>",
        ),
    ],
)
def test_gate(gate, state, want_tensor):
    """Tests that the correct transformation is applied by a gate to a basis state."""
    have_tensor = jet.contract_tensors(gate.tensor(), state)
    assert have_tensor.data == pytest.approx(want_tensor.data)
    assert have_tensor.shape == want_tensor.shape
    assert have_tensor.indices == want_tensor.indices
