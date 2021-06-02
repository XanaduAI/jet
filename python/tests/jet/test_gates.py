import math

import numpy as np
import pytest

import jet


INV_SQRT2 = 1 / math.sqrt(2)


class TestGate:
    @pytest.fixture
    def gate(self):
        """Returns a generic gate with two wires and one parameter."""
        return jet.Gate(name="G", num_wires=2, params=[1])

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


@pytest.mark.parametrize(
    ["gate", "state", "want_tensor"],
    [
        ############################################################################################
        # Hadamard gate
        ############################################################################################
        (
            jet.Hadamard(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, INV_SQRT2]),
        ),
        (
            jet.Hadamard(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, -INV_SQRT2]),
        ),
        ############################################################################################
        # Pauli gates
        ############################################################################################
        (
            jet.PauliX(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1]),
        ),
        (
            jet.PauliX(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.PauliY(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
        (
            jet.PauliY(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
        ),
        (
            jet.PauliZ(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.PauliZ(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -1]),
        ),
        ############################################################################################
        # Unparametrized phase shift gates
        ############################################################################################
        (
            jet.S(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.S(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
        (
            jet.T(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.T(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, INV_SQRT2 + INV_SQRT2 * 1j]),
        ),
        (
            jet.SX(),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1 / 2 + 1j / 2, 1 / 2 - 1j / 2]),
        ),
        (
            jet.SX(),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[1 / 2 - 1j / 2, 1 / 2 + 1j / 2]),
        ),
        ############################################################################################
        # Parametrized phase shift gates
        ############################################################################################
        (
            jet.PhaseShift(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.PhaseShift(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
        (
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
        ),
        (
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
        ),
        ############################################################################################
        # Control gates
        ############################################################################################
        (
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
        ),
        (
            jet.CNOT(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
        ),
        (
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
        ),
        (
            jet.CY(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
        ),
        (
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
        ),
        (
            jet.CZ(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -1]),
        ),
        ############################################################################################
        # SWAP gates
        ############################################################################################
        (
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
        ),
        (
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.SWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
        ),
        (
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1j, 0]),
        ),
        (
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1j, 0, 0]),
        ),
        (
            jet.ISWAP(),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
        ),
        (
            jet.CSWAP(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
        ),
        ############################################################################################
        # Toffoli gate
        ############################################################################################
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[1, 0, 0, 0, 0, 0, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 1, 0, 0, 0, 0, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 1, 0, 0, 0, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 1, 0, 0, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 1, 0, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 1, 0, 0]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
        ),
        (
            jet.Toffoli(),
            jet.Tensor(indices=["3", "4", "5"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1", "2"], shape=[2, 2, 2], data=[0, 0, 0, 0, 0, 0, 1, 0]),
        ),
        ############################################################################################
        # Rotation gates
        ############################################################################################
        (
            jet.RX(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -1j]),
        ),
        (
            jet.RX(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
        ),
        (
            jet.RY(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1]),
        ),
        (
            jet.RY(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
        ),
        (
            jet.RZ(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
        ),
        (
            jet.RZ(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
        (
            jet.Rot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -INV_SQRT2 + INV_SQRT2 * 1j]),
        ),
        (
            jet.Rot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2 + INV_SQRT2 * 1j, 0]),
        ),
        ############################################################################################
        # Controlled rotation gates
        ############################################################################################
        (
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -1j]),
        ),
        (
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
        ),
        (
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
        ),
        (
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1, 0]),
        ),
        (
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
        ),
        (
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
        ),
        (
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
        ),
        (
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
        ),
        (
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(
                indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -INV_SQRT2 + INV_SQRT2 * 1j]
            ),
        ),
        (
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(
                indices=["0", "1"], shape=[2, 2], data=[0, 0, INV_SQRT2 + INV_SQRT2 * 1j, 0]
            ),
        ),
        ############################################################################################
        # U gates
        ############################################################################################
        (
            jet.U1(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
        ),
        (
            jet.U1(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
        (
            jet.U2(math.pi / 2, math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, INV_SQRT2 * 1j]),
        ),
        (
            jet.U2(math.pi / 2, math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, -INV_SQRT2 * 1j]),
        ),
        (
            jet.U3(2 * math.pi, math.pi, math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
        ),
        (
            jet.U3(2 * math.pi, math.pi, math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
        ),
    ],
)
def test_gate(gate, state, want_tensor):
    have_tensor = jet.contract_tensors(gate.tensor(), state)
    assert have_tensor.data == pytest.approx(want_tensor.data)
    assert have_tensor.shape == want_tensor.shape
    assert have_tensor.indices == want_tensor.indices
