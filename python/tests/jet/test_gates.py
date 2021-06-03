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


# def get_test_gate_parameter_id(val) -> str:
#     """Returns the ID of the given test_gate() parameter."""
#     if isinstance(val, jet.Gate):
#         return val.name
#     elif isinstance(val, jet.TensorC128):
#         return str(tuple(val.indices))


@pytest.mark.parametrize(
    ["gate", "state", "want_tensor"],
    [
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
            jet.PhaseShift(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="P|0>",
        ),
        pytest.param(
            jet.PhaseShift(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="P|1>",
        ),
        pytest.param(
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CP|00>",
        ),
        pytest.param(
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CP|01>",
        ),
        pytest.param(
            jet.CPhaseShift(math.pi / 2),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 1, 0]),
            id="CP|10>",
        ),
        pytest.param(
            jet.CPhaseShift(math.pi / 2),
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
            jet.RX(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -1j]),
            id="RX(pi)|0>",
        ),
        pytest.param(
            jet.RX(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
            id="RX(pi)|1>",
        ),
        pytest.param(
            jet.RY(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1]),
            id="RY(pi)|0>",
        ),
        pytest.param(
            jet.RY(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
            id="RY(pi)|1>",
        ),
        pytest.param(
            jet.RZ(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1j, 0]),
            id="RZ(pi)|0>",
        ),
        pytest.param(
            jet.RZ(math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="RY(pi)|1>",
        ),
        pytest.param(
            jet.Rot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, -INV_SQRT2 + INV_SQRT2 * 1j]),
            id="Rot(pi/2,pi,2*pi)|0>",
        ),
        pytest.param(
            jet.Rot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2 + INV_SQRT2 * 1j, 0]),
            id="Rot(pi/2,pi,2*pi)|1>",
        ),
        ############################################################################################
        # Controlled rotation gates
        ############################################################################################
        pytest.param(
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRX|00>",
        ),
        pytest.param(
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRX|01>",
        ),
        pytest.param(
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -1j]),
            id="CRX|10>",
        ),
        pytest.param(
            jet.CRX(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
            id="CRX|11>",
        ),
        pytest.param(
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRY|00>",
        ),
        pytest.param(
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRY|01>",
        ),
        pytest.param(
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1]),
            id="CRY|10>",
        ),
        pytest.param(
            jet.CRY(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1, 0]),
            id="CRY|11>",
        ),
        pytest.param(
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRZ|00>",
        ),
        pytest.param(
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRZ|01>",
        ),
        pytest.param(
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, -1j, 0]),
            id="CRZ|10>",
        ),
        pytest.param(
            jet.CRZ(math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 0, 1]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, 1j]),
            id="CRZ|11>",
        ),
        pytest.param(
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[1, 0, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[1, 0, 0, 0]),
            id="CRot|00>",
        ),
        pytest.param(
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 1, 0, 0]),
            jet.Tensor(indices=["0", "1"], shape=[2, 2], data=[0, 1, 0, 0]),
            id="CRot|01>",
        ),
        pytest.param(
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
            jet.Tensor(indices=["2", "3"], shape=[2, 2], data=[0, 0, 1, 0]),
            jet.Tensor(
                indices=["0", "1"], shape=[2, 2], data=[0, 0, 0, -INV_SQRT2 + INV_SQRT2 * 1j]
            ),
            id="CRot|10>",
        ),
        pytest.param(
            jet.CRot(math.pi / 2, math.pi, 2 * math.pi),
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
            jet.U1(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[1, 0]),
            id="U1|0>",
        ),
        pytest.param(
            jet.U1(math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="U1|1>",
        ),
        pytest.param(
            jet.U2(math.pi / 2, math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, INV_SQRT2 * 1j]),
            id="U2|0>",
        ),
        pytest.param(
            jet.U2(math.pi / 2, math.pi),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[INV_SQRT2, -INV_SQRT2 * 1j]),
            id="U2|1>",
        ),
        pytest.param(
            jet.U3(2 * math.pi, math.pi, math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[1, 0]),
            jet.Tensor(indices=["0"], shape=[2], data=[-1, 0]),
            id="U3(2*pi,pi,pi/2)|0>",
        ),
        pytest.param(
            jet.U3(2 * math.pi, math.pi, math.pi / 2),
            jet.Tensor(indices=["1"], shape=[2], data=[0, 1]),
            jet.Tensor(indices=["0"], shape=[2], data=[0, 1j]),
            id="U3(2*pi,pi,pi/2)|1>",
        ),
    ],
)
def test_gate(gate, state, want_tensor):
    have_tensor = jet.contract_tensors(gate.tensor(), state)
    assert have_tensor.data == pytest.approx(want_tensor.data)
    assert have_tensor.shape == want_tensor.shape
    assert have_tensor.indices == want_tensor.indices
