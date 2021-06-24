from math import sqrt

import numpy as np
import pytest

import jet


def test_wire_index():
    """Tests that the index property of a wire has the correct form."""
    assert jet.Wire(id_=1, depth=2).index == "1-2"


class TestCircuit:
    @pytest.fixture
    def circuit(self):
        """Returns a qubit circuit with four wires."""
        return jet.Circuit(num_wires=4, dim=2)

    @pytest.fixture
    def validator(self):
        """Returns the wrapper function inside the circuit append validator."""
        return jet.Circuit._append_validator(lambda *args: None)

    def test_constructor(self, circuit):
        """Tests that the circuit constructor initializes the correct wires and parts."""
        assert list(circuit.wires) == [jet.Wire(i, 0, False) for i in range(4)]
        assert list(circuit.parts) == [jet.Qubit() for _ in range(4)]

    def test_append_dangling_wire(self, circuit, validator):
        """Tests that a ValueError is raised when the append validator is given
        a circuit component which spans a different number of wires.
        """
        match = (
            r"Number of wire IDs \(2\) must match the number of "
            r"wires connected to the circuit component \(1\)."
        )
        with pytest.raises(ValueError, match=match):
            validator(self=circuit, part=jet.Qubit(), wire_ids=[0, 1])

    def test_append_fake_wire(self, circuit, validator):
        """Tests that a ValueError is raised when the append validator is given
        a wire ID which does not exist in the circuit.
        """
        with pytest.raises(ValueError, match=r"Wire ID 4 falls outside the range \[0, 4\)."):
            validator(self=circuit, part=jet.Qubit(), wire_ids=[4])

    def test_append_duplicate_wire(self, circuit, validator):
        """Tests that a ValueError is raised when the append validator is given
        a duplicate wire ID.
        """
        with pytest.raises(ValueError, match="Wire ID 0 is specified more than once."):
            validator(self=circuit, part=jet.QubitRegister(2), wire_ids=[0, 0])

    def test_append_closed_wire(self, circuit, validator):
        """Tests that a ValueError is raised when the append validator is given
        a wire ID associated with a closed wire.
        """
        circuit._wires[0].closed = True
        with pytest.raises(ValueError, match="Wire 0 is closed."):
            validator(self=circuit, part=jet.Qubit(), wire_ids=[0])

    def test_append_one_wire_gate(self, circuit):
        """Tests that a gate which transforms one wire can be appended to the circuit."""
        gate = jet.GateFactory.create("H")
        circuit.append_gate(gate, wire_ids=[3])
        assert gate.indices == ["3-1", "3-0"]
        assert list(circuit.parts)[-1] == gate
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=False),
            jet.Wire(1, depth=0, closed=False),
            jet.Wire(2, depth=0, closed=False),
            jet.Wire(3, depth=1, closed=False),
        ]

    def test_append_two_wire_gate(self, circuit):
        """Tests that a gate which transforms two wires can be appended to the circuit."""
        gate = jet.GateFactory.create("CNOT")
        circuit.append_gate(gate, wire_ids=[2, 3])
        assert gate.indices == ["2-1", "3-1", "2-0", "3-0"]
        assert list(circuit.parts)[-1] == gate
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=False),
            jet.Wire(1, depth=0, closed=False),
            jet.Wire(2, depth=1, closed=False),
            jet.Wire(3, depth=1, closed=False),
        ]

    def test_append_one_wire_state(self, circuit):
        """Tests that a state which terminates one wire can be appended to the circuit."""
        state = jet.Qubit()
        circuit.append_state(state, wire_ids=[0])
        assert state.indices == ["0-0"]
        assert list(circuit.parts)[-1] == state
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=True),
            jet.Wire(1, depth=0, closed=False),
            jet.Wire(2, depth=0, closed=False),
            jet.Wire(3, depth=0, closed=False),
        ]

    def test_append_two_wire_state(self, circuit):
        """Tests that a state which terminates two wires can be appended to the circuit."""
        state = jet.QubitRegister(size=2)
        circuit.append_state(state, wire_ids=[0, 1])
        assert state.indices == ["0-0", "1-0"]
        assert list(circuit.parts)[-1] == state
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=True),
            jet.Wire(1, depth=0, closed=True),
            jet.Wire(2, depth=0, closed=False),
            jet.Wire(3, depth=0, closed=False),
        ]

    def test_indices(self, circuit):
        """Tests that the correct index labels are derived for a sequence of wires."""
        gate = jet.GateFactory.create("H")
        circuit.append_gate(gate, wire_ids=[0])
        assert circuit.indices([0]) == ["0-1"]
        assert circuit.indices([1, 2, 3]) == ["1-0", "2-0", "3-0"]

    def test_tensor_network_flip(self):
        """Tests that a quantum circuit which flips a single qubit can be
        converted into a tensor network.
        """
        circuit = jet.Circuit(num_wires=1)
        circuit.append_gate(jet.PauliX(), wire_ids=[0])
        tn = circuit.tensor_network()

        tensor = tn.contract()
        assert tensor.indices == ["0-1"]
        assert tensor.shape == [2]
        assert tensor.data == pytest.approx([0, 1])

    def test_tensor_network_bell(self):
        """Tests that a quantum circuit which produces a Bell state can be
        converted into a tensor network.
        """
        circuit = jet.Circuit(num_wires=2)
        circuit.append_gate(jet.GateFactory.create("H"), wire_ids=[0])
        circuit.append_gate(jet.GateFactory.create("CNOT"), wire_ids=[0, 1])

        tn = circuit.tensor_network()
        tensor = tn.contract()

        assert tensor.indices == ["0-2", "1-1"]
        assert tensor.shape == [2, 2]
        assert tensor.data == pytest.approx([1 / sqrt(2), 0, 0, 1 / sqrt(2)])

    @pytest.mark.parametrize(
        "state, want_amplitude",
        [
            (jet.QubitRegister(size=2, data=np.array([1, 0, 0, 0])), 1 / sqrt(2)),
            (jet.QubitRegister(size=2, data=np.array([0, 1, 0, 0])), 0),
            (jet.QubitRegister(size=2, data=np.array([0, 0, 1, 0])), 0),
            (jet.QubitRegister(size=2, data=np.array([0, 0, 0, 1])), 1 / sqrt(2)),
        ],
    )
    def test_tensor_network_amplitude(self, state, want_amplitude):
        """Tests that a quantum circuit with an amplitude measurement can be
        converted into a tensor network.
        """
        circuit = jet.Circuit(num_wires=2)
        circuit.append_gate(jet.GateFactory.create("H"), wire_ids=[0])
        circuit.append_gate(jet.GateFactory.create("CNOT"), wire_ids=[0, 1])
        circuit.append_state(state, wire_ids=[0, 1])
        have_amplitude = circuit.tensor_network().contract().scalar
        assert have_amplitude == want_amplitude
