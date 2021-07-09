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

    def test_constructor(self, circuit):
        """Tests that the circuit constructor initializes the correct wires and operations."""
        assert list(circuit.wires) == [jet.Wire(i, 0, False) for i in range(4)]
        assert list(circuit.operations) == [jet.Operation(jet.Qubit(), [i]) for i in range(4)]

    def test_validate_fake_wire(self, circuit):
        """Tests that a ValueError is raised when the wire ID validator is given
        a wire ID which does not exist in the circuit.
        """
        with pytest.raises(ValueError, match=r"Wire ID 4 falls outside the range \[0, 4\)."):
            circuit._validate_wire_ids(wire_ids=[4])

    def test_validate_duplicate_wire(self, circuit):
        """Tests that a ValueError is raised when the wire ID validator is given
        a duplicate wire ID.
        """
        with pytest.raises(ValueError, match="Wire ID 0 is specified more than once."):
            circuit._validate_wire_ids(wire_ids=[0, 0])

    def test_validate_closed_wire(self, circuit):
        """Tests that a ValueError is raised when the wire ID validator is given
        a wire ID associated with a closed wire.
        """
        circuit._wires[0].closed = True
        with pytest.raises(ValueError, match="Wire 0 is closed."):
            circuit._validate_wire_ids(wire_ids=[0])

    def test_append_dangling_gate(self, circuit):
        """Tests that a ValueError is raised when a gate is appended to the
        circuit with the wrong number of wires.
        """
        match = (
            r"Number of wire IDs \(2\) must match the number of wires connected to the gate \(1\)."
        )
        gate = jet.GateFactory.create("H")
        with pytest.raises(ValueError, match=match):
            circuit.append_gate(gate, wire_ids=[0, 1])

    def test_append_one_wire_gate(self, circuit):
        """Tests that a gate which transforms one wire can be appended to the circuit."""
        gate = jet.GateFactory.create("H")
        circuit.append_gate(gate, wire_ids=[3])
        assert gate.indices == ["3-1", "3-0"]
        assert list(circuit.operations)[-1] == jet.Operation(gate, [3])
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
        assert list(circuit.operations)[-1] == jet.Operation(gate, [2, 3])
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=False),
            jet.Wire(1, depth=0, closed=False),
            jet.Wire(2, depth=1, closed=False),
            jet.Wire(3, depth=1, closed=False),
        ]

    def test_append_dangling_state(self, circuit):
        """Tests that a ValueError is raised when a state is appended to the
        circuit with the wrong number of wires.
        """
        match = (
            r"Number of wire IDs \(2\) must match the number of wires connected to the state \(1\)."
        )
        state = jet.Qubit()
        with pytest.raises(ValueError, match=match):
            circuit.append_state(state, wire_ids=[0, 1])

    def test_append_one_wire_state(self, circuit):
        """Tests that a state which terminates one wire can be appended to the circuit."""
        state = jet.Qubit()
        circuit.append_state(state, wire_ids=[0])
        assert state.indices == ["0-0"]
        assert list(circuit.operations)[-1] == jet.Operation(state, [0])
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
        assert list(circuit.operations)[-1] == jet.Operation(state, [0, 1])
        assert list(circuit.wires) == [
            jet.Wire(0, depth=0, closed=True),
            jet.Wire(1, depth=0, closed=True),
            jet.Wire(2, depth=0, closed=False),
            jet.Wire(3, depth=0, closed=False),
        ]

    @pytest.mark.parametrize(
        ["operations", "observable", "want_result"],
        [
            (
                [
                    jet.Operation(part=jet.GateFactory.create("Hadamard"), wire_ids=[0]),
                    jet.Operation(part=jet.GateFactory.create("CNOT"), wire_ids=[0, 1]),
                ],
                [],
                1,
            ),
            (
                [],
                [
                    jet.Operation(part=jet.GateFactory.create("Z"), wire_ids=[0]),
                ],
                1,
            ),
            (
                [
                    jet.Operation(part=jet.GateFactory.create("X"), wire_ids=[0]),
                ],
                [
                    jet.Operation(part=jet.GateFactory.create("Z"), wire_ids=[0]),
                ],
                -1,
            ),
            (
                [
                    jet.Operation(part=jet.GateFactory.create("RX", 1), wire_ids=[0]),
                    jet.Operation(part=jet.GateFactory.create("RY", 2), wire_ids=[1]),
                    jet.Operation(part=jet.GateFactory.create("CNOT"), wire_ids=[0, 1]),
                ],
                [
                    jet.Operation(part=jet.GateFactory.create("Y"), wire_ids=[0]),
                    jet.Operation(part=jet.GateFactory.create("X"), wire_ids=[1]),
                ],
                -0.8414709848078962,
            ),
        ],
    )
    def test_take_expected_value(self, circuit, operations, observable, want_result):
        """Tests that the correct expected value is returned with respect to the observable."""
        for op in operations:
            circuit.append_gate(gate=op.part, wire_ids=op.wire_ids)

        circuit.take_expected_value(observable)

        tn = circuit.tensor_network()
        tensor = tn.contract()
        have_result = tensor.scalar

        assert have_result == pytest.approx(want_result)

    def test_indices(self, circuit):
        """Tests that the correct index labels are derived for a sequence of wires."""
        gate = jet.GateFactory.create("H")
        circuit.append_gate(gate, wire_ids=[0])
        assert list(circuit.indices([0])) == ["0-1"]
        assert list(circuit.indices([1, 2, 3])) == ["1-0", "2-0", "3-0"]

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
