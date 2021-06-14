from copy import deepcopy
from typing import List

import numpy as np

from xir import XIRProgram
from xir.interfaces import find_number_of_modes

from .circuit import Circuit
from .gate import GateFactory
from .state import Qudit

__all__ = ["run_xir_program"]


def run_xir_program(program: XIRProgram) -> List[np.generic]:
    """Executes an XIR program.

    Args:
        program (XIRProgram): XIR script to execute.

    Returns:
        List of NumPy values representing the output of the XIR program.
    """
    result: List[np.generic] = []

    num_wires = find_number_of_modes(program)
    # TODO: Extract the Fock cutoff dimension from the XIR script.
    circuit = Circuit(num_wires=num_wires, dim=2)

    for statement in program.statements:
        name = statement.name.lower()

        if name in GateFactory.registry:
            gate = GateFactory.create(name, *statement.params)
            circuit.append_gate(gate, wire_ids=statement.wires)

        elif name == "amplitude":
            if "state" not in statement.params:
                raise ValueError(f"Statement '{statement}' is missing a 'state' parameter.")

            # TODO: Use a list representation of the "state" key.
            state = list(map(int, bin(statement.params["state"])[2:].zfill(num_wires)))
            output = _compute_amplitude(circuit=circuit, state=state)
            result.append(output)

        else:
            raise ValueError(f"Statement '{statement}' is not supported.")

    return result


def _compute_amplitude(circuit: Circuit, state: List[int]) -> np.complex128:
    """Computes the amplitude of a state at the end of a circuit.

    Args:
        circuit (Circuit): circuit to apply the amplitude measurement to.
        state (list[int]): state to measure the amplitude of.

    Returns:
        Complex number representing the amplitude of the given state.
    """
    # Don't augment the original circuit with the terminal amplitude qudits.
    circuit = deepcopy(circuit)

    for (i, value) in enumerate(state):
        # Fill a state vector with zeros save for a one at index `value`.
        data = (np.arange(circuit.dimension) == value).astype(np.complex128)
        qudit = Qudit(dim=circuit.dimension, data=data)
        circuit.append_state(qudit, wire_ids=[i])

    # TODO: Find a contraction path and use the TBCC.
    tn = circuit.tensor_network()
    amplitude = tn.contract()
    return np.complex128(amplitude.scalar)
