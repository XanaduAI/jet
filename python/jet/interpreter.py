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

    Raises:
        ValueError: If the given program contains an unsupported or invalid statement.

    Args:
        program (XIRProgram): XIR script to execute.

    Returns:
        List of NumPy values representing the output of the XIR program.
    """
    result: List[np.generic] = []

    num_wires = find_number_of_modes(program)
    # TODO: Extract the Fock cutoff dimension from the XIR script.
    circuit = Circuit(num_wires=num_wires, dim=2)

    for stmt in program.statements:
        name = stmt.name.lower()

        if name in GateFactory.registry:
            # TODO: Automatically insert the Fock cutoff dimension for CV gates.
            gate = GateFactory.create(name, *stmt.params)
            circuit.append_gate(gate, wire_ids=stmt.wires)

        elif name == "amplitude":
            if "state" not in stmt.params:
                raise ValueError(f"Statement '{stmt}' is missing a 'state' parameter.")

            # TODO: Use a list representation of the "state" key.
            state = list(map(int, bin(stmt.params["state"])[2:].zfill(num_wires)))

            if len(state) != num_wires:
                raise ValueError(f"Statement '{stmt}' has an invalid 'state' parameter.")

            if stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_amplitude(circuit=circuit, state=state)
            result.append(output)

        else:
            raise ValueError(f"Statement '{stmt}' is not supported.")

    return result


def _compute_amplitude(
    circuit: Circuit, state: List[int], dtype: type = np.complex128
) -> np.number:
    """Computes the amplitude of a state at the end of a circuit.

    Args:
        circuit (Circuit): Circuit to apply the amplitude measurement to.
        state (list[int]): State to measure the amplitude of.
        dtype (type): Data type of the amplitude.

    Returns:
        NumPy number representing the amplitude of the given state.
    """
    # Do not modify the original circuit.
    circuit = deepcopy(circuit)

    for (i, value) in enumerate(state):
        # Fill a state vector with zeros save for a one at index `value`.
        data = (np.arange(circuit.dimension) == value).astype(np.complex128)
        qudit = Qudit(dim=circuit.dimension, data=data)
        circuit.append_state(qudit, wire_ids=[i])

    # TODO: Find a contraction path and use the TBCC.
    tn = circuit.tensor_network(dtype=dtype)
    amplitude = tn.contract()
    return dtype(amplitude.scalar)
