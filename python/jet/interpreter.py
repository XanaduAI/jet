from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Sequence, Set, Union

import numpy as np

from xir import Statement, XIRProgram, XIRTransformer, xir_parser
from xir.interfaces import find_number_of_modes

from .circuit import Circuit
from .gate import GateFactory
from .state import Qudit

__all__ = [
    "get_xir_library",
    "run_xir_program",
]


Params = Dict[str, Any]
Wires = Dict[str, int]
Stack = Set[str]
GateSignature = Dict[str, Sequence]
StatementGenerator = Callable[[Params, Wires, Stack], Iterator[Statement]]


def get_xir_library() -> XIRProgram:
    """Returns an XIRProgram containing the gate declarations supported by Jet."""
    lines = []

    for name, cls in sorted(GateFactory.registry.items()):
        # Instantiating the Gate subclass (with placeholder parameters) is an
        # easy way to access properties such as the number of wires a Gate can
        # be applied to. The -1 below is a consequence of the fact that the
        # first __init__ parameter, ``self``, is not explicitly passed.
        params = list(signature(cls.__init__).parameters)[1:]
        gate = cls(*[None for _ in params])

        # TODO: Replace gate definitions with gate declarations when parameter
        #       and wire names are supported in gate declarations.
        xir_params = "" if not params else "(" + ", ".join(params) + ")"
        xir_wires = list(range(gate.num_wires))

        line = f"gate {name} {xir_params} {xir_wires}: {name} {xir_params} | {xir_wires}; end;"
        lines.append(line)

    script = "\n".join(lines)

    # TODO: Replace the following line with a call to xir.parse_script() when #30 is merged.
    return XIRTransformer().transform(xir_parser.parse(script))


def run_xir_program(program: XIRProgram) -> List[Union[np.number, np.ndarray]]:
    """Executes an XIR program.

    Raises:
        ValueError: If the given program contains an unsupported gate or output
            statement or an invalid gate definition.

    Args:
        program (XIRProgram): XIR script to execute.

    Returns:
        List of NumPy values representing the output of the XIR program.
    """
    result: List[Union[np.number, np.ndarray]] = []

    num_wires = find_number_of_modes(program)
    # TODO: Extract the Fock cutoff dimension from the XIR script.
    circuit = Circuit(num_wires=num_wires, dim=2)

    for stmt in _expand_xir_program_statements(program):
        if stmt.name in GateFactory.registry:
            # TODO: Automatically insert the Fock cutoff dimension for CV gates.
            gate = GateFactory.create(stmt.name, **stmt.params)
            circuit.append_gate(gate, wire_ids=stmt.wires)

        elif stmt.name == "amplitude" or stmt.name == "Amplitude":
            if "state" not in stmt.params:
                raise ValueError(f"Statement '{stmt}' is missing a 'state' parameter.")

            # TODO: Use a list representation of the "state" key.
            state = list(map(int, bin(stmt.params["state"])[2:].zfill(num_wires)))

            if len(state) != num_wires:
                raise ValueError(f"Statement '{stmt}' has a 'state' parameter which is too large.")

            if stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_amplitude(circuit=circuit, state=state)
            result.append(output)

        else:
            raise ValueError(f"Statement '{stmt}' is not supported.")

    return result


def _expand_xir_program_statements(program: XIRProgram) -> Iterator[Statement]:
    # TODO: Merge the two XIRPrograms (once merging is implemented) and use
    #       gate declarations when parameter and wire names are supported.
    gate_signature_map = {**get_xir_library().gates, **program.gates}

    # Create a mapping from gate names to XIR statement generators.
    stmt_generator_map = {}

    for name in GateFactory.registry:
        stmt_generator_map[name] = _create_statement_generator_for_terminal_gate(name=name)

    for name in program.gates:
        stmt_generator_map[name] = _create_statement_generator_for_composite_gate(
            name=name, stmt_generator_map=stmt_generator_map, gate_signature_map=gate_signature_map
        )

    for stmt in program.statements:
        if stmt.name in stmt_generator_map:
            params = _bind_statement_params(gate_signature_map, stmt)
            wires = _bind_statement_wires(gate_signature_map, stmt)
            yield from stmt_generator_map[stmt.name](params, wires, set())

        else:
            yield stmt


def _create_statement_generator_for_terminal_gate(name: str) -> StatementGenerator:
    """Returns a statement generator for a terminal (Jet) gate.

    Args:
        name (str): Name of the terminal gate.

    Returns:
        StatementGenerator: Function that yields a sequence ``xir.Statement``
            objects which implement the given terminal gate.
    """

    def generator(params: Params, wires: Wires, _: Stack) -> Iterator[Statement]:
        # The ``Statement`` constructor expects ``wires`` to be a list.
        wires_tuple = tuple(wires[i] for i in sorted(wires, key=int))
        yield Statement(name=name, params=params, wires=wires_tuple)

    return generator


def _create_statement_generator_for_composite_gate(
    name: str,
    stmt_generator_map: Dict[str, StatementGenerator],
    gate_signature_map: Dict[str, GateSignature],
) -> StatementGenerator:
    """Creates a statement generator for a composite (user-defined) gate.

    Args:
        name (str): Name of the terminal gate.
        stmt_generator_map (Dict): Map which associates gate names with statement generators.
        gate_signature_map (Dict): Map which associates gate names with gate signatures.

    Returns:
        StatementGenerator: Function that yields a sequence ``xir.Statement``
            objects which implement the given composite gate.
    """

    def generator(params: Params, wires: Wires, stack: Stack) -> Iterator[Statement]:
        if name in stack:
            raise ValueError(f"Gate '{name}' has a circular dependency.")

        for stmt in gate_signature_map[name]["statements"]:
            stmt_params = _bind_statement_params(gate_signature_map, stmt=stmt)
            stmt_wires = _bind_statement_wires(gate_signature_map, stmt=stmt)

            # Replace the value of each (applicable) parameter or wire in the
            # current statement with the value mapped to the corresponding
            # parameter in the signature of this composite gate.
            eval_params = {key: params.get(val, val) for key, val in stmt_params.items()}
            eval_wires = {key: wires.get(val, val) for key, val in stmt_wires.items()}

            # Insert the name of this composite gate into the stack to detect circular dependencies.
            yield from stmt_generator_map[stmt.name](eval_params, eval_wires, stack | {name})

    return generator


def _bind_statement_params(gate_signature_map: Dict[str, GateSignature], stmt: Statement) -> Params:
    """Binds the parameters of a statement to the parameters of its gate.

    Args:
        gate_signature_map (Dict): Map which associates gate names with gate signatures.
        stmt (Statement): Statement whose parameters are to be bound.

    Returns:
        Params: Map which associates the names of the parameters of the gate in
            the statement with with the values of the parameters in the statement.
    """
    if stmt.name not in gate_signature_map:
        raise ValueError(f"Statement '{stmt}' applies a gate which has not been defined.")

    have_params = stmt.params
    want_params = gate_signature_map[stmt.name]["params"]

    if isinstance(have_params, list):
        if len(have_params) != len(want_params):
            raise ValueError(f"Statement '{stmt}' has the wrong number of parameters.")
        return {name: have_params[i] for (i, name) in enumerate(want_params)}

    else:
        if set(have_params) != set(want_params):
            raise ValueError(f"Statement '{stmt}' has an invalid set of parameters.")
        return {name: have_params[name] for name in want_params}


def _bind_statement_wires(gate_signature_map: Dict[str, GateSignature], stmt: Statement) -> Wires:
    """Binds the wires of a statement to the wires of its gate.

    Args:
        gate_signature_map (Dict): Map which associates gate names with gate signatures.
        stmt (Statement): Statement whose wires are to be bound.

    Returns:
        Params: Map which associates the names of the wires of the gate in the
            statement with the with the values of the wires in the statement.
    """
    if stmt.name not in gate_signature_map:
        raise ValueError(f"Statement '{stmt}' applies a gate which has not been defined.")

    have_wires = stmt.wires
    want_wires = gate_signature_map[stmt.name]["wires"]

    if len(have_wires) != len(want_wires):
        raise ValueError(f"Statement '{stmt}' has the wrong number of wires.")

    return {name: have_wires[i] for (i, name) in enumerate(want_wires)}


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
