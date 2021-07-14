from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Sequence, Set, Union
import warnings

import numpy as np

from xir import Statement, XIRProgram
from xir import parse_script as parse_xir_script

from .circuit import Circuit
from .gate import FockGate, GateFactory
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
    """Returns an ``XIRProgram`` containing the gate declarations supported by Jet.

    Returns:
        xir.program.XIRProgram: Declarations of the gates supported by Jet.

    **Example**

    Simply call the function directly:

    .. code-block:: python

        import jet

        program = jet.get_xir_library()

        assert program.gates
    """
    lines = []

    for name, cls in sorted(GateFactory.registry.items()):
        # Instantiating the Gate subclass (with placeholder parameters) is an
        # easy way to access properties such as the number of wires a Gate can
        # be applied to.
        param_vals = signature(cls.__init__).parameters.values()
        # There is no need to mock parameters with default values; the [1:] is
        # required because self is not explicitly passed.
        param_keys = [param.name for param in param_vals if param.default is param.empty][1:]
        gate = cls(*[None for _ in param_keys])

        # TODO: Replace gate definitions with gate declarations when parameter
        #       and wire names are supported in gate declarations.
        xir_params = "" if not param_keys else "(" + ", ".join(param_keys) + ")"
        xir_wires = list(range(gate.num_wires))

        line = f"gate {name} {xir_params} {xir_wires}: {name} {xir_params} | {xir_wires}; end;"
        lines.append(line)

    script = "\n".join(lines)
    return parse_xir_script(script)


def run_xir_program(program: XIRProgram) -> List[Union[np.number, np.ndarray]]:
    """Executes an XIR program.

    Raises:
        ValueError: If the given program contains an unsupported gate or output
            statement or an invalid gate definition.

    Args:
        program (xir.program.XIRProgram): XIR script to execute.

    Returns:
        List[Union[np.number, np.ndarray]]: List of NumPy values representing the
        output of the XIR program.

    **Example**

    Consider the following XIR script which generates a Bell state and then
    measures the amplitude of each basis state:

    .. code-block:: haskell

        use xstd;

        H | [0];
        CNOT | [0, 1];

        amplitude(state: [0, 0]) | [0, 1];
        amplitude(state: [0, 1]) | [0, 1];
        amplitude(state: [1, 0]) | [0, 1];
        amplitude(state: [1, 1]) | [0, 1];

    If the contents of this script are stored in a string called ``xir_script``,
    then each amplitude of the Bell state can be displayed using Jet as follows:

    .. code-block:: python

        import jet
        import xir

        # Parse the XIR script into an XIR program.
        xir_program = xir.parse_script(xir_script)

        # Run the XIR program using Jet and wait for the results.
        result = jet.run_xir_program(xir_program)

        # Display the returned amplitudes.
        for i, amp in enumerate(result):
            print(f"Amplitude |{i:02b}> = {amp}")
    """
    result: List[Union[np.number, np.ndarray]] = []

    _validate_xir_program_options(program=program)

    num_wires = len(program.wires)
    dimension = program.options.get("dimension", 2)
    circuit = Circuit(num_wires=num_wires, dim=dimension)

    for stmt in _resolve_xir_program_statements(program):
        if stmt.name in GateFactory.registry:
            gate = GateFactory.create(stmt.name, **stmt.params)

            if isinstance(gate, FockGate):
                gate.dimension = circuit.dimension

            if gate.dimension != circuit.dimension:
                raise ValueError(
                    f"Statement '{stmt}' applies a gate with a dimension ({gate.dimension}) "
                    f"that differs from the dimension of the circuit ({circuit.dimension})."
                )

            circuit.append_gate(gate, wire_ids=stmt.wires)

        elif stmt.name == "amplitude" or stmt.name == "Amplitude":
            if not isinstance(stmt.params, dict) or "state" not in stmt.params:
                raise ValueError(f"Statement '{stmt}' is missing a 'state' parameter.")

            state = stmt.params["state"]

            if not isinstance(state, list):
                raise ValueError(
                    f"Statement '{stmt}' has a 'state' parameter which is not an array."
                )

            elif not all(0 <= entry < dimension for entry in state):
                raise ValueError(
                    f"Statement '{stmt}' has a 'state' parameter with at least "
                    f"one entry that falls outside the range [0, {dimension})."
                )

            elif len(state) != num_wires:
                raise ValueError(
                    f"Statement '{stmt}' has a 'state' parameter with "
                    f"{len(state)} (!= {num_wires}) entries."
                )

            elif stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_amplitude(circuit=circuit, state=state)
            result.append(output)

        else:
            raise ValueError(f"Statement '{stmt}' is not supported.")

    return result


def _validate_xir_program_options(program: XIRProgram) -> None:
    """Validates the options in an ``XIRProgram``.

    Args:
        program (XIRProgram): Program with the options to validate.

    Raises:
        ValueError: If the value of at least one option is invalid.
    """
    ignored_options = program.options.copy()

    if "dimension" in ignored_options:
        dimension = ignored_options.pop("dimension")

        if not isinstance(dimension, int):
            raise ValueError("Option 'dimension' must be an integer.")

        elif dimension < 2:
            raise ValueError("Option 'dimension' must be greater than one.")

    for option in ignored_options:
        warnings.warn(f"Option '{option}' is not supported and will be ignored.")


def _resolve_xir_program_statements(program: XIRProgram) -> Iterator[Statement]:
    """Resolves the statements in an ``XIRProgram`` such that each yielded
    gate application ``Statement`` is applied to a registered Jet gate.

    Args:
        program (XIRProgram): Program with the statements to be resolved.

    Returns:
        Iterator[Statement]: Resolved statements in the given ``XIRProgram``.
    """
    # TODO: Merge the two XIRPrograms (once merging is implemented) and use
    #       gate declarations when parameter and wire names are supported.
    gate_signature_map = {**get_xir_library().gates, **program.gates}

    # Create a mapping from gate names to XIR statement generators.
    stmt_generator_map = {}

    for name in GateFactory.registry:
        stmt_generator_map[name] = _create_statement_generator_for_terminal_gate(name=name)

    for name in program.gates:
        if name in stmt_generator_map:
            warnings.warn(f"Gate '{name}' overrides the Jet gate with the same name.")

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
        StatementGenerator: Function that yields a sequence of ``Statement``
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
        StatementGenerator: Function that yields a sequence of ``Statement``
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
            the statement with the values of the parameters in the statement.
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
        Wires: Map which associates the names of the wires of the gate in the
            statement with the values of the wires in the statement.
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
