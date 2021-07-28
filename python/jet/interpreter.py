import random
import warnings
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Sequence, Set, Union

import numpy as np

from xir import OperatorStmt, Statement, XIRProgram
from xir import parse_script as parse_xir_script

from .bindings import PathInfo
from .circuit import Circuit, Operation
from .factory import TaskBasedContractor, TensorNetworkType, TensorType
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
        # There is no need to mock parameters that have default values; the [1:]
        # is required because self is not explicitly passed to cls.__init__().
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

        elif stmt.name in ("Amplitude", "amplitude"):
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

        elif stmt.name in ("Probabilities", "probabilities"):
            if stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_probabilities(circuit=circuit)
            result.append(output)

        elif stmt.name in ("Expval", "expval"):
            if not isinstance(stmt.params, dict) or "observable" not in stmt.params:
                raise ValueError(f"Statement '{stmt}' is missing an 'observable' parameter.")

            operator = stmt.params["observable"]

            if operator not in program.operators:
                raise ValueError(
                    f"Statement '{stmt}' has an 'observable' parameter which "
                    f"references an undefined operator."
                )

            elif program.operators[operator]["params"]:
                raise ValueError(
                    f"Statement '{stmt}' has an 'observable' parameter which "
                    f"references a parameterized operator."
                )

            elif stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            observable = _generate_observable_from_operation_statements(
                program.operators[operator]["statements"]
            )

            output = _compute_expected_value(circuit=circuit, observable=observable)
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
    # A deep copy is not needed since the value of each option will not be changed.
    options = program.options.copy()

    if "dimension" in options:
        dimension = options.pop("dimension")

        if not isinstance(dimension, int):
            raise ValueError("Option 'dimension' must be an integer.")

        elif dimension < 2:
            raise ValueError("Option 'dimension' must be greater than one.")

    for option in sorted(options):
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


def _generate_observable_from_operation_statements(
    stmts: Iterator[OperatorStmt],
) -> Iterator[Operation]:
    """Generates an observable from a series of operator statements.

    Args:
        stmts (Iterator[OperatorStmt]): Operator statements defining the observable.

    Returns:
        Iterator[Operation]: Iterator over a sequence of ``Operation`` objects
            which implement the observable given by the operator statements.
    """
    for stmt in stmts:
        try:
            scalar = float(stmt.pref)
        except ValueError:
            raise ValueError(
                f"Operator statement '{stmt}' has a prefactor ({stmt.pref}) "
                f"which cannot be converted to a floating-point number."
            )

        for gate_name, wire_id in stmt.terms:
            gate = GateFactory.create(gate_name, scalar=scalar)
            yield Operation(part=gate, wire_ids=[wire_id])


def _compute_amplitude(
    circuit: Circuit, state: List[int], dtype: np.dtype = np.complex128
) -> np.number:
    """Computes the amplitude of a state at the end of a circuit.

    Args:
        circuit (Circuit): Circuit to apply the amplitude measurement to.
        state (list[int]): State to measure the amplitude of.
        dtype (np.dtype): Data type of the amplitude.

    Returns:
        Number: NumPy number representing the amplitude of the given state.
    """
    # Do not modify the original circuit.
    circuit = deepcopy(circuit)

    for (i, value) in enumerate(state):
        # Fill a state vector with zeros save for a one at index `value`.
        data = (np.arange(circuit.dimension) == value).astype(np.complex128)
        qudit = Qudit(dim=circuit.dimension, data=data)
        circuit.append_state(qudit, wire_ids=[i])

    amplitude = _simulate(circuit=circuit, dtype=dtype)
    return dtype(amplitude.scalar)


def _compute_probabilities(circuit: Circuit, dtype: np.dtype = np.complex128) -> np.ndarray:
    """Computes the probability distribution at the end of a circuit.

    Args:
        circuit (Circuit): Circuit to produce the probability distribution for.
        dtype (np.dtype): Data type of the probability computation.

    Returns:
        Array: NumPy array representing the probability of measuring each basis state.
    """
    tensor = _simulate(circuit=circuit, dtype=dtype)

    # Arrange the indices in increasing order of wire ID.
    state = tensor.transpose([wire.index for wire in circuit.wires])

    amplitudes = np.array(state.data).flatten()
    return amplitudes.conj() * amplitudes


def _compute_expected_value(
    circuit: Circuit, observable: Iterator[Operation], dtype: np.dtype = np.complex128
) -> np.number:
    """Computes the expected value of an observable with respect to a circuit.

    Args:
        circuit (Circuit): Circuit to apply the expectation measurement to.
        observable (Observable): Observable to take the expected value of.
        dtype (np.dtype): Data type of the expected value.

    Returns:
        Number: NumPy number representing the expected value.
    """
    # Do not modify the original circuit.
    circuit = deepcopy(circuit)

    circuit.take_expected_value(observable)

    expval = _simulate(circuit=circuit, dtype=dtype)
    return dtype(expval.scalar)


def _simulate(circuit: Circuit, dtype: np.dtype = np.complex128) -> TensorType:
    """Simulates a circuit using the task-based contractor.

    Args:
        circuit (Circuit): Circuit to simulate.
        dtype (np.dtype): Data type of the tensor network to contract.

    Returns:
        Tensor: Result of the simulation.
    """
    tn = circuit.tensor_network(dtype=dtype)

    if len(tn.nodes) == 1:
        # No contractions are necessary for a single tensor.
        return tn.nodes[0].tensor

    path_info = _find_contraction_path(tn=tn)

    tbc = TaskBasedContractor(dtype=dtype)
    tbc.add_contraction_tasks(tn=tn, path_info=path_info)
    tbc.add_deletion_tasks()

    # Warning: The call to contract() below can take a while depending on the
    # size of the tensor network and the quality of the contraction path.
    tbc.contract()

    return tbc.results[0]


def _find_contraction_path(tn: TensorNetworkType, samples: int = 100) -> PathInfo:
    """Finds a contraction path for a tensor network. This is done by sampling
    several random contraction paths and choosing the one which minimizes the
    total number of FLOPS.

    Args:
        tn (TensorNetwork): Tensor network to be contracted.
        samples (int): Number of contraction paths to sample.

    Returns:
        PathInfo: Contraction path for the given tensor network.
    """
    paths = (_sample_contraction_path(tn=tn) for _ in range(samples))
    return min(paths, key=lambda path: path.total_flops())


def _sample_contraction_path(tn: TensorNetworkType) -> PathInfo:
    """Samples a random contraction path for a tensor network.

    Args:
        tn (TensorNetwork): Tensor network to be contracted.

    Returns:
        PathInfo: Contraction path for the given tensor network. Contractions
        between nodes that share an index are always preferred.
    """
    path = []

    # Caching the index-to-edge map improves performance significantly.
    index_to_edge_map = tn.index_to_edge_map

    # Build an adjacency list using the node IDs in the tensor network.
    neighbours = {node.id: set() for node in tn.nodes}
    for node in tn.nodes:
        for index in node.indices:
            node_ids = index_to_edge_map[index].node_ids
            neighbours[node.id].update(node_ids)

        # Nodes are not neighbours with themselves.
        neighbours[node.id].remove(node.id)

    # Track the set of tensors with no neighbours.
    isolated = list(sorted(node_id for node_id in neighbours if not neighbours[node_id]))

    for node_id in isolated:
        del neighbours[node_id]

    # Iteratively contract two adjacent tensors while it is possible to do so.
    while len(neighbours) > 1:
        node_id_1 = random.choice(tuple(neighbours))
        node_id_2 = random.choice(tuple(neighbours[node_id_1]))
        path.append((node_id_1, node_id_2))

        # Derive the neighbours of the contracted node.
        node_id_3 = max(*neighbours, *isolated) + 1
        neighbours[node_id_3] = neighbours[node_id_1] | neighbours[node_id_2]
        neighbours[node_id_3] -= {node_id_1, node_id_2}

        # Decide whether the contracted node is isolated.
        if not neighbours[node_id_3]:
            isolated.append(node_id_3)
            del neighbours[node_id_3]

        # Replace node_id_1 with node_id_3 and then do the same for node_id_2.
        for node_id in neighbours[node_id_1]:
            if node_id != node_id_2:
                neighbours[node_id].remove(node_id_1)
                neighbours[node_id].add(node_id_3)

        for node_id in neighbours[node_id_2]:
            if node_id != node_id_1:
                neighbours[node_id].remove(node_id_2)
                neighbours[node_id].add(node_id_3)

        del neighbours[node_id_1]
        del neighbours[node_id_2]

    if len(isolated) > 1:
        # The final iteration of the while loop always yields an isolated node.
        # The ID of this node is always the largest ID in the tensor network.
        node_id_1 = isolated[-1]
        for node_id_2 in isolated[:-1]:
            path.append((node_id_1, node_id_2))
            node_id_1 += 1

    return PathInfo(tn=tn, path=path)
