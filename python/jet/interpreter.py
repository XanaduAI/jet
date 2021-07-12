from copy import deepcopy
from inspect import signature
from typing import List, Union
import random

import numpy as np

from xir import XIRProgram, XIRTransformer, xir_parser

from .bindings import PathInfo
from .circuit import Circuit
from .factory import TaskBasedContractor, TensorNetworkType
from .gate import GateFactory
from .state import Qudit

__all__ = [
    "get_xir_library",
    "run_xir_program",
]


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

    for key, cls in sorted(GateFactory.registry.items()):
        # Instantiating the Gate subclass (with placeholder parameters) is an
        # easy way to access properties such as the number of wires a Gate can
        # be applied to. The -1 below is a consequence of the fact that the
        # first __init__ parameter, ``self``, is not explicitly passed.
        num_params = len(signature(cls.__init__).parameters) - 1
        gate = GateFactory.create(key, *[None for _ in range(num_params)])

        line = f"gate {key}, {num_params}, {gate.num_wires};"
        lines.append(line)

    script = "\n".join(lines)

    # TODO: Replace the following line with a call to xir.parse_script() when #30 is merged.
    return XIRTransformer().transform(xir_parser.parse(script))


def run_xir_program(program: XIRProgram) -> List[Union[np.number, np.ndarray]]:
    """Executes an XIR program.

    Raises:
        ValueError: If the given program contains an unsupported or invalid statement.

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

        amplitude(state: 0) | [0, 1];
        amplitude(state: 1) | [0, 1];
        amplitude(state: 2) | [0, 1];
        amplitude(state: 3) | [0, 1];

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

    num_wires = len(program.wires)
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
                raise ValueError(f"Statement '{stmt}' has a 'state' parameter which is too large.")

            if stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_amplitude(circuit=circuit, state=state)
            result.append(output)

        elif name in ("Probabilities", "probabilities"):
            if stmt.wires != tuple(range(num_wires)):
                raise ValueError(f"Statement '{stmt}' must be applied to [0 .. {num_wires - 1}].")

            output = _compute_probabilities(circuit=circuit)
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
        Number: NumPy number representing the amplitude of the given state.
    """
    # Do not modify the original circuit.
    circuit = deepcopy(circuit)

    for (i, value) in enumerate(state):
        # Fill a state vector with zeros save for a one at index `value`.
        data = (np.arange(circuit.dimension) == value).astype(np.complex128)
        qudit = Qudit(dim=circuit.dimension, data=data)
        circuit.append_state(qudit, wire_ids=[i])

    tn = circuit.tensor_network(dtype=dtype)
    path_info = _find_contraction_path(tn=tn)

    tbc = TaskBasedContractor(dtype=dtype)
    tbc.add_contraction_tasks(tn=tn, path_info=path_info)
    tbc.add_deletion_tasks()
    tbc.contract()

    amplitude = tn.contract()
    return dtype(amplitude.scalar)


def _compute_probabilities(circuit: Circuit, dtype: type = np.complex128) -> np.ndarray:
    """Computes the probability distribution at the end of a circuit.

    Args:
        circuit (Circuit): Circuit to produce the probability distribution for.
        dtype (type): Data type of the probability computation.

    Returns:
        Array: NumPy array representing the probability of measuring each basis state.
    """
    tn = circuit.tensor_network(dtype=dtype)

    if len(tn.nodes) == 1:
        # No contractions are necessary for a single tensor.
        amplitudes = np.array(tn.nodes[0].tensor.data)

    else:
        # Contract all the tensors (not just the ones with shared indices).
        # TODO: Use a better contraction path algorithm.
        path = [(i, i + 1) for i in range(0, 2 * len(tn.nodes) - 3, 2)]
        path_info = PathInfo(tn=tn, path=path)

        tbc = TaskBasedContractor(dtype=dtype)
        tbc.add_contraction_tasks(tn=tn, path_info=path_info)
        tbc.add_deletion_tasks()
        tbc.contract()

        # Arrange the indices in increasing order of wire ID.
        state = tbc.results[0].transpose([wire.index for wire in circuit.wires])
        amplitudes = np.array(state.data).flatten()

    return amplitudes.conj() * amplitudes


def _find_contraction_path(tn: TensorNetworkType, samples: int = 100) -> PathInfo:
    """Finds a contraction path for a connected tensor network. This is done by
    sampling several random contraction paths and then choosing the one which
    minimizes the number of FLOPS.

    Args:
        tn (TensorNetwork): Tensor network to be contracted.
        samples (int): Number of contraction paths to sample.

    Returns:
        PathInfo: Contraction path for the given tensor network.
    """
    paths = (_sample_contraction_path(tn=tn) for _ in range(samples))
    return min(paths, key=lambda path: path.total_flops())


def _sample_contraction_path(tn: TensorNetworkType) -> PathInfo:
    """Samples a random contraction path for a connected tensor network.

    Args:
        tn (TensorNetwork): Tensor network to be contracted.

    Returns:
        PathInfo: Contraction path for the given tensor network.
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

    # Contract a random edge in the adjacency list during each iteration.
    while len(neighbours) > 1:
        # Choose two nodes (tensors) to contract.
        node_id_1 = random.choice(tuple(neighbours))
        node_id_2 = random.choice(tuple(neighbours[node_id_1]))
        path.append((node_id_1, node_id_2))

        # Derive the neighbours of the contracted node.
        node_id_3 = max(neighbours) + 1
        neighbours[node_id_3] = neighbours[node_id_1] | neighbours[node_id_2]
        neighbours[node_id_3] -= {node_id_1, node_id_2}

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

    return PathInfo(tn=tn, path=path)
