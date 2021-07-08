import random
import timeit

import jet
import numpy as np

import UnitaryPreperation

# ------------------------------------------------------------------------------

@jet.GateFactory.register(names=["Unitary", "unitary", "u", "U"])
class Unitary(jet.Gate):
    """Unitary represents an arbitrary unitary gate.

    Args:
        num_wires (int): Number of wires the gate is applied to.
        params (array): Array of gate parameters.
    """

    def __init__(self, num_wires: int, params: np.ndarray):
        super().__init__(name="Unitary", num_wires=num_wires, params=params)

    def _data(self) -> np.ndarray:
        return UnitaryPreperation.param_unitary(dim=2**self.num_wires, params=self.params)

# ------------------------------------------------------------------------------

# Defining the parameters of the circuit
depth = 4
n_qubits = 64
loc = 8

num_gates = 2**depth-1

# Generating random parameters for the gates
params = []
for i in range(num_gates):
    params.append(np.random.rand(2**loc, 2**loc))

# ------------------------------------------------------------------------------

circuit = jet.Circuit(dim=2, num_wires=n_qubits)

# Qubits are initialized to |0> but we can easily change this. For example:
for op in circuit.operations:
    op.part._state_vector = np.array([0, 1], dtype=np.complex128)

param_index = 0
for layer in UnitaryPreperation.compute_indices(loc, depth):
    for wire_ids in layer:
        gate = Unitary(num_wires=loc, params=params[param_index])
        circuit.append_gate(gate, wire_ids=wire_ids)
        param_index += 1

# ------------------------------------------------------------------------------

tn = circuit.tensor_network()
print(f"Created a tensor network with {len(tn.nodes)} tensors.")

# ------------------------------------------------------------------------------

tnf_str = jet.TensorNetworkSerializer()(tn)

filename = "tnf.json"
with open(filename, "w") as f:
    f.write(tnf_str)

print(f"Wrote the tensor network to '{filename}'.")

# ------------------------------------------------------------------------------

def sample_contraction_path(tn: jet.TensorNetworkType) -> jet.PathInfo:
    path = []

    # Construct a graph that associates nodes with its neighbouring in the TN.
    graph = {}
    for node in tn.nodes:
        graph[node.id] = set()
        for index in node.indices:
            node_ids = tn.index_to_edge_map[index].node_ids
            graph[node.id].update(node_ids)
        graph[node.id].remove(node.id)

    # Iteratively contract a random edge from the graph.
    while len(graph) > 1:
        node_id_1 = random.choice(tuple(graph))
        node_id_2 = random.choice(tuple(graph[node_id_1]))
        path.append((node_id_1, node_id_2))

        node_id_3 = max(graph) + 1

        for node_ids in graph.values():
            for node_id in (node_id_1, node_id_2):
                if node_id in node_ids:
                    node_ids.remove(node_id)
                    node_ids.add(node_id_3)

        graph[node_id_3] = graph[node_id_1] ^ graph[node_id_2]

        del graph[node_id_1]
        del graph[node_id_2]

    return jet.PathInfo(tn=tn, path=path)

path_info = sample_contraction_path(tn)

# ------------------------------------------------------------------------------

tbc = jet.TaskBasedContractor()
tbc.add_contraction_tasks(tn=tn, path_info=path_info)
tbc.add_deletion_tasks()

print("Starting to contract tensor network. This can take a while...")
t0 = timeit.default_timer()
tbc.contract()
t1 = timeit.default_timer()

# Display the contraction result and duration.
print(f"Got contraction result of size {len(tbc.results[0])} in {t1 - t0:.3f}s.")