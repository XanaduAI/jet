import jet
import numpy as np
import timeit

# Load the Sycamore (m=10) circuit from a JSON file.
with open("m10.json", "r") as f:
    m10_json = f.read()

# Currently, Jet supports 64-bit and 128-bit complex data types.
dtype = np.dtype(np.complex64)

# The __call__ operator deserializes a circuit into a "tensor network file" object.
tnf = jet.TensorNetworkSerializer(dtype=dtype)(m10_json)

num_tensors = tnf.tensors.num_tensors
num_indices = tnf.tensors.num_indices
print(f"Loaded tensor network with {num_tensors} tensors and {num_indices} indices.")

# The memory reported by a contraction path assumes single-byte tensor elements.
num_steps = len(tnf.path.steps)
num_bytes = int(tnf.path.total_memory()) * dtype.itemsize
print(f"Loaded contraction path with {num_steps} steps that uses {num_bytes / 10**9:.1f}GB of memory.")

# Choose a set of index labels to slice; some sets reduce latency better than others.
index_labels_to_slice = ["p7", "s7", "h4", "m1", "m2", "I2"]

# The TBCC uses task-based parallelism to accelerate tensor network contractions.
tbcc = jet.TaskBasedCpuContractor(dtype=dtype)

for index in range(1 << len(index_labels_to_slice)):
    # Support for cloning tensor networks using deepcopy() is coming soon.
    tn = jet.TensorNetwork(dtype=dtype)
    for node in tnf.tensors.nodes:
        tn.add_tensor(node.tensor)

    # Slice the tensor network along the current lexicographic index.
    tn.slice_indices(index_labels_to_slice, index)

    # Contraction tasks transform tensor networks into scalar values.
    path = jet.PathInfo(tn, tnf.path.path)
    tbcc.add_contraction_tasks(tn, path)

# Deletion tasks save memory by deallocating tensors when they are no longer needed.
tbcc.add_deletion_tasks()

# Reduction tasks sum the scalars at the end of each tensor network contraction.
tbcc.add_reduction_task()

# TaskBasedCpuContractor.contract() is a blocking call.
print("Starting to contract tensor network. This can take a while...")
t0 = timeit.default_timer()
tbcc.contract()
t1 = timeit.default_timer()

# Display the contraction result and duration.
print(f"Got contraction result {tbcc.reduction_result.scalar:.3e} in {t1 - t0:.3f}s.")