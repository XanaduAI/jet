import jet
import timeit

# Load the Sycamore (m=10) circuit from a JSON file.
with open("m10.json", "r") as f:
    m10_json = f.read()

# The __call__ operator deserializes a circuit into a "tensor network file" object.
tnf = jet.TensorNetworkSerializer()(m10_json)

# Choose a set of index labels to slice; some sets reduce latency better than others.
index_labels_to_slice = ["p7", "s7", "h4", "m1", "m2", "I2"]

# The TBCC uses task-based parallelism to accelerate tensor network contractions.
tbcc = jet.TaskBasedCpuContractor()

for index in range(1 << len(index_labels_to_slice)):
    # Support for cloning tensor networks using deepcopy() is coming soon.
    tn = jet.TensorNetwork()
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