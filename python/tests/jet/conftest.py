import pytest

import jet


@pytest.fixture
def tensor_network():
    """Returns a function that contructs a tensor network with three connected tensors."""

    def tensor_network_(dtype: str):
        A = jet.Tensor(shape=[2, 2], indices=["i", "j"], data=[1, 1j, -1j, 1], dtype=dtype)
        B = jet.Tensor(shape=[2, 2], indices=["j", "k"], data=[1, 0, 0, 1], dtype=dtype)
        C = jet.Tensor(shape=[2], indices=["k"], data=[1, 0], dtype=dtype)

        tn = jet.TensorNetwork(dtype=dtype)
        tn.add_tensor(A, ["A", "Hermitian"])
        tn.add_tensor(B, ["B", "Identity", "Real"])
        tn.add_tensor(C, ["C", "Vector", "Real"])
        return tn

    return tensor_network_
