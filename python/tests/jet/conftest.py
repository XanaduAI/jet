import pytest

import jet

# Jet types associated with a 64-bit complex data type.
JET_COMPLEX_64_TYPES = (
    jet.TaskBasedCpuContractorC64,
    jet.Tensor32,
    jet.TensorNetwork32,
    jet.TensorNetworkFile32,
    jet.TensorNetworkSerializer32,
)

# Jet types associated with a 128-bit complex data type.
JET_COMPLEX_128_TYPES = (
    jet.TaskBasedCpuContractorC128,
    jet.Tensor64,
    jet.TensorNetwork64,
    jet.TensorNetworkFile64,
    jet.TensorNetworkSerializer64,
)


@pytest.fixture
def tensor_network(get_tensor_type, get_tensor_network_type):
    def tensor_network_(JetType: type):
        """Returns a tensor network with three tensors that is compatible with the given Jet type."""
        Tensor = get_tensor_type(JetType)
        TensorNetwork = get_tensor_network_type(JetType)

        A = Tensor(shape=[2, 2], indices=["i", "j"], data=[1, 1j, -1j, 1])
        B = Tensor(shape=[2, 2], indices=["j", "k"], data=[1, 0, 0, 1])
        C = Tensor(shape=[2], indices=["k"], data=[1, 0])

        tn = TensorNetwork()
        tn.add_tensor(A, ["A", "Hermitian"])
        tn.add_tensor(B, ["B", "Identity", "Real"])
        tn.add_tensor(C, ["C", "Vector", "Real"])
        return tn

    return tensor_network_


@pytest.fixture
def get_tensor_type():
    def get_tensor_type_(JetType: type) -> type:
        """Returns the tensor type associated with the given Jet type."""
        if JetType in JET_COMPLEX_64_TYPES:
            return jet.Tensor32
        elif JetType in JET_COMPLEX_128_TYPES:
            return jet.Tensor64
        else:
            raise Exception(f"Unknown Jet type '{JetType}'.")

    return get_tensor_type_


@pytest.fixture
def get_tensor_network_type():
    def get_tensor_network_type_(JetType: type) -> type:
        """Returns the tensor network type associated with the given Jet type."""
        if JetType in JET_COMPLEX_64_TYPES:
            return jet.TensorNetwork32
        elif JetType in JET_COMPLEX_128_TYPES:
            return jet.TensorNetwork64
        else:
            raise Exception(f"Unknown Jet type '{JetType}'.")

    return get_tensor_network_type_
