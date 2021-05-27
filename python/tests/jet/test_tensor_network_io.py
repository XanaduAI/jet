import re

import pytest

import jet


@pytest.mark.parametrize("TensorNetworkFile", [jet.TensorNetworkFileC64, jet.TensorNetworkFileC128])
def test_tensor_network_file(TensorNetworkFile):
    """Tests that a tensor network file can be constructed."""
    tnf = TensorNetworkFile()
    assert tnf.path is None
    assert len(tnf.tensors.nodes) == 0


@pytest.mark.parametrize(
    "Tensor, TensorNetwork, TensorNetworkSerializer",
    [
        (
            jet.TensorC64,
            jet.TensorNetworkC64,
            jet.TensorNetworkSerializerC64,
        ),
        (
            jet.TensorC128,
            jet.TensorNetworkC128,
            jet.TensorNetworkSerializerC128,
        ),
    ],
)
class TestTensorNetworkSerializer:
    @pytest.fixture
    def tensor_network(self, Tensor, TensorNetwork):
        """Returns a tensor network with three tensors of the given type."""
        A = Tensor(shape=[2, 2], indices=["i", "j"], data=[1, 1j, -1j, 1])
        B = Tensor(shape=[2, 2], indices=["j", "k"], data=[1, 0, 0, 1])
        C = Tensor(shape=[2], indices=["k"], data=[1, 0])

        tn = TensorNetwork()
        tn.add_tensor(A, ["A", "Hermitian"])
        tn.add_tensor(B, ["B", "Identity", "Real"])
        tn.add_tensor(C, ["C", "Vector", "Real"])
        return tn

    @pytest.fixture
    def serialized_tensor_network(self) -> str:
        """Returns a serialized tensor network file representing a tensor
        network.
        """
        return """
        {
            "tensors": [
                [
                    ["A", "Hermitian"],
                    ["i", "j"],
                    [2, 2],
                    [[1.0, 0.0], [0.0, 1.0], [-0.0, -1.0], [1.0, 0.0]]
                ], [
                    ["B", "Identity", "Real"],
                    ["j", "k"],
                    [2, 2],
                    [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
                ], [
                    ["C", "Vector", "Real"],
                    ["k"],
                    [2],
                    [[1.0, 0.0], [0.0, 0.0]]
                ]
            ]
        }
        """

    @pytest.fixture
    def serialized_tensor_network_and_path(self) -> str:
        """Returns a serialized tensor network file representing a tensor
        network and a contraction path.
        """
        return """
        {
            "path": [[0, 2], [2, 1]],
            "tensors": [
                [
                    ["A", "Hermitian"],
                    ["i", "j"],
                    [2, 2],
                    [[1.0, 0.0], [0.0, 1.0], [-0.0, -1.0], [1.0, 0.0]]
                ], [
                    ["B", "Identity", "Real"],
                    ["j", "k"],
                    [2, 2],
                    [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
                ], [
                    ["C", "Vector", "Real"],
                    ["k"],
                    [2],
                    [[1.0, 0.0], [0.0, 0.0]]
                ]
            ]
        }
        """

    def test_serialize_tensor_network(
        self, tensor_network, TensorNetworkSerializer, serialized_tensor_network
    ):
        """Tests that a tensor network file representing a tensor network can be
        serialized.
        """
        have_json = TensorNetworkSerializer()(tensor_network)
        want_json = re.sub(r"\s+", "", serialized_tensor_network)
        assert have_json == want_json

    def test_serialize_tensor_network_and_path(
        self,
        tensor_network,
        TensorNetworkSerializer,
        serialized_tensor_network_and_path,
    ):
        """Tests that a tensor network file representing a tensor network and a
        contraction path can be serialized.
        """
        path = jet.PathInfo(tn=tensor_network, path=[[0, 2], [2, 1]])
        have_json = TensorNetworkSerializer()(tensor_network, path)
        want_json = re.sub(r"\s+", "", serialized_tensor_network_and_path)
        assert have_json == want_json

    def test_deserialize_tensor_network(
        self, tensor_network, TensorNetworkSerializer, serialized_tensor_network
    ):
        """Tests that a tensor network file representing a tensor network can be
        deserialized.
        """
        tnf = TensorNetworkSerializer()(serialized_tensor_network)
        assert tnf.path is None
        assert len(tnf.tensors.nodes) == len(tensor_network.nodes)

    def test_deserialize_tensor_network_and_path(
        self,
        tensor_network,
        TensorNetworkSerializer,
        serialized_tensor_network_and_path,
    ):
        """Tests that a tensor network file representing a tensor network and a
        contraction path can be deserialized.
        """
        tnf = TensorNetworkSerializer()(serialized_tensor_network_and_path)
        assert tnf.path is not None
        assert tnf.path.path == [(0, 2), (2, 1)]
        assert len(tnf.tensors.nodes) == len(tensor_network.nodes)
