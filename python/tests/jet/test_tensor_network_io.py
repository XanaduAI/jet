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
    "TensorNetworkSerializer",
    [
        jet.TensorNetworkSerializerC64,
        jet.TensorNetworkSerializerC128,
    ],
)
class TestTensorNetworkSerializer:
    @pytest.fixture
    def tensor_network(self):
        def tensor_network_(dtype: str):
            """Returns a tensor network with three tensors of the given type."""
            A = jet.Tensor(dtype=dtype, shape=[2, 2], indices=["i", "j"], data=[1, 1j, -1j, 1])
            B = jet.Tensor(dtype=dtype, shape=[2, 2], indices=["j", "k"], data=[1, 0, 0, 1])
            C = jet.Tensor(dtype=dtype, shape=[2], indices=["k"], data=[1, 0])

            tn = jet.TensorNetwork(dtype=dtype)
            tn.add_tensor(A, ["A", "Hermitian"])
            tn.add_tensor(B, ["B", "Identity", "Real"])
            tn.add_tensor(C, ["C", "Vector", "Real"])
            return tn

        return tensor_network_

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
        tn = tensor_network(dtype=TensorNetworkSerializer.dtype)
        have_json = TensorNetworkSerializer()(tn)
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
        tn = tensor_network(dtype=TensorNetworkSerializer.dtype)
        path = jet.PathInfo(tn=tn, path=[[0, 2], [2, 1]])
        have_json = TensorNetworkSerializer()(tn, path)
        want_json = re.sub(r"\s+", "", serialized_tensor_network_and_path)
        assert have_json == want_json

    def test_deserialize_tensor_network(
        self, tensor_network, TensorNetworkSerializer, serialized_tensor_network
    ):
        """Tests that a tensor network file representing a tensor network can be
        deserialized.
        """
        tn = tensor_network(dtype=TensorNetworkSerializer.dtype)
        tnf = TensorNetworkSerializer()(serialized_tensor_network)
        assert tnf.path is None
        assert len(tnf.tensors.nodes) == len(tn.nodes)

    def test_deserialize_tensor_network_and_path(
        self,
        tensor_network,
        TensorNetworkSerializer,
        serialized_tensor_network_and_path,
    ):
        """Tests that a tensor network file representing a tensor network and a
        contraction path can be deserialized.
        """
        tn = tensor_network(dtype=TensorNetworkSerializer.dtype)
        tnf = TensorNetworkSerializer()(serialized_tensor_network_and_path)
        assert tnf.path is not None
        assert tnf.path.path == [(0, 2), (2, 1)]
        assert len(tnf.tensors.nodes) == len(tn.nodes)
