import re

import pytest

import jet


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_tensor_network_file(dtype):
    """Tests that a tensor network file can be constructed."""
    tnf = jet.TensorNetworkFile(dtype=dtype)
    assert tnf.path is None
    assert len(tnf.tensors.nodes) == 0


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
class TestTensorNetworkSerializer:
    @pytest.fixture
    def tensor_network(self):
        def tensor_network_(dtype: str):
            """Returns a tensor network with three tensors of the given type."""
            A = jet.Tensor(shape=[2, 2], indices=["i", "j"], data=[1, 1j, -1j, 1], dtype=dtype)
            B = jet.Tensor(shape=[2, 2], indices=["j", "k"], data=[1, 0, 0, 1], dtype=dtype)
            C = jet.Tensor(shape=[2], indices=["k"], data=[1, 0], dtype=dtype)

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

    def test_serialize_tensor_network(self, dtype, tensor_network, serialized_tensor_network):
        """Tests that a tensor network file representing a tensor network can be
        serialized.
        """
        tn = tensor_network(dtype=dtype)
        have_json = jet.TensorNetworkSerializer(dtype=dtype)(tn)
        want_json = re.sub(r"\s+", "", serialized_tensor_network)
        assert have_json == want_json

    def test_serialize_tensor_network_and_path(
        self,
        dtype,
        tensor_network,
        serialized_tensor_network_and_path,
    ):
        """Tests that a tensor network file representing a tensor network and a
        contraction path can be serialized.
        """
        tn = tensor_network(dtype=dtype)
        path = jet.PathInfo(tn=tn, path=[[0, 2], [2, 1]])
        have_json = jet.TensorNetworkSerializer(dtype=dtype)(tn, path)
        want_json = re.sub(r"\s+", "", serialized_tensor_network_and_path)
        assert have_json == want_json

    def test_deserialize_tensor_network(self, dtype, tensor_network, serialized_tensor_network):
        """Tests that a tensor network file representing a tensor network can be deserialized."""
        tn = tensor_network(dtype=dtype)
        tnf = jet.TensorNetworkSerializer(dtype=dtype)(serialized_tensor_network)
        assert tnf.path is None
        assert len(tnf.tensors.nodes) == len(tn.nodes)

    def test_deserialize_tensor_network_and_path(
        self,
        dtype,
        tensor_network,
        serialized_tensor_network_and_path,
    ):
        """Tests that a tensor network file representing a tensor network and a
        contraction path can be deserialized.
        """
        tn = tensor_network(dtype=dtype)
        tnf = jet.TensorNetworkSerializer(dtype=dtype)(serialized_tensor_network_and_path)
        assert tnf.path is not None
        assert tnf.path.path == [(0, 2), (2, 1)]
        assert len(tnf.tensors.nodes) == len(tn.nodes)
