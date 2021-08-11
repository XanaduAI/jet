import pytest

import jet


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
class TestTensorNetwork:
    def test_constructor(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        with pytest.raises(IndexError):
            tn.nodes[0]

        assert tn.num_tensors == 0
        assert tn.path == []

    def test_add_tensor(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        a = jet.Tensor(shape=[1, 1], indices=["a", "b"], dtype=dtype)

        a_id = tn.add_tensor(a, ["A"])

        assert tn.nodes[a_id].tensor == a
        assert tn.nodes[a_id].tags == ["A"]

    def test_tag_to_nodes_map(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        a = jet.Tensor(dtype=dtype)
        b = jet.Tensor(dtype=dtype)

        a_id = tn.add_tensor(a, ["a", "shared_tag"])
        b_id = tn.add_tensor(b, ["b", "shared_tag"])

        assert tn.tag_to_node_id_map["a"] == [a_id]
        assert tn.tag_to_node_id_map["b"] == [b_id]
        assert set(tn.tag_to_node_id_map["shared_tag"]) == set((a_id, b_id))

    def test_contract_implicit(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        a1 = jet.Tensor(indices=["A"], shape=[3], data=[0, 1, 2], dtype=dtype)
        a2 = jet.Tensor(indices=["A"], shape=[3], data=[0, 1, 2], dtype=dtype)

        tn.add_tensor(a1)
        tn.add_tensor(a2)

        result = tn.contract()

        assert result.data == [5]
        assert tn.path == [(0, 1)]

    def test_contract_explicit(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        a1 = jet.Tensor(indices=["A"], shape=[3], data=[0, 1, 2], dtype=dtype)
        a2 = jet.Tensor(indices=["A"], shape=[3], data=[0, 1, 2], dtype=dtype)

        tn.add_tensor(a1)
        tn.add_tensor(a2)

        result = tn.contract([[0, 1]])

        assert result.data == [5]
        assert tn.path == [(0, 1)]

    def test_slice(self, dtype):
        tn = jet.TensorNetwork(dtype=dtype)

        a = jet.Tensor(indices=["A0", "B1", "C2"], shape=[2, 3, 4], data=range(24), dtype=dtype)
        b = jet.Tensor(indices=["D3"], shape=[2], data=[0, 1], dtype=dtype)

        tn.add_tensor(a)
        tn.add_tensor(b)

        tn.slice_indices(["D3"], 0)

        node = tn.nodes[0]

        assert node.name == "A0B1C2"
        assert node.indices == ["A0", "B1", "C2"]
        assert node.tensor.indices == node.indices
        assert node.tensor.shape == [2, 3, 4]
        assert node.tensor.data == list(range(24))
