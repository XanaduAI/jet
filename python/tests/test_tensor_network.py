import jet
import pytest


@pytest.mark.parametrize(
    "TensorNetwork, Tensor",
    [(jet.TensorNetwork32, jet.Tensor32), (jet.TensorNetwork64, jet.Tensor64)],
)
class TestTensorNetwork:
    def test_constructor(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        with pytest.raises(IndexError):
            node = tn.nodes[0]

        assert tn.num_tensors == 0
        assert tn.path == []

    def test_add_tensor(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor(shape=[1, 1], indices=["a", "b"])

        a_id = tn.add_tensor(a, ["A"])

        assert tn.nodes[a_id].tensor == a
        assert tn.nodes[a_id].tags == ["A"]

    def test_get_node_ids_by_tag(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor()
        b = Tensor()

        a_id = tn.add_tensor(a, ["a", "shared_tag"])
        b_id = tn.add_tensor(b, ["b", "shared_tag"])

        tn.get_node_ids_by_tag("a") == [a_id]
        tn.get_node_ids_by_tag("b") == [b_id]

        assert set(tn.get_node_ids_by_tag("shared_tag")) == {a_id, b_id}

    def test_contract_implicit(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a1 = Tensor(indices=["A"], shape=[3], data=[0, 1, 2])
        a2 = Tensor(indices=["A"], shape=[3], data=[0, 1, 2])

        tn.add_tensor(a1, [])
        tn.add_tensor(a2, [])

        result = tn.contract([])

        assert result.data == [5]
        assert tn.path == [(0, 1)]

    def test_contract_explicit(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a1 = Tensor(indices=["A"], shape=[3], data=[0, 1, 2])
        a2 = Tensor(indices=["A"], shape=[3], data=[0, 1, 2])

        tn.add_tensor(a1, [])
        tn.add_tensor(a2, [])

        result = tn.contract([[0, 1]])

        assert result.data == [5]
        assert tn.path == [(0, 1)]

    def test_slice(self, TensorNetwork, Tensor):

        tn = TensorNetwork()

        a = Tensor(
            indices=["A0", "B1", "C2"], shape=[2, 3, 4], data=range(24)
        )
        b = Tensor(indices=["D3"], shape=[2], data=[0, 1])

        tn.add_tensor(a, [])
        tn.add_tensor(b, [])

        tn.slice_indices(["D3"], 0)

        node = tn.nodes[0]

        assert node.name == "A0B1C2"
        assert node.indices == ["A0", "B1", "C2"]
        assert node.tensor.indices == node.indices
        assert node.tensor.shape == [2, 3, 4]
        assert node.tensor.data == list(range(24))
