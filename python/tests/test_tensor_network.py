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
            node = tn[0]

        assert tn.num_tensors == 0
        assert tn.path == []

    def test_getitem(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor(shape=[1, 1], indices=["a", "b"])

        a_id = tn.add_tensor(a, ["A"])

        assert tn[a_id].tensor == a
        assert tn[a_id].tags == ["A"]

    def test_get_node_ids_by_tag(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor()
        b = Tensor()

        a_id = tn.add_tensor(a, ["a", "shared_tag"])
        b_id = tn.add_tensor(b, ["b", "shared_tag"])

        tn.get_node_ids_by_tag("a") == [a_id]
        tn.get_node_ids_by_tag("b") == [b_id]

        assert sorted(tn.get_node_ids_by_tag("shared_tag")) == sorted([a_id, b_id])
