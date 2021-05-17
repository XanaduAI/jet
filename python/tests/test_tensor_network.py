import jet
import pytest


@pytest.mark.parametrize(
    "TensorNetwork, Tensor",
    [(jet.TensorNetwork32, jet.Tensor32), (jet.TensorNetwork64, jet.Tensor64)],
)
class TestTensorNetwork:
    def test_constructor(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        assert tn.nodes == []

        with pytest.raises(IndexError):
            node = tn[0]

        assert tn.num_tensors == 0
        assert tn.path == []

    def test_getitem(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor(shape=[1, 1], indices=["a", "b"])

        tn.add_tensor(a, ["A"])
        a_id = 0

        assert tn[a_id] == a

    def test_get_node_ids_by_tag(self, TensorNetwork, Tensor):
        tn = TensorNetwork()

        a = Tensor()
        b = Tensor()

        tn.add_tensor(a, ["a", "shared_tag"])
        tn.add_tensor(b, ["b", "shared_tag"])

        assert len(tn.get_node_ids_by_tag("a")) == 1
        a_id = tn.get_node_ids_by_tag("a")[0]

        assert len(tn.get_node_ids_by_tag("b")) == 1
        b_id = tn.get_node_ids_by_tag("b")[0]

        assert sorted(tn.get_node_ids_by_tag("shared_tag")) == sorted([a_id, b_id])
