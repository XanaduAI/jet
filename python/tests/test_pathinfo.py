import jet
import pytest


@pytest.mark.parametrize("Tensor, TensorNetwork", [(jet.Tensor32, jet.TensorNetwork32), (jet.TensorNetwork64, jet.TensorNetwork64)]) 
class TestPathInfo:
    def test_default_constructor(self):
        path_info = jet.PathInfo()

        assert path_info.num_leaves == 0
        assert path_info.index_to_size_map == {}
        assert path_info.steps == []
        assert path_info.num_leaves == 0
        assert path_info.path == []

        assert path_info.total_flops() == 0
        assert path_info.total_memory() == 0

    def test_constructor(self):
        pass

