import jet
import pytest


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
