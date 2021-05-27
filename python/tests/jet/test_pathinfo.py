import jet


class TestPathInfo:
    def test_default_constructor(self):
        """Tests that the default constructor is called."""
        path_info = jet.PathInfo()

        assert path_info.num_leaves == 0
        assert path_info.index_to_size_map == {}
        assert path_info.steps == []
        assert path_info.num_leaves == 0
        assert path_info.path == []

        assert path_info.total_flops() == 0
        assert path_info.total_memory() == 0

    def test_constructor(self):
        """Tests that the tensor network and path constructor is called."""
        tn = jet.TensorNetwork()

        id_a = tn.add_tensor(jet.Tensor(), ["A"])
        id_b = tn.add_tensor(jet.Tensor(), ["B"])
        id_c = tn.add_tensor(jet.Tensor(), ["C"])

        path = [(id_a, id_b)]

        path_info = jet.PathInfo(tn, path)

        assert path_info.num_leaves == 3
        assert path_info.path == [(id_a, id_b)]
        assert len(path_info.steps) == 4
