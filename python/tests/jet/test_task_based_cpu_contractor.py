import pytest

import jet


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
class TestTaskBasedContractor:
    def test_default_constructor(self, dtype):
        """Tests that the default constructor is called."""
        tbc = jet.TaskBasedContractor(dtype=dtype)
        assert tbc.name_to_tensor_map == {}
        assert tbc.name_to_parents_map == {}
        assert tbc.results == []
        assert tbc.reduction_result == jet.Tensor(dtype=dtype)
        assert tbc.flops == 0
        assert tbc.memory == 0

    def test_contract(self, dtype, tensor_network):
        """Tests that a tensor network can be contracted."""
        tn = tensor_network(dtype=dtype)
        path = jet.PathInfo(tn=tn, path=[[0, 1], [2, 3]])
        tbc = jet.TaskBasedContractor(dtype=dtype)

        assert tbc.add_contraction_tasks(tn, path) == 0
        assert tbc.add_deletion_tasks() == 4
        assert tbc.add_reduction_task() == 1

        tbc.contract()

        want_result = jet.Tensor(shape=[2], indices=["i"], data=[1, -1j], dtype=dtype)
        want_name_to_tensor_map = {
            "0:ij": None,
            "1:jk": None,
            "2:k": None,
            "3:ik": None,
            "4:i:results[0]": want_result,
        }
        want_name_to_parents_map = {
            "0:ij": {"3:ik"},
            "1:jk": {"3:ik"},
            "2:k": {"4:i:results[0]"},
            "3:ik": {"4:i:results[0]"},
        }

        assert tbc.name_to_tensor_map == want_name_to_tensor_map
        assert tbc.name_to_parents_map == want_name_to_parents_map
        assert tbc.results == [want_result]
        assert tbc.reduction_result == want_result
        assert tbc.flops == 18
        assert tbc.memory == 6
