import pytest

import jet


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
class TestTaskBasedCpuContractor:
    def test_default_constructor(self, dtype):
        """Tests that the default constructor is called."""
        tbcc = jet.TaskBasedCpuContractor(dtype=dtype)
        assert tbcc.name_to_tensor_map == {}
        assert tbcc.name_to_parents_map == {}
        assert tbcc.results == []
        assert tbcc.reduction_result == jet.Tensor(dtype=dtype)
        assert tbcc.flops == 0
        assert tbcc.memory == 0

    def test_contract(self, dtype, tensor_network):
        """Tests that a tensor network can be contracted."""
        tn = tensor_network(dtype=dtype)
        path = jet.PathInfo(tn=tn, path=[[0, 1], [2, 3]])
        tbcc = jet.TaskBasedCpuContractor(dtype=dtype)

        assert tbcc.add_contraction_tasks(tn, path) == 0
        assert tbcc.add_deletion_tasks() == 4
        assert tbcc.add_reduction_task() == 1

        tbcc.contract()

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

        assert tbcc.name_to_tensor_map == want_name_to_tensor_map
        assert tbcc.name_to_parents_map == want_name_to_parents_map
        assert tbcc.results == [want_result]
        assert tbcc.reduction_result == want_result
        assert tbcc.flops == 18
        assert tbcc.memory == 6
