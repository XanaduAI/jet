import pytest

import jet


@pytest.mark.parametrize(
    "TaskBasedCpuContractor",
    [
        jet.TaskBasedCpuContractorC64,
        jet.TaskBasedCpuContractorC128,
    ],
)
class TestTaskBasedCpuContractor:
    def test_default_constructor(self, get_tensor_type, TaskBasedCpuContractor):
        """Tests that the default constructor is called."""
        zero = get_tensor_type(JetType=TaskBasedCpuContractor)()
        tbcc = TaskBasedCpuContractor()
        assert tbcc.name_to_tensor_map == {}
        assert tbcc.name_to_parents_map == {}
        assert tbcc.results == []
        assert tbcc.reduction_result == zero
        assert tbcc.flops == 0
        assert tbcc.memory == 0

    def test_contract(self, get_tensor_type, tensor_network, TaskBasedCpuContractor):
        """Tests that a tensor network can be contracted."""
        tn = tensor_network(JetType=TaskBasedCpuContractor)
        path = jet.PathInfo(tn=tn, path=[[0, 1], [2, 3]])
        tbcc = TaskBasedCpuContractor()

        assert tbcc.add_contraction_tasks(tn, path) == 0
        assert tbcc.add_deletion_tasks() == 4
        assert tbcc.add_reduction_task() == 1

        tbcc.contract()

        Tensor = get_tensor_type(TaskBasedCpuContractor)
        want_result = Tensor(shape=[2], indices=["i"], data=[1, -1j])
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
