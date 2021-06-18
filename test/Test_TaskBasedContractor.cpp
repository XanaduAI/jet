#include <complex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/PathInfo.hpp"
#include "jet/TaskBasedContractor.hpp"
#include "jet/Tensor.hpp"
#include "jet/TensorNetwork.hpp"
#include "jet/Utilities.hpp"

using namespace Jet;

using complex_t = std::complex<float>;
using tensor_t = Tensor<complex_t>;

using Indices = std::vector<std::string>;
using Shape = std::vector<size_t>;
using TaskMap = std::unordered_map<std::string, std::string>;
using TensorMap = std::unordered_map<std::string, std::vector<complex_t>>;
using ParentsMap =
    std::unordered_map<std::string, std::unordered_set<std::string>>;

namespace {
/**
 * @brief Constructs a tensor with sequentially increasing values.
 *
 * @param indices Indices of the tensor.
 * @param shape Shape of the tensor.
 * @return Tensor with the given indices and shape.  Each element in the tensor
 *         is populated with the value of its linear index.
 */
Tensor<std::complex<float>> MakeTensor(const Indices &indices,
                                       const Shape &shape)
{
    Tensor<std::complex<float>> tensor(indices, shape);
    if (!shape.empty()) {
        for (size_t i = 0; i < tensor.GetSize(); i++) {
            const auto index = Jet::Utilities::UnravelIndex(i, shape);
            tensor.SetValue(index, i);
        }
    }
    return tensor;
}

/**
 * @brief Returns a name-to-task map that is suitable for equality comparison.
 *
 * @param tbc Task-based contractor holding the name-to-task map.
 * @return Modified name-to-task map with a defined `==` operator.
 */
TaskMap GetTaskMap(const TaskBasedContractor<tensor_t> &tbc)
{
    TaskMap task_map;
    for (const auto &[name, task] : tbc.GetNameToTaskMap()) {
        task_map[name] = task.name();
    }
    return task_map;
}

/**
 * @brief Returns a name-to-tensor map that is suitable for equality comparison.
 *
 * @param tbc Task-based contractor holding the name-to-tensor map.
 * @return Modified name-to-tensor map with a defined `==` operator.
 */
TensorMap GetTensorMap(const TaskBasedContractor<tensor_t> &tbc)
{
    TensorMap tensor_map;
    for (const auto &[name, tensor_ptr] : tbc.GetNameToTensorMap()) {
        std::vector<complex_t> tensor_data;
        if (tensor_ptr.get() != nullptr) {
            tensor_data = tensor_ptr->GetData();
        }
        tensor_map.emplace(name, tensor_data);
    }
    return tensor_map;
}
} // namespace

TEST_CASE("TaskBasedContractor::AddContractionTasks()", "[TaskBasedContractor]")
{
    TaskBasedContractor<tensor_t> tbc;

    SECTION("Tensor network is empty")
    {
        const TensorNetwork<tensor_t> tn;
        const PathInfo path_info;

        const auto result = tbc.AddContractionTasks(tn, path_info);

        const size_t have_shared_tasks = result;
        const size_t want_shared_tasks = 0;
        CHECK(have_shared_tasks == want_shared_tasks);

        const double have_flops = tbc.GetFlops();
        const double want_flops = 0;
        CHECK(have_flops == want_flops);

        const double have_memory = tbc.GetMemory();
        const double want_memory = 0;
        CHECK(have_memory == want_memory);

        const TaskMap have_task_map = GetTaskMap(tbc);
        const TaskMap want_task_map = {};
        CHECK(have_task_map == want_task_map);

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {};
        CHECK(have_tensor_map == want_tensor_map);

        const ParentsMap have_parents_map = tbc.GetNameToParentsMap();
        const ParentsMap want_parents_map = {};
        CHECK(have_parents_map == want_parents_map);
    }

    SECTION("Path is empty")
    {
        const auto tensor_1 = MakeTensor({"A0", "C2"}, {2, 4});
        const auto tensor_2 = MakeTensor({"A0", "B1"}, {2, 3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {});

        const auto result = tbc.AddContractionTasks(tn, path_info);

        const size_t have_shared_tasks = result;
        const size_t want_shared_tasks = 0;
        CHECK(have_shared_tasks == want_shared_tasks);

        const double have_flops = tbc.GetFlops();
        const double want_flops = 0;
        CHECK(have_flops == want_flops);

        const double have_memory = tbc.GetMemory();
        const double want_memory = 0;
        CHECK(have_memory == want_memory);

        const TaskMap have_task_map = GetTaskMap(tbc);
        const TaskMap want_task_map = {};
        CHECK(have_task_map == want_task_map);

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {};
        CHECK(have_tensor_map == want_tensor_map);

        const ParentsMap have_parents_map = tbc.GetNameToParentsMap();
        const ParentsMap want_parents_map = {};
        CHECK(have_parents_map == want_parents_map);
    }

    SECTION("Path is not empty")
    {
        const auto tensor_1 = MakeTensor({"A0", "C2"}, {2, 4});
        const auto tensor_2 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_3 = MakeTensor({"B1", "C2"}, {3, 4});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}, {1, 2}, {3, 4}});

        const auto result = tbc.AddContractionTasks(tn, path_info);

        const size_t have_shared_tasks = result;
        const size_t want_shared_tasks = 0;
        CHECK(have_shared_tasks == want_shared_tasks);

        const double have_flops = tbc.GetFlops();
        const double want_flops = (3 * 4) * 3 + (2 * 4) * 5 + (2 * 3) * 7;
        CHECK(have_flops == want_flops);

        const double have_memory = tbc.GetMemory();
        const double want_memory = (3 * 4) + (2 * 4) + (2 * 3);
        CHECK(have_memory == want_memory);

        const TaskMap have_task_map = GetTaskMap(tbc);
        const TaskMap want_task_map = {
            {"3:C2B1", "3:C2B1"},
            {"4:A0C2", "4:A0C2"},
            {"5:B1A0:results[0]", "5:B1A0:results[0]"}};
        CHECK(have_task_map == want_task_map);

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {
            {"0:A0C2", {0, 1, 2, 3, 4, 5, 6, 7}},
            {"1:A0B1", {0, 1, 2, 3, 4, 5}},
            {"2:B1C2", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
            {"3:C2B1", {}},
            {"4:A0C2", {}},
            {"5:B1A0:results[0]", {}},
        };
        CHECK(have_tensor_map == want_tensor_map);

        const ParentsMap have_parents_map = tbc.GetNameToParentsMap();
        const ParentsMap want_parents_map = {
            {"0:A0C2", {"3:C2B1"}},
            {"1:A0B1", {"3:C2B1", "4:A0C2"}},
            {"2:B1C2", {"4:A0C2"}},
            {"3:C2B1", {"5:B1A0:results[0]"}},
            {"4:A0C2", {"5:B1A0:results[0]"}},
        };
        CHECK(have_parents_map == want_parents_map);
    }

    SECTION("Paths have shared contractions")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"A0"}, {2});
        const auto tensor_3 = MakeTensor({"B1"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}, {2, 3}});

        const auto result_1 = tbc.AddContractionTasks(tn, path_info);
        const auto result_2 = tbc.AddContractionTasks(tn, path_info);

        const size_t have_shared_tasks_1 = result_1;
        const size_t want_shared_tasks_1 = 0;
        CHECK(have_shared_tasks_1 == want_shared_tasks_1);

        const size_t have_shared_tasks_2 = result_2;
        const size_t want_shared_tasks_2 = 1;
        CHECK(have_shared_tasks_2 == want_shared_tasks_2);

        const double have_flops = tbc.GetFlops();
        const double want_flops = 3 * 3 + 2 * 5;
        CHECK(have_flops == want_flops);

        const double have_memory = tbc.GetMemory();
        const double want_memory = 3 + 2 * 1;
        CHECK(have_memory == want_memory);

        const TaskMap have_task_map = GetTaskMap(tbc);
        const TaskMap want_task_map = {{"3:B1", "3:B1"},
                                       {"4:_:results[0]", "4:_:results[0]"},
                                       {"4:_:results[1]", "4:_:results[1]"}};
        CHECK(have_task_map == want_task_map);

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {
            {"0:A0B1", {0, 1, 2, 3, 4, 5}},
            {"1:A0", {0, 1}},
            {"2:B1", {0, 1, 2}},
            {"3:B1", {}},
            {"4:_:results[0]", {}},
            {"4:_:results[1]", {}},
        };
        CHECK(have_tensor_map == want_tensor_map);

        const ParentsMap have_parents_map = tbc.GetNameToParentsMap();
        const ParentsMap want_parents_map = {
            {"0:A0B1", {"3:B1"}},
            {"1:A0", {"3:B1"}},
            {"2:B1", {"4:_:results[0]", "4:_:results[1]"}},
            {"3:B1", {"4:_:results[0]", "4:_:results[1]"}},
        };
        CHECK(have_parents_map == want_parents_map);
    }
}

TEST_CASE("TaskBasedContractor::Contract()", "[TaskBasedContractor]")
{
    TaskBasedContractor<tensor_t> tbc;

    SECTION("Taskflow is empty")
    {
        tbc.Contract().wait();

        const std::vector<tensor_t> have_results = tbc.GetResults();
        const std::vector<tensor_t> want_results = {};
        CHECK(have_results == want_results);
    }

    SECTION("Taskflow has a single contraction task")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {2});
        const auto tensor_2 = MakeTensor({"A0", "B1"}, {2, 3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        tbc.AddContractionTasks(tn, path_info);
        tbc.Contract().wait();

        const std::vector<tensor_t> have_results = tbc.GetResults();
        const std::vector<tensor_t> want_results = {
            tensor_t({"B1"}, {3}, {3, 4, 5})};
        CHECK(have_results == want_results);
    }

    SECTION("Taskflow has multiple contraction tasks")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"A0"}, {2});
        const auto tensor_3 = MakeTensor({"B1"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}, {2, 3}});

        tbc.AddContractionTasks(tn, path_info);
        tbc.Contract().wait();

        const std::vector<tensor_t> have_results = tbc.GetResults();
        const std::vector<tensor_t> want_results = {tensor_t({}, {}, {14})};
        CHECK(have_results == want_results);
    }

    SECTION("Taskflow has multiple result tasks")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});
        const auto tensor_3 = MakeTensor({"B1"}, {4});
        const auto tensor_4 = MakeTensor({"B1"}, {4});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});
        tn.AddTensor(tensor_4, {});

        const PathInfo path_info_A0(tn, {{0, 1}});
        const PathInfo path_info_B1(tn, {{2, 3}});

        tbc.AddContractionTasks(tn, path_info_A0);
        tbc.AddContractionTasks(tn, path_info_B1);
        tbc.Contract().wait();

        const std::vector<tensor_t> have_results = tbc.GetResults();
        const std::vector<tensor_t> want_results = {tensor_t({}, {}, {5}),
                                                    tensor_t({}, {}, {14})};
        CHECK(have_results == want_results);
    }
}

TEST_CASE("TaskBasedContractor::AddDeletionTasks()", "[TaskBasedContractor]")
{
    TaskBasedContractor<tensor_t> tbc;

    SECTION("No results")
    {
        const TensorNetwork<tensor_t> tn;
        const PathInfo path_info;

        tbc.AddContractionTasks(tn, path_info);
        const auto result = tbc.AddDeletionTasks();

        const size_t have_delete_tasks = result;
        const size_t want_delete_tasks = 0;
        CHECK(have_delete_tasks == want_delete_tasks);
    }

    SECTION("Taskflow has a single contraction task")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        tbc.AddContractionTasks(tn, path_info);
        const auto result = tbc.AddDeletionTasks();

        const size_t have_delete_tasks = result;
        const size_t want_delete_tasks = 2;
        CHECK(have_delete_tasks == want_delete_tasks);

        tbc.Contract().wait();

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {
            {"0:A0", {}},
            {"1:A0", {}},
            {"2:_:results[0]", {5}},
        };
        CHECK(have_tensor_map == want_tensor_map);
    }

    SECTION("Taskflow has multiple contraction tasks")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"A0"}, {2});
        const auto tensor_3 = MakeTensor({"B1"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}, {2, 3}});

        tbc.AddContractionTasks(tn, path_info);
        const auto result = tbc.AddDeletionTasks();

        const size_t have_delete_tasks = result;
        const size_t want_delete_tasks = 4;
        CHECK(have_delete_tasks == want_delete_tasks);

        tbc.Contract().wait();

        const TensorMap have_tensor_map = GetTensorMap(tbc);
        const TensorMap want_tensor_map = {
            {"0:A0B1", {}},           {"1:A0", {}}, {"2:B1", {}}, {"3:B1", {}},
            {"4:_:results[0]", {14}},
        };
        CHECK(have_tensor_map == want_tensor_map);
    }
}

TEST_CASE("TaskBasedContractor::AddReductionTask()", "[TaskBasedContractor]")
{
    TaskBasedContractor<tensor_t> tbc;

    SECTION("Results vector is empty")
    {
        const tensor_t have_result = tbc.GetReductionResult();
        const tensor_t want_result;
        CHECK(have_result == want_result);
    }

    SECTION("Results vector has a single scalar")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        tbc.AddContractionTasks(tn, path_info);
        tbc.AddReductionTask();
        tbc.Contract().wait();

        const tensor_t have_result = tbc.GetReductionResult();
        const tensor_t want_result = tensor_t({}, {}, {5});
        CHECK(have_result == want_result);
    }

    SECTION("Results vector has a single non-scalar")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"A0"}, {2});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        tbc.AddContractionTasks(tn, path_info);
        tbc.AddReductionTask();
        tbc.Contract().wait();

        const tensor_t have_result = tbc.GetReductionResult();
        const tensor_t want_result = tensor_t({"B1"}, {3}, {3, 4, 5});
        CHECK(have_result == want_result);
    }

    SECTION("Results vector has multiple values")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});
        const auto tensor_3 = MakeTensor({"B1"}, {4});
        const auto tensor_4 = MakeTensor({"B1"}, {4});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});
        tn.AddTensor(tensor_4, {});

        const PathInfo path_info_A0(tn, {{0, 1}});
        const PathInfo path_info_B1(tn, {{2, 3}});

        tbc.AddContractionTasks(tn, path_info_A0);
        tbc.AddContractionTasks(tn, path_info_B1);
        tbc.AddReductionTask();
        tbc.Contract().wait();

        const tensor_t have_result = tbc.GetReductionResult();
        const tensor_t want_result = tensor_t({}, {}, {5 + 14});
        CHECK(have_result == want_result);
    }

    SECTION("Multiple reduction tasks are created")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        TensorNetwork<tensor_t> tn;
        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        tbc.AddContractionTasks(tn, path_info);
        size_t return_val_1 = tbc.AddReductionTask();
        size_t return_val_2 = tbc.AddReductionTask();
        size_t return_val_3 = tbc.AddReductionTask();
        tbc.Contract().wait();

        CHECK(return_val_1 == 1);
        CHECK(return_val_2 == 0);
        CHECK(return_val_3 == 0);

        const tensor_t have_result = tbc.GetReductionResult();
        const tensor_t want_result = tensor_t({}, {}, {5});
        CHECK(have_result == want_result);
    }
}
