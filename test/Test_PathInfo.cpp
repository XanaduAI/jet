#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/PathInfo.hpp"
#include "jet/Tensor.hpp"
#include "jet/TensorNetwork.hpp"

using namespace Jet;

using complex_t = std::complex<float>;
using tensor_t = Tensor<complex_t>;

using index_to_size_map_t = PathInfo::index_to_size_map_t;
using path_t = PathInfo::path_t;
using steps_t = PathInfo::steps_t;

using children_t = std::pair<size_t, size_t>;
using indices_t = std::vector<std::string>;
using shape_t = std::vector<size_t>;
using tags_t = std::vector<std::string>;

namespace {
/**
 * @brief Constructs a tensor with sequentially increasing values.
 *
 * @param indices Indices of the tensor.
 * @param shape Shape of the tensor.
 * @return Tensor with the given indices and shape.  Each element in the tensor
 *         is populated with the value of its linear index.
 */
tensor_t MakeTensor(const indices_t &indices, const shape_t &shape)
{
    tensor_t tensor(indices, shape);
    if (!shape.empty()) {
        for (size_t i = 0; i < tensor.GetSize(); i++) {
            const auto index = Jet::Utilities::UnravelIndex(i, shape);
            tensor.SetValue(index, i);
        }
    }
    return tensor;
}
} // namespace

TEST_CASE("PathInfo::PathInfo()", "[PathInfo]")
{
    const PathInfo path_info;

    const size_t have_leaves = path_info.GetNumLeaves();
    const size_t want_leaves = 0;
    CHECK(have_leaves == want_leaves);

    const path_t have_path = path_info.GetPath();
    const path_t want_path = {};
    CHECK(have_path == want_path);

    const index_to_size_map_t have_index_sizes = path_info.GetIndexSizes();
    const index_to_size_map_t want_index_sizes = {};
    CHECK(have_index_sizes == want_index_sizes);

    const steps_t have_steps = path_info.GetSteps();
    CHECK(have_steps.empty());
}

TEST_CASE("PathInfo::PathInfo(TensorNetwork, path_t)", "[PathInfo]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Tensor network is empty")
    {
        const PathInfo path_info(tn, {});

        const size_t have_leaves = path_info.GetNumLeaves();
        const size_t want_leaves = 0;
        CHECK(have_leaves == want_leaves);

        const path_t have_path = path_info.GetPath();
        const path_t want_path = {};
        CHECK(have_path == want_path);

        const index_to_size_map_t have_index_sizes = path_info.GetIndexSizes();
        const index_to_size_map_t want_index_sizes = {};
        CHECK(have_index_sizes == want_index_sizes);

        const steps_t have_steps = path_info.GetSteps();
        CHECK(have_steps.empty());
    }

    SECTION("Path is empty")
    {
        const auto tensor = MakeTensor({"A0"}, {3});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const size_t have_leaves = path_info.GetNumLeaves();
        const size_t want_leaves = 1;
        CHECK(have_leaves == want_leaves);

        const path_t have_path = path_info.GetPath();
        const path_t want_path = {};
        CHECK(have_path == want_path);

        const index_to_size_map_t have_index_sizes = path_info.GetIndexSizes();
        const index_to_size_map_t want_index_sizes = {{"A0", 3}};
        CHECK(have_index_sizes == want_index_sizes);

        const steps_t steps = path_info.GetSteps();
        REQUIRE(steps.size() == 1);

        {
            const auto &step = steps[0];
            CHECK(step.id == 0);
            CHECK(step.parent == -1UL);
            CHECK(step.children == children_t{-1, -1});
            CHECK(step.node_indices == indices_t{"A0"});
            CHECK(step.tensor_indices == indices_t{"A0"});
            CHECK(step.contracted_indices == indices_t{});
            CHECK(step.tags == tags_t{});
        }
    }

    SECTION("Tensor network and path are not empty")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {3, 2});
        const auto tensor_2 = MakeTensor({"B1", "C2"}, {2, 4});
        const auto tensor_3 = MakeTensor({"D3"}, {5});

        tn.AddTensor(tensor_1, {"apple"});
        tn.AddTensor(tensor_2, {"banana"});
        tn.AddTensor(tensor_3, {"cherry"});

        const PathInfo path_info(tn, {{0, 1}});

        const size_t have_leaves = path_info.GetNumLeaves();
        const size_t want_leaves = 3;
        CHECK(have_leaves == want_leaves);

        const path_t have_path = path_info.GetPath();
        const path_t want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const index_to_size_map_t have_index_sizes = path_info.GetIndexSizes();
        const index_to_size_map_t want_index_sizes = {
            {"A0", 3}, {"B1", 2}, {"C2", 4}, {"D3", 5}};
        CHECK(have_index_sizes == want_index_sizes);

        const steps_t steps = path_info.GetSteps();
        REQUIRE(steps.size() == 4);

        {
            const auto &step = steps[0];
            CHECK(step.id == 0);
            CHECK(step.parent == 3);
            CHECK(step.children == children_t{-1, -1});
            CHECK(step.node_indices == indices_t{"A0", "B1"});
            CHECK(step.tensor_indices == indices_t{"A0", "B1"});
            CHECK(step.contracted_indices == indices_t{});
            CHECK(step.tags == tags_t{"apple"});
        }
        {
            const auto &step = steps[1];
            CHECK(step.id == 1);
            CHECK(step.parent == 3);
            CHECK(step.children == children_t{-1, -1});
            CHECK(step.node_indices == indices_t{"B1", "C2"});
            CHECK(step.tensor_indices == indices_t{"B1", "C2"});
            CHECK(step.contracted_indices == indices_t{});
            CHECK(step.tags == tags_t{"banana"});
        }
        {
            const auto &step = steps[2];
            CHECK(step.id == 2);
            CHECK(step.parent == -1ULL);
            CHECK(step.children == children_t{-1, -1});
            CHECK(step.node_indices == indices_t{"D3"});
            CHECK(step.tensor_indices == indices_t{"D3"});
            CHECK(step.contracted_indices == indices_t{});
            CHECK(step.tags == tags_t{"cherry"});
        }
        {
            const auto &step = steps[3];
            CHECK(step.id == 3);
            CHECK(step.parent == -1UL);
            CHECK(step.children == children_t{0, 1});
            CHECK(step.node_indices == indices_t{"A0", "C2"});
            CHECK(step.tensor_indices == indices_t{"A0", "C2"});
            CHECK(step.contracted_indices == indices_t{"B1"});
            CHECK(step.tags == tags_t{"apple", "banana"});
        }
    }
}

TEST_CASE("PathInfo::GetPathStepFlops()", "[PathInfo]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Path step is a leaf")
    {
        const auto tensor = MakeTensor({"A0"}, {2});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_flops = path_info.GetPathStepFlops(0);
        const double want_flops = 0;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path step has no indices")
    {
        const auto tensor_1 = MakeTensor({}, {});
        const auto tensor_2 = MakeTensor({}, {});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        const double have_flops = path_info.GetPathStepFlops(2);
        const double want_flops = 1;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path step has no contracted indices")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {2});
        const auto tensor_2 = MakeTensor({"B1"}, {3});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        const double have_flops = path_info.GetPathStepFlops(2);
        const double want_flops = 6;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path step has no extended indices")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        const double have_flops = path_info.GetPathStepFlops(2);
        const double want_flops = 3 + 2;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path step has extended and contracted indices")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1", "C2"}, {2, 3, 4});
        const auto tensor_2 = MakeTensor({"A0", "B1", "C3"}, {2, 3, 5});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        const double have_flops = path_info.GetPathStepFlops(2);
        const double want_flops = (4 * 5) * (6 + 5);
        CHECK(have_flops == want_flops);
    }
}

TEST_CASE("PathInfo::GetTotalFlops()", "[PathInfo]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Path is empty")
    {
        const PathInfo path_info(tn, {});

        const double have_flops = path_info.GetTotalFlops();
        const double want_flops = 0;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path has no contractions")
    {
        const auto tensor = MakeTensor({"A0"}, {2});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_flops = path_info.GetTotalFlops();
        const double want_flops = 0;
        CHECK(have_flops == want_flops);
    }

    SECTION("Path has several contractions")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {4, 2});
        const auto tensor_2 = MakeTensor({"B1", "C2"}, {2, 3});
        const auto tensor_3 = MakeTensor({"C2", "D3"}, {3, 5});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}, {2, 3}});

        const double have_flops = path_info.GetTotalFlops();
        const double want_flops = 12 * (2 + 1) + 20 * (3 + 2);
        CHECK(have_flops == want_flops);
    }
}

TEST_CASE("PathInfo::GetPathStepMemory()", "[PathInfo]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Tensor is a scalar")
    {
        const auto tensor = MakeTensor(indices_t{}, {});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetPathStepMemory(0);
        const double want_memory = 1;
        CHECK(have_memory == want_memory);
    }

    SECTION("Tensor is a vector")
    {
        const auto tensor = MakeTensor({"A0"}, {2});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetPathStepMemory(0);
        const double want_memory = 2;
        CHECK(have_memory == want_memory);
    }

    SECTION("Tensor is a matrix")
    {
        const auto tensor = MakeTensor({"A0", "B1"}, {2, 3});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetPathStepMemory(0);
        const double want_memory = 2 * 3;
        CHECK(have_memory == want_memory);
    }

    SECTION("Tensor has order 5")
    {
        const auto tensor =
            MakeTensor({"A0", "B1", "C2", "D3", "E4"}, {1, 2, 3, 4, 5});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetPathStepMemory(0);
        const double want_memory = 120;
        CHECK(have_memory == want_memory);
    }
}

TEST_CASE("PathInfo::GetTotalMemory()", "[PathInfo]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Path is empty")
    {
        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetTotalMemory();
        const double want_memory = 0;
        CHECK(have_memory == want_memory);
    }

    SECTION("Path contains a single tensor")
    {
        const auto tensor = MakeTensor({"A0"}, {2});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        const double have_memory = path_info.GetTotalMemory();
        const double want_memory = 2;
        CHECK(have_memory == want_memory);
    }

    SECTION("Path contains multiple tensors")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {3, 2});
        const auto tensor_2 = MakeTensor({"B1", "C2"}, {2, 4});
        const auto tensor_3 = MakeTensor({"D3"}, {5});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const PathInfo path_info(tn, {{0, 1}});

        const double have_memory = path_info.GetTotalMemory();
        const double want_memory = (3 * 2) + (2 * 4) + (5) + (3 * 4);
        CHECK(have_memory == want_memory);
    }
}