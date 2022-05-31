#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/PathInfo.hpp"
#include "jet/Tensor.hpp"
#include "jet/TensorNetwork.hpp"

using namespace Jet;

using TestTensor = Tensor<std::complex<float>>;

using IndexToSizeMap = PathInfo::IndexToSizeMap;
using Path = PathInfo::Path;
using Steps = PathInfo::Steps;

using Children = std::pair<size_t, size_t>;
using Indices = std::vector<std::string>;
using Shape = std::vector<size_t>;
using Tags = std::vector<std::string>;

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
} // namespace

TEST_CASE("PathInfo::PathInfo()", "[PathInfo]")
{
    const PathInfo path_info;

    const size_t have_leaves = path_info.GetNumLeaves();
    const size_t want_leaves = 0;
    CHECK(have_leaves == want_leaves);

    const Path have_path = path_info.GetPath();
    const Path want_path = {};
    CHECK(have_path == want_path);

    const IndexToSizeMap have_index_sizes = path_info.GetIndexSizes();
    const IndexToSizeMap want_index_sizes = {};
    CHECK(have_index_sizes == want_index_sizes);

    const Steps have_steps = path_info.GetSteps();
    CHECK(have_steps.empty());
}

TEST_CASE("PathInfo::PathInfo(TensorNetwork, Path)", "[PathInfo]")
{
    TensorNetwork<TestTensor> tn;

    SECTION("Tensor network is empty")
    {
        const PathInfo path_info(tn, {});

        const size_t have_leaves = path_info.GetNumLeaves();
        const size_t want_leaves = 0;
        CHECK(have_leaves == want_leaves);

        const Path have_path = path_info.GetPath();
        const Path want_path = {};
        CHECK(have_path == want_path);

        const IndexToSizeMap have_index_sizes = path_info.GetIndexSizes();
        const IndexToSizeMap want_index_sizes = {};
        CHECK(have_index_sizes == want_index_sizes);

        const Steps have_steps = path_info.GetSteps();
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

        const Path have_path = path_info.GetPath();
        const Path want_path = {};
        CHECK(have_path == want_path);

        const IndexToSizeMap have_index_sizes = path_info.GetIndexSizes();
        const IndexToSizeMap want_index_sizes = {{"A0", 3}};
        CHECK(have_index_sizes == want_index_sizes);

        const Steps steps = path_info.GetSteps();
        REQUIRE(steps.size() == 1);

        {
            const auto &step = steps[0];
            CHECK(step.id == 0);
            CHECK(step.parent == -1UL);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"A0"});
            CHECK(step.tensor_indices == Indices{"A0"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{});
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

        const Path have_path = path_info.GetPath();
        const Path want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const IndexToSizeMap have_index_sizes = path_info.GetIndexSizes();
        const IndexToSizeMap want_index_sizes = {
            {"A0", 3}, {"B1", 2}, {"C2", 4}, {"D3", 5}};
        CHECK(have_index_sizes == want_index_sizes);

        const Steps steps = path_info.GetSteps();
        REQUIRE(steps.size() == 4);

        {
            const auto &step = steps[0];
            CHECK(step.id == 0);
            CHECK(step.parent == 3);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"A0", "B1"});
            CHECK(step.tensor_indices == Indices{"A0", "B1"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{"apple"});
        }
        {
            const auto &step = steps[1];
            CHECK(step.id == 1);
            CHECK(step.parent == 3);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"B1", "C2"});
            CHECK(step.tensor_indices == Indices{"B1", "C2"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{"banana"});
        }
        {
            const auto &step = steps[2];
            CHECK(step.id == 2);
            CHECK(step.parent == -1ULL);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"D3"});
            CHECK(step.tensor_indices == Indices{"D3"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{"cherry"});
        }
        {
            const auto &step = steps[3];
            CHECK(step.id == 3);
            CHECK(step.parent == -1UL);
            CHECK(step.children == Children{0, 1});
            CHECK(step.node_indices == Indices{"A0", "C2"});
            CHECK(step.tensor_indices == Indices{"A0", "C2"});
            CHECK(step.contracted_indices == Indices{"B1"});
            CHECK(step.tags == Tags{"apple", "banana"});
        }
    }

    SECTION("Sliced tensor network with non-empty path")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {3, 2});
        const auto tensor_2 = MakeTensor({"B1", "C2"}, {2, 4});

        tn.AddTensor(tensor_1, {"apple"});
        tn.AddTensor(tensor_2, {"banana"});

        tn.SliceIndices({"A0", "C2"}, 0);

        const PathInfo path_info(tn, {{0, 1}});

        const size_t have_leaves = path_info.GetNumLeaves();
        const size_t want_leaves = 2;
        CHECK(have_leaves == want_leaves);

        const Path have_path = path_info.GetPath();
        const Path want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const IndexToSizeMap have_index_sizes = path_info.GetIndexSizes();
        const IndexToSizeMap want_index_sizes = {{"B1", 2}};
        CHECK(have_index_sizes == want_index_sizes);

        const Steps steps = path_info.GetSteps();
        REQUIRE(steps.size() == 3);

        {
            const auto &step = steps[0];
            CHECK(step.id == 0);
            CHECK(step.parent == 2);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"A0(0)", "B1"});
            CHECK(step.tensor_indices == Indices{"B1"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{"apple"});
        }
        {
            const auto &step = steps[1];
            CHECK(step.id == 1);
            CHECK(step.parent == 2);
            CHECK(step.children == Children{-1, -1});
            CHECK(step.node_indices == Indices{"B1", "C2(0)"});
            CHECK(step.tensor_indices == Indices{"B1"});
            CHECK(step.contracted_indices == Indices{});
            CHECK(step.tags == Tags{"banana"});
        }
        {
            const auto &step = steps[2];
            CHECK(step.id == 2);
            CHECK(step.parent == -1UL);
            CHECK(step.children == Children{0, 1});
            CHECK(step.node_indices == Indices{"A0(0)", "C2(0)"});
            CHECK(step.tensor_indices == Indices{});
            CHECK(step.contracted_indices == Indices{"B1"});
            CHECK(step.tags == Tags{"apple", "banana"});
        }
    }
}

TEST_CASE("PathInfo::GetPathStepFlops()", "[PathInfo]")
{
    TensorNetwork<TestTensor> tn;

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
        const double want_flops = 3 + 3;
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
        const double want_flops = (4 * 5) * (6 + 6);
        CHECK(have_flops == want_flops);
    }
    SECTION("Path step with invalid ID")
    {
        using namespace Catch::Matchers;

        const auto tensor_1 = MakeTensor({"A0", "B1", "C2"}, {2, 3, 4});
        const auto tensor_2 = MakeTensor({"A0", "B1", "C3"}, {2, 3, 5});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const PathInfo path_info(tn, {{0, 1}});

        CHECK_THROWS_WITH(path_info.GetPathStepFlops(20),
                          Contains("Step ID is invalid."));
    }
}

TEST_CASE("PathInfo::GetTotalFlops()", "[PathInfo]")
{
    TensorNetwork<TestTensor> tn;

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

    SECTION("Path has invalid node ID")
    {
        using namespace Catch::Matchers;

        const auto tensor_1 = MakeTensor({"A0", "B1"}, {4, 2});
        const auto tensor_2 = MakeTensor({"B1", "C2"}, {2, 3});
        const auto tensor_3 = MakeTensor({"C2", "D3"}, {3, 5});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        CHECK_THROWS_WITH(
            PathInfo(tn, {{10, 1}, {2, 3}}),
            Contains("Node ID 1 in contraction path pair is invalid."));
        CHECK_THROWS_WITH(
            PathInfo(tn, {{0, 10}, {2, 3}}),
            Contains("Node ID 2 in contraction path pair is invalid."));
        CHECK_THROWS_WITH(
            PathInfo(tn, {{0, 1}, {20, 3}}),
            Contains("Node ID 1 in contraction path pair is invalid."));
        CHECK_THROWS_WITH(
            PathInfo(tn, {{0, 1}, {2, 30}}),
            Contains("Node ID 2 in contraction path pair is invalid."));
    }
}

TEST_CASE("PathInfo::GetPathStepMemory()", "[PathInfo]")
{
    TensorNetwork<TestTensor> tn;

    SECTION("Tensor is a scalar")
    {
        const auto tensor = MakeTensor(Indices{}, {});
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
    SECTION("Request invalid ID")
    {
        using namespace Catch::Matchers;

        const auto tensor =
            MakeTensor({"A0", "B1", "C2", "D3", "E4"}, {1, 2, 3, 4, 5});
        tn.AddTensor(tensor, {});

        const PathInfo path_info(tn, {});

        CHECK_THROWS_WITH(path_info.GetPathStepMemory(10),
                          Contains("Step ID is invalid."));
    }
}

TEST_CASE("PathInfo::GetTotalMemory()", "[PathInfo]")
{
    TensorNetwork<TestTensor> tn;

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
