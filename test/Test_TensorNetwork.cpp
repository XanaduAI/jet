#include <complex>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/Tensor.hpp"
#include "jet/TensorNetwork.hpp"

using namespace Jet;

using complex_t = std::complex<float>;
using tensor_t = Tensor<complex_t>;

using data_t = std::vector<complex_t>;
using indices_t = std::vector<std::string>;
using index_to_edge_map_t = TensorNetwork<tensor_t>::index_to_edge_map_t;
using path_t = TensorNetwork<tensor_t>::path_t;
using shape_t = std::vector<size_t>;
using tag_to_node_ids_map_t =
    std::unordered_map<std::string, std::vector<size_t>>;
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
            tensor.SetValue(index, complex_t{static_cast<float>(i),
                                             static_cast<float>(2 * i)});
        }
    }
    return tensor;
}

/**
 * @brief Returns a tag-to-nodes map that is suitable for equality comparison.
 *
 * @param tn Tensor network holding the tag-to-nodes map.
 * @return Modified tag-to-nodes map with a defined `==` operator.
 */
tag_to_node_ids_map_t GetTagToNodeIDsMap(const TensorNetwork<tensor_t> &tn)
{
    tag_to_node_ids_map_t tag_map;
    for (const auto &[tag, node_id] : tn.GetTagToNodesMap()) {
        tag_map[tag].emplace_back(node_id);
    }
    // Canonicalize the order of the node IDs.
    for (auto &[_, node_ids] : tag_map) {
        std::sort(node_ids.begin(), node_ids.end());
    }
    return tag_map;
}
} // namespace

TEST_CASE("TensorNetwork::NumIndices", "[TensorNetwork]")
{
    TensorNetwork<tensor_t> tn;
    CHECK(tn.NumIndices() == 0);

    const auto tensor_AB = MakeTensor({"A0", "B1"}, {2, 3});
    tn.AddTensor(tensor_AB, {});
    CHECK(tn.NumIndices() == 2);

    const auto tensor_BC = MakeTensor({"B1", "C2"}, {3, 2});
    tn.AddTensor(tensor_BC, {});
    CHECK(tn.NumIndices() == 3);
}

TEST_CASE("TensorNetwork::NumTensors", "[TensorNetwork]")
{
    TensorNetwork<tensor_t> tn;
    CHECK(tn.NumTensors() == 0);

    const tensor_t tensor;

    tn.AddTensor(tensor, {});
    CHECK(tn.NumTensors() == 1);

    tn.AddTensor(tensor, {});
    CHECK(tn.NumTensors() == 2);
}

TEST_CASE("TensorNetwork::AddTensor", "[TensorNetwork]")
{
    TensorNetwork<tensor_t> tn;

    SECTION("Single tensor with no indices or tags")
    {
        const auto tensor = MakeTensor({}, {});
        tn.AddTensor(tensor, {});

        REQUIRE(tn.GetNodes().size() == 1);

        const auto &node = tn.GetNodes().front();
        CHECK(node.id == 0);
        CHECK(node.name == "_");
        CHECK(node.indices == indices_t{});
        CHECK(node.tags == tags_t{});
        CHECK(node.contracted == false);

        CHECK(tn.GetIndexToEdgeMap().empty());
        CHECK(tn.GetTagToNodesMap().empty());
    }

    SECTION("Single tensor with indices and tags")
    {
        const auto tensor = MakeTensor({"A0", "B1"}, {2, 3});
        tn.AddTensor(tensor, {"chaotic", "neutral"});

        REQUIRE(tn.GetNodes().size() == 1);

        const auto &node = tn.GetNodes().front();
        CHECK(node.id == 0);
        CHECK(node.name == "A0B1");
        CHECK(node.indices == indices_t{"A0", "B1"});
        CHECK(node.tags == tags_t{"chaotic", "neutral"});
        CHECK(node.contracted == false);

        const index_to_edge_map_t have_edge_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_edge_map = {{"A0", {2, {0}}},
                                                   {"B1", {3, {0}}}};
        CHECK(have_edge_map == want_edge_map);

        const tag_to_node_ids_map_t have_tag_map = GetTagToNodeIDsMap(tn);
        const tag_to_node_ids_map_t want_tag_map = {{"chaotic", {0}},
                                                    {"neutral", {0}}};
        CHECK(have_tag_map == want_tag_map);
    }

    SECTION("Multiple tensors with indices and tags")
    {
        const auto tensor_1 = MakeTensor({"C2", "D3"}, {1, 4});
        const auto tensor_2 = MakeTensor({"D3", "E4"}, {4, 2});

        tn.AddTensor(tensor_1, {"bot", "mid"});
        tn.AddTensor(tensor_2, {"mid", "top"});

        REQUIRE(tn.GetNodes().size() == 2);
        {
            const auto &node = tn.GetNodes()[0];
            CHECK(node.id == 0);
            CHECK(node.name == "C2D3");
            CHECK(node.indices == indices_t{"C2", "D3"});
            CHECK(node.tags == tags_t{"bot", "mid"});
            CHECK(node.contracted == false);
        }
        {
            const auto &node = tn.GetNodes()[1];
            CHECK(node.id == 1);
            CHECK(node.name == "D3E4");
            CHECK(node.indices == indices_t{"D3", "E4"});
            CHECK(node.tags == tags_t{"mid", "top"});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_edge_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_edge_map = {{"D3", {4, {0, 1}}},
                                                   {"E4", {2, {1}}}};
        CHECK(have_edge_map == want_edge_map);

        const tag_to_node_ids_map_t have_tag_map = GetTagToNodeIDsMap(tn);
        const tag_to_node_ids_map_t want_tag_map = {
            {"bot", {0}}, {"mid", {0, 1}}, {"top", {1}}};
        CHECK(have_tag_map == want_tag_map);
    }
}

TEST_CASE("TensorNetwork::SliceIndices", "[TensorNetwork]")
{
    using namespace Catch::Matchers;
    TensorNetwork<tensor_t> tn;

    const auto tensor_1 = MakeTensor({"A0", "B1", "C2"}, {2, 3, 4});
    const auto tensor_2 = MakeTensor({"D3"}, {2});

    tn.AddTensor(tensor_1, {});
    tn.AddTensor(tensor_2, {});

    SECTION("Slice [:, :, :]")
    {
        tn.SliceIndices({"D3"}, 0);
        const auto &node = tn.GetNodes().front();

        const std::string have_name = node.name;
        const std::string want_name = "A0B1C2";
        CHECK(have_name == want_name);

        const indices_t have_node_indices = node.indices;
        const indices_t want_node_indices = {"A0", "B1", "C2"};
        CHECK(have_node_indices == want_node_indices);

        const indices_t have_tensor_indices = node.tensor.GetIndices();
        const indices_t want_tensor_indices = {"A0", "B1", "C2"};
        CHECK(have_tensor_indices == want_tensor_indices);

        const shape_t have_tensor_shape = node.tensor.GetShape();
        const shape_t want_tensor_shape = {2, 3, 4};
        CHECK(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = node.tensor.GetData();
        const data_t want_tensor_data = {
            {0, 0},   {1, 2},   {2, 4},   {3, 6},   {4, 8},   {5, 10},
            {6, 12},  {7, 14},  {8, 16},  {9, 18},  {10, 20}, {11, 22},
            {12, 24}, {13, 26}, {14, 28}, {15, 30}, {16, 32}, {17, 34},
            {18, 36}, {19, 38}, {20, 40}, {21, 42}, {22, 44}, {23, 46}};
        CHECK(have_tensor_data == want_tensor_data);
    }

    SECTION("Slice [0, :, :]")
    {
        tn.SliceIndices({"A0"}, 0);
        const auto &node = tn.GetNodes().front();

        const std::string have_name = node.name;
        const std::string want_name = "A0(0)B1C2";
        CHECK(have_name == want_name);

        const indices_t have_node_indices = node.indices;
        const indices_t want_node_indices = {"A0(0)", "B1", "C2"};
        CHECK(have_node_indices == want_node_indices);

        const indices_t have_tensor_indices = node.tensor.GetIndices();
        const indices_t want_tensor_indices = {"B1", "C2"};
        CHECK(have_tensor_indices == want_tensor_indices);

        const shape_t have_tensor_shape = node.tensor.GetShape();
        const shape_t want_tensor_shape = {3, 4};
        CHECK(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = node.tensor.GetData();
        const data_t want_tensor_data = {{0, 0},  {1, 2},  {2, 4},   {3, 6},
                                         {4, 8},  {5, 10}, {6, 12},  {7, 14},
                                         {8, 16}, {9, 18}, {10, 20}, {11, 22}};
        CHECK(have_tensor_data == want_tensor_data);
    }

    SECTION("Slice [1, :, :]")
    {
        tn.SliceIndices({"A0"}, 1);
        const auto &node = tn.GetNodes().front();

        const std::string have_name = node.name;
        const std::string want_name = "A0(1)B1C2";
        CHECK(have_name == want_name);

        const indices_t have_node_indices = node.indices;
        const indices_t want_node_indices = {"A0(1)", "B1", "C2"};
        CHECK(have_node_indices == want_node_indices);

        const indices_t have_tensor_indices = node.tensor.GetIndices();
        const indices_t want_tensor_indices = {"B1", "C2"};
        CHECK(have_tensor_indices == want_tensor_indices);

        const shape_t have_tensor_shape = node.tensor.GetShape();
        const shape_t want_tensor_shape = {3, 4};
        CHECK(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = node.tensor.GetData();
        const data_t want_tensor_data = {
            {12, 24}, {13, 26}, {14, 28}, {15, 30}, {16, 32}, {17, 34},
            {18, 36}, {19, 38}, {20, 40}, {21, 42}, {22, 44}, {23, 46}};
        CHECK(have_tensor_data == want_tensor_data);
    }

    SECTION("Slice [1, :, 2]")
    {
        tn.SliceIndices({"A0", "C2"}, 1 + 2 * 2);
        const auto &node = tn.GetNodes().front();

        const std::string have_name = node.name;
        const std::string want_name = "A0(1)B1C2(2)";
        CHECK(have_name == want_name);

        const indices_t have_node_indices = node.indices;
        const indices_t want_node_indices = {"A0(1)", "B1", "C2(2)"};
        CHECK(have_node_indices == want_node_indices);

        const indices_t have_tensor_indices = node.tensor.GetIndices();
        const indices_t want_tensor_indices = {"B1"};
        CHECK(have_tensor_indices == want_tensor_indices);

        const shape_t have_tensor_shape = node.tensor.GetShape();
        const shape_t want_tensor_shape = {3};
        CHECK(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = node.tensor.GetData();
        const data_t want_tensor_data = {{14, 28}, {18, 36}, {22, 44}};
        CHECK(have_tensor_data == want_tensor_data);
    }

    SECTION("Slice [1, 2, 3]")
    {
        tn.SliceIndices({"A0", "B1", "C2"}, 1 + 2 * 2 + 3 * 2 * 3);
        const auto &node = tn.GetNodes().front();

        const std::string have_name = node.name;
        const std::string want_name = "A0(1)B1(2)C2(3)";
        CHECK(have_name == want_name);

        const indices_t have_node_indices = node.indices;
        const indices_t want_node_indices = {"A0(1)", "B1(2)", "C2(3)"};
        CHECK(have_node_indices == want_node_indices);

        const indices_t have_tensor_indices = node.tensor.GetIndices();
        const indices_t want_tensor_indices = {};
        CHECK(have_tensor_indices == want_tensor_indices);

        const shape_t have_tensor_shape = node.tensor.GetShape();
        const shape_t want_tensor_shape = {};
        CHECK(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = node.tensor.GetData();
        const data_t want_tensor_data = {{23, 46}};
        CHECK(have_tensor_data == want_tensor_data);
    }
    SECTION("Slice non-existent index")
    {
        CHECK_THROWS_WITH(tn.SliceIndices({"E0", "B0"}, 0),
                          Contains("Sliced index does not exist."));
    }
}

TEST_CASE("TensorNetwork::Contract", "[TensorNetwork]")
{
    using namespace Catch::Matchers;
    TensorNetwork<tensor_t> tn;

    SECTION("Implicit contraction of network [2]")
    {
        const auto tensor = MakeTensor({"A0"}, {2});
        tn.AddTensor(tensor, {});

        const auto result = tn.Contract();

        const path_t have_path = tn.GetPath();
        const path_t want_path = {};
        CHECK(have_path == want_path);

        const data_t have_tensor_data = result.GetData();
        const data_t want_tensor_data = {{0, 0}, {1, 2}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 1);
        {
            CHECK(nodes[0].contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {{"A0", {2, {0}}}};
        CHECK(have_map == want_map);
    }

    SECTION("Implicit contraction of network [3] -(0)- [3]")
    {
        const auto tensor_1 = MakeTensor({"A0"}, {3});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const auto result = tn.Contract();

        const path_t have_path = tn.GetPath();
        const path_t want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const data_t have_tensor_data = {result.GetValue({})};
        const data_t want_tensor_data = {{-15, 20}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 3);
        {
            CHECK(nodes[0].contracted == true);
            CHECK(nodes[1].contracted == true);
        }
        {
            const auto &node = nodes[2];
            CHECK(node.id == 2);
            CHECK(node.name == "_");
            CHECK(node.indices == indices_t{});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {};
        CHECK(have_map == want_map);
    }

    SECTION("Implicit contraction of network [3, 2] -(0)- [3]")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {3, 2});
        const auto tensor_2 = MakeTensor({"A0"}, {3});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const auto result = tn.Contract();

        const path_t have_path = tn.GetPath();
        const path_t want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const data_t have_tensor_data = result.GetData();
        const data_t want_tensor_data = {{-30, 40}, {-39, 52}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 3);
        {
            CHECK(nodes[0].contracted == true);
            CHECK(nodes[1].contracted == true);
        }
        {
            const auto &node = nodes[2];
            CHECK(node.id == 2);
            CHECK(node.name == "B1");
            CHECK(node.indices == indices_t{"B1"});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {{"B1", {2, {2}}}};
        CHECK(have_map == want_map);
    }

    SECTION("Implicit contraction of network [2, 3] -(0, 1)- [2, 3]")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {3, 2});
        const auto tensor_2 = MakeTensor({"A0", "B1"}, {3, 2});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const auto result = tn.Contract();

        const path_t have_path = tn.GetPath();
        const path_t want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const data_t have_tensor_data = {result.GetValue({})};
        const data_t want_tensor_data = {{-165, 220}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 3);
        {
            CHECK(nodes[0].contracted == true);
            CHECK(nodes[1].contracted == true);
        }
        {
            const auto &node = nodes[2];
            CHECK(node.id == 2);
            CHECK(node.name == "_");
            CHECK(node.indices == indices_t{});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {};
        CHECK(have_map == want_map);
    }

    SECTION("Implicit contraction of network [2, 3] -(1)- [3, 3]")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"C2", "B1"}, {3, 3});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});

        const auto result = tn.Contract();

        const path_t have_path = tn.GetPath();
        const path_t want_path = {{0, 1}};
        CHECK(have_path == want_path);

        const shape_t have_tensor_shape = result.GetShape();
        const shape_t want_tensor_shape = {2, 3};
        REQUIRE(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = result.GetData();
        const data_t want_tensor_data = {{-15, 20}, {-42, 56},   {-69, 92},
                                         {-42, 56}, {-150, 200}, {-258, 344}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 3);
        {
            CHECK(nodes[0].contracted == true);
            CHECK(nodes[1].contracted == true);
        }
        {
            const auto &node = nodes[2];
            CHECK(node.id == 2);
            CHECK(node.name == "A0C2");
            CHECK(node.indices == indices_t{"A0", "C2"});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {{"A0", {2, {2}}},
                                              {"C2", {3, {2}}}};
        CHECK(have_map == want_map);
    }

    SECTION("Explicit contraction of network [2, 3] -(1)- [2, 3] -(0)- [2, 2]")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"C2", "B1"}, {2, 3});
        const auto tensor_3 = MakeTensor({"C2", "D3"}, {2, 2});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        const auto result = tn.Contract({{1, 2}, {0, 3}});

        const path_t have_path = tn.GetPath();
        const path_t want_path = {{1, 2}, {0, 3}};
        CHECK(have_path == want_path);

        const shape_t have_tensor_shape = result.GetShape();
        const shape_t want_tensor_shape = {2, 2};
        REQUIRE(have_tensor_shape == want_tensor_shape);

        const data_t have_tensor_data = result.GetData();
        const data_t want_tensor_data = {
            {-308, -56}, {-517, -94}, {-1100, -200}, {-1804, -328}};
        CHECK(have_tensor_data == want_tensor_data);

        const auto &nodes = tn.GetNodes();
        REQUIRE(nodes.size() == 5);
        {
            CHECK(nodes[0].contracted == true);
            CHECK(nodes[1].contracted == true);
            CHECK(nodes[2].contracted == true);
        }
        {
            const auto &node = nodes[3];
            CHECK(node.id == 3);
            CHECK(node.name == "B1D3");
            CHECK(node.indices == indices_t{"B1", "D3"});
            CHECK(node.contracted == true);
        }
        {
            const auto &node = nodes[4];
            CHECK(node.id == 4);
            CHECK(node.name == "A0D3");
            CHECK(node.indices == indices_t{"A0", "D3"});
            CHECK(node.contracted == false);
        }

        const index_to_edge_map_t have_map = tn.GetIndexToEdgeMap();
        const index_to_edge_map_t want_map = {{"D3", {2, {4}}},
                                              {"A0", {2, {4}}}};
        CHECK(have_map == want_map);
    }

    SECTION("Contract empty network")
    {
        CHECK_THROWS_WITH(
            tn.Contract(),
            Contains("An empty tensor network cannot be contracted."));
    }
    SECTION("Invalid node ID 1")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"C2", "B1"}, {2, 3});
        const auto tensor_3 = MakeTensor({"C2", "D3"}, {2, 2});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        CHECK_THROWS_WITH(
            tn.Contract({{10, 2}, {0, 3}}),
            Contains("Node ID 1 in contraction pair is invalid."));
    }
    SECTION("Invalid node ID 2")
    {
        const auto tensor_1 = MakeTensor({"A0", "B1"}, {2, 3});
        const auto tensor_2 = MakeTensor({"C2", "B1"}, {2, 3});
        const auto tensor_3 = MakeTensor({"C2", "D3"}, {2, 2});

        tn.AddTensor(tensor_1, {});
        tn.AddTensor(tensor_2, {});
        tn.AddTensor(tensor_3, {});

        CHECK_THROWS_WITH(
            tn.Contract({{1, 2}, {0, 4}}),
            Contains("Node ID 2 in contraction pair is invalid."));
    }
}
