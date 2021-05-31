#include <catch2/catch.hpp>

#include <string>
#include <vector>

#include "jet/Tensor.hpp"
#include "jet/TensorNetworkIO.hpp"
#include "jet/external/nlohmann/json.hpp"

using json = nlohmann::json;

TEMPLATE_TEST_CASE("TensorNetworkSerializer::operator()",
                   "[TensorNetworkSerializer]", std::complex<float>,
                   std::complex<double>)
{

    using namespace Jet;

    using Path = std::vector<std::pair<size_t, size_t>>;
    using tensor_t = Tensor<TestType>;

    SECTION("empty string")
    {
        CHECK_THROWS_AS(TensorNetworkSerializer<tensor_t>()(""),
                        json::exception);
    }

    SECTION("json string with array root")
    {
        CHECK_THROWS_AS(TensorNetworkSerializer<tensor_t>()("[]"),
                        TensorFileException);
    }
    SECTION("json string with no keys")
    {
        CHECK_THROWS_AS(TensorNetworkSerializer<tensor_t>()("{}"),
                        TensorFileException);
    }

    SECTION("json string path key only")
    {
        CHECK_THROWS_AS(
            TensorNetworkSerializer<tensor_t>()(R"({"path": [[0,1]]})"),
            TensorFileException);
    }

    SECTION("json string with invalid complex number")
    {
        using Catch::Matchers::Contains;

        auto js_str = json::parse(R"({"tensors": [
                [["I0"], ["a"], [2], [[1.0], [0.0,0.0]]],
                [["I1"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I2"], ["a"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I3"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]]]})")
                          .dump(-1);

        CHECK_THROWS_WITH(TensorNetworkSerializer<tensor_t>()(js_str),
                          Contains("[1.0]"));
    }

    SECTION("json string with 4 tensors and 4 indices")
    {
        auto js_str = json::parse(R"({"tensors": [
                [["I0"], ["a"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I1"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I2"], ["c"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I3"], ["d"], [2], [[1.0, 0.0], [0.0,0.0]]]]})")
                          .dump(-1);

        auto tn_f = TensorNetworkSerializer<tensor_t>()(js_str);

        CHECK(tn_f.path.has_value() == false);

        auto tn = tn_f.tensors;

        CHECK(tn.NumIndices() == 4);
        CHECK(tn.NumTensors() == 4);

        CHECK(TensorNetworkSerializer<tensor_t>(-1)(tn) == js_str);
    }

    SECTION("json string with 4 tensors, 4 indices and a path")
    {
        auto js_str = json::parse(R"({
            "path": [[0, 1], [1, 2]],
            "tensors": [
                [["I0"], ["a"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I1"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I2"], ["c"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I3"], ["d"], [2], [[1.0, 0.0], [0.0,0.0]]]]})")
                          .dump(-1);

        auto tn_f = TensorNetworkSerializer<tensor_t>()(js_str);

        REQUIRE(tn_f.path.has_value() == true);

        CHECK(tn_f.path.value().GetPath() == Path({{0, 1}, {1, 2}}));

        auto tn = tn_f.tensors;

        CHECK(tn.NumIndices() == 4);
        CHECK(tn.NumTensors() == 4);

        CHECK(TensorNetworkSerializer<tensor_t>(-1)(tn, tn_f.path.value()) ==
              js_str);
    }

    SECTION("json string with 4 tensors and 2 indices")
    {
        auto js_str = json::parse(R"({"tensors": [
                [["I0"], ["a"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I1"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I2"], ["a"], [2], [[1.0, 0.0], [0.0,0.0]]],
                [["I3"], ["b"], [2], [[1.0, 0.0], [0.0,0.0]]]]})")
                          .dump(-1);

        auto tn_f = TensorNetworkSerializer<tensor_t>()(js_str);

        CHECK(tn_f.path.has_value() == false);

        auto tn = tn_f.tensors;

        CHECK(tn.NumIndices() == 2);
        CHECK(tn.NumTensors() == 4);

        CHECK(TensorNetworkSerializer<tensor_t>(-1)(tn) == js_str);
    }
}
