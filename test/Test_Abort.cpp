#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/Abort.hpp"

using namespace Jet;

TEST_CASE("Jet::Exception", "[Abort]")
{
    SECTION("Exception::Exception(message)")
    {
        Exception ex("Message");
        CHECK(std::string(ex.what()) == "Message");
    }
    SECTION("Throw Exception")
    {
        CHECK_THROWS_AS(throw Exception("Thrown message"), Exception);
        CHECK_THROWS_WITH(throw Exception("Thrown message"), "Thrown message");
    }
}

TEST_CASE("Jet::Abort", "[Abort]")
{
    std::string message = "Abort message";
    std::string file_name = "MyFile.hpp";
    int line = 1471;
    std::string function_name = "my_function";

    CHECK_THROWS_AS(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Exception);
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Catch::Matchers::Contains(message));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Catch::Matchers::Contains(file_name));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Catch::Matchers::Contains(std::to_string(line)));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Catch::Matchers::Contains(function_name));
}

TEST_CASE("Abort Macros", "[Abort]")
{
    SECTION("JET_ASSERT")
    {
        auto assert_lambda = [](bool a) { JET_ASSERT(a); };
        CHECK_NOTHROW(assert_lambda(true));
        CHECK_THROWS_AS(assert_lambda(false), Exception);
    }
    SECTION("JET_ABORT_IF_NOT")
    {
        auto abort_if_not_lambda = [](bool a, std::string msg) {
            JET_ABORT_IF_NOT(a, msg.c_str());
        };
        CHECK_NOTHROW(abort_if_not_lambda(true, "No abort"));
        CHECK_THROWS_AS(abort_if_not_lambda(false, "Abort"), Exception);
    }
    SECTION("JET_ABORT_IF")
    {
        auto abort_if_lambda = [](bool a, std::string msg) {
            JET_ABORT_IF(a, msg.c_str());
        };
        CHECK_NOTHROW(abort_if_lambda(false, "No abort"));
        CHECK_THROWS_AS(abort_if_lambda(true, "Abort"), Exception);
    }
    SECTION("JET_ABORT")
    {
        auto abort_lambda = [](std::string msg) { JET_ABORT(msg.c_str()); };
        CHECK_THROWS_AS(abort_lambda("Abort"), Exception);
    }
}
