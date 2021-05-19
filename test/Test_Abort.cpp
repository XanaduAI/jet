#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "jet/Abort.hpp"
#include <catch2/catch.hpp>

using namespace Jet;

TEST_CASE("Jet::JetException", "[Abort]")
{
    using namespace Jet;
    SECTION("JetException::JetException(message)", "[Abort]")
    {
        JetException ex("Message");
        CHECK(std::string(ex.what()) == "Message");
    }
    SECTION("Throw JetException")
    {
        CHECK_THROWS_AS(throw JetException("Thrown message"), JetException);
        CHECK_THROWS_WITH(throw JetException("Thrown message"),
                          "Thrown message");
    }
}
TEST_CASE("Jet::Abort", "[Abort]")
{
    using namespace Jet;
    using namespace Catch::Matchers;

    std::string message = "Abort message";
    std::string file_name = "MyFile.hpp";
    int line = 1471;
    std::string function_name = "my_function";

    CHECK_THROWS_AS(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        JetException);
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Contains(message));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Contains(file_name));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Contains(std::to_string(line)));
    CHECK_THROWS_WITH(
        Abort(message.c_str(), file_name.c_str(), line, function_name.c_str()),
        Contains(function_name));
}

TEST_CASE("Abort Macros", "[Abort]")
{
    using namespace Jet;
    using namespace Catch::Matchers;

    SECTION("JET_ASSERT")
    {
        auto assert_lambda = [](bool a) { JET_ASSERT(a); };
        CHECK_NOTHROW(assert_lambda);
        CHECK_THROWS_AS(assert_lambda(false), JetException);
    }
    SECTION("JET_ABORT_IF_NOT")
    {
        auto abort_if_not_lambda = [](bool a, std::string msg) {
            JET_ABORT_IF_NOT(a, msg.c_str());
        };
        CHECK_NOTHROW(abort_if_not_lambda(true, "No abort"));
        CHECK_THROWS_AS(abort_if_not_lambda(false, "Abort"), JetException);
    }
    SECTION("JET_ABORT_IF")
    {
        auto abort_if_lambda = [](bool a, std::string msg) {
            JET_ABORT_IF(a, msg.c_str());
        };
        CHECK_NOTHROW(abort_if_lambda(false, "No abort"));
        CHECK_THROWS_AS(abort_if_lambda(true, "Abort"), JetException);
    }
    SECTION("JET_ABORT")
    {
        auto abort_lambda = [](std::string msg) { JET_ABORT(msg.c_str()); };
        CHECK_THROWS_AS(abort_lambda("Abort"), JetException);
    }
}
