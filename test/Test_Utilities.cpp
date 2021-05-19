#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/Utilities.hpp"

using namespace Jet::Utilities;

using complex_t = std::complex<double>;
using matrix_t = std::vector<complex_t>;

TEST_CASE("ostream << pair", "[Utilities]")
{
    const std::pair<std::string, int> jet = {"Boeing", 737};

    std::ostringstream stream;

    stream << jet;

    REQUIRE(stream.str() == "{Boeing,737}");
}

TEST_CASE("ostream << vector", "[Utilities]")
{
    std::ostringstream stream;

    SECTION("Vector is empty")
    {
        const std::vector<bool> empty;
        stream << empty;
        CHECK(stream.str() == "{}");
    }
    SECTION("Vector has one element")
    {
        const std::vector<int> single = {1};
        stream << single;
        CHECK(stream.str() == "{1}");
    }
    SECTION("Vector has multiple elements")
    {
        const std::vector<std::string> sky = {"Air", "Clouds"};
        stream << sky;
        REQUIRE(stream.str() == "{Air  Clouds}");
    }
}

TEST_CASE("GenerateStringIndex", "[Utilities]")
{
    SECTION("ID is 0") { CHECK(GenerateStringIndex(0) == "a"); }
    SECTION("ID in (0, 52)") { CHECK(GenerateStringIndex(30) == "E"); }
    SECTION("ID in [52, 2 * 52)") { CHECK(GenerateStringIndex(100) == "W0"); }
    SECTION("ID in [2 * 52, 10 * 52)")
    {
        CHECK(GenerateStringIndex(200) == "S2");
    }
    SECTION("ID is no smaller than 10 * 52")
    {
        CHECK(GenerateStringIndex(1000) == "m18");
    }
}

TEST_CASE("Order", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t mat(0);
        CHECK(Order(mat) == 0);
    }
    SECTION("Order 1")
    {
        const matrix_t mat(1 * 1);
        CHECK(Order(mat) == 1);
    }
    SECTION("Order 3")
    {
        const matrix_t mat(3 * 3);
        CHECK(Order(mat) == 3);
    }
    SECTION("Order 1234")
    {
        const matrix_t mat(1234 * 1234);
        CHECK(Order(mat) == 1234);
    }
}

TEST_CASE("Eye", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t eye = Eye<double>(0);
        CHECK(eye.empty());
    }
    SECTION("Order 1")
    {
        const matrix_t have_eye = Eye<double>(1);
        const matrix_t want_eye = {1};
        CHECK(have_eye == want_eye);
    }
    SECTION("Order 3")
    {
        const matrix_t have_eye = Eye<double>(3);
        const matrix_t want_eye = {
            1, 0, 0, 0, 1, 0, 0, 0, 1,
        };
        CHECK(have_eye == want_eye);
    }
}

TEST_CASE("MultiplySquareMatrices", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t m1;
        const matrix_t m2;
        const matrix_t have_prod = MultiplySquareMatrices(m1, m2, 0);
        const matrix_t want_prod;
        CHECK(have_prod == want_prod);
    }
    SECTION("Order 1")
    {
        const matrix_t m1 = {2};
        const matrix_t m2 = {3};
        const matrix_t have_prod = MultiplySquareMatrices(m1, m2, 1);
        const matrix_t want_prod = {6};
        CHECK(have_prod == want_prod);
    }
    SECTION("Order 2")
    {
        const matrix_t m1 = {1, 2, {0, 3}, {4, 4}};
        const matrix_t m2 = {{0, 5}, {6, 6}, 8, 9};
        const matrix_t have_prod = MultiplySquareMatrices(m1, m2, 2);
        const matrix_t want_prod = {{16, 5}, {24, 6}, {17, 32}, {18, 54}};
        CHECK(have_prod == want_prod);
    }
}

TEST_CASE("Pow", "[Utilities]")
{
    const matrix_t mat = {{0, 1}};

    SECTION("Power of 0")
    {
        const matrix_t have_pow = Pow(mat, 0);
        const matrix_t want_pow = {1};
        CHECK(have_pow == want_pow);
    }
    SECTION("Power of 1")
    {
        const matrix_t have_pow = Pow(mat, 1);
        const matrix_t want_pow = {{0, 1}};
        CHECK(have_pow == want_pow);
    }
    SECTION("Power of 2")
    {
        const matrix_t have_pow = Pow(mat, 2);
        const matrix_t want_pow = {-1};
        CHECK(have_pow == want_pow);
    }
    SECTION("Power of 3")
    {
        const matrix_t have_pow = Pow(mat, 3);
        const matrix_t want_pow = {{0, -1}};
        CHECK(have_pow == want_pow);
    }
}

TEST_CASE("Matrix + Matrix", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t m1;
        const matrix_t m2;
        const matrix_t have_sum = m1 + m2;
        const matrix_t want_sum;
        CHECK(have_sum == want_sum);
    }
    SECTION("Order 1")
    {
        const matrix_t m1 = {1};
        const matrix_t m2 = {2};
        const matrix_t have_sum = m1 + m2;
        const matrix_t want_sum = {3};
        CHECK(have_sum == want_sum);
    }
    SECTION("Order 2")
    {
        const matrix_t m1 = {1, 2, 3, 4};
        const matrix_t m2 = {5, 6, {0, 5}, {6, 6}};
        const matrix_t have_sum = m1 + m2;
        const matrix_t want_sum = {6, 8, {3, 5}, {10, 6}};
        CHECK(have_sum == want_sum);
    }
}

TEST_CASE("Matrix - Matrix", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t m1;
        const matrix_t m2;
        const matrix_t have_diff = m1 - m2;
        const matrix_t want_diff;
        CHECK(have_diff == want_diff);
    }
    SECTION("Order 1")
    {
        const matrix_t m1 = {3};
        const matrix_t m2 = {1};
        const matrix_t have_diff = m1 - m2;
        const matrix_t want_diff = {2};
        CHECK(have_diff == want_diff);
    }
    SECTION("Order 2")
    {
        const matrix_t m1 = {5, 6, 7, 8};
        const matrix_t m2 = {2, 1, {0, 3}, {4, 4}};
        const matrix_t have_diff = m1 - m2;
        const matrix_t want_diff = {3, 5, {7, -3}, {4, -4}};
        CHECK(have_diff == want_diff);
    }
}

TEST_CASE("Matrix * Scalar", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t mat;
        const complex_t c = 2;
        const matrix_t have_prod = mat * c;
        const matrix_t want_prod;
        CHECK(have_prod == want_prod);
    }
    SECTION("Order 1")
    {
        const matrix_t mat = {0, 1, 2};
        const complex_t c = {2, 3};
        const matrix_t have_prod = mat * c;
        const matrix_t want_prod = {0, {2, 3}, {4, 6}};
        CHECK(have_prod == want_prod);
    }
    SECTION("Scalar is 0")
    {
        const matrix_t mat = {1, 2, 3};
        const complex_t c = 0;
        const matrix_t have_prod = mat * c;
        const matrix_t want_prod = {0, 0, 0};
        CHECK(have_prod == want_prod);
    }
}

TEST_CASE("DiagExp", "[Utilities]")
{
    SECTION("Order 0")
    {
        const matrix_t v;
        const matrix_t have_diag = DiagExp(v);
        const matrix_t want_diag;
        CHECK(have_diag == want_diag);
    }
    SECTION("Order 1")
    {
        const matrix_t v = {1};
        const matrix_t have_diag = DiagExp(v);
        const matrix_t want_diag = {std::exp(1)};
        CHECK(have_diag == want_diag);
    }
    SECTION("Order 2")
    {
        const matrix_t v = {1, 2, 3, 0};
        const matrix_t have_diag = DiagExp(v);
        const matrix_t want_diag = {std::exp(1), 0, 0, 1};
        CHECK(have_diag == want_diag);
    }
}

TEST_CASE("InVector", "[Utilities]")
{
    SECTION("Vector is empty")
    {
        const bool s = false;
        const std::vector<bool> v;
        CHECK_FALSE(InVector(s, v));
    }
    SECTION("Element exists in vector")
    {
        const int s = 2;
        const std::vector<int> v = {1, 2, 3};
        CHECK(InVector(s, v));
    }
    SECTION("Element is missing from vector")
    {
        const std::string s = "jet";
        const std::vector<std::string> v = {"bra", "ket"};
        CHECK_FALSE(InVector(s, v));
    }
}

TEST_CASE("VectorIntersection", "[Utilities]")
{
    SECTION("Vectors are disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3};
        const std::vector<int> v2 = {4, 5, 6};
        const std::vector<int> have_set = VectorIntersection(v1, v2);
        const std::vector<int> want_set;
        CHECK(have_set == want_set);
    }
    SECTION("Vectors are not disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3, 4};
        const std::vector<int> v2 = {3, 4, 5, 6};
        const std::vector<int> have_set = VectorIntersection(v1, v2);
        const std::vector<int> want_set = {3, 4};
        CHECK(have_set == want_set);
    }
}

TEST_CASE("VectorUnion", "[Utilities]")
{
    SECTION("Vectors are disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3};
        const std::vector<int> v2 = {4, 5, 6};
        const std::vector<int> have_set = VectorUnion(v1, v2);
        const std::vector<int> want_set = {1, 2, 3, 4, 5, 6};
        CHECK(have_set == want_set);
    }
    SECTION("Vectors are not disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3, 4};
        const std::vector<int> v2 = {3, 4, 5, 6};
        const std::vector<int> have_set = VectorUnion(v1, v2);
        const std::vector<int> want_set = {1, 2, 3, 4, 5, 6};
        CHECK(have_set == want_set);
    }
}

TEST_CASE("VectorSubtraction", "[Utilities]")
{
    SECTION("Vectors are disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3};
        const std::vector<int> v2 = {4, 5, 6};
        const std::vector<int> have_set = VectorSubtraction(v1, v2);
        const std::vector<int> want_set = v1;
        CHECK(have_set == want_set);
    }
    SECTION("Vectors are not disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3, 4};
        const std::vector<int> v2 = {3, 4, 5, 6};
        const std::vector<int> have_set = VectorSubtraction(v1, v2);
        const std::vector<int> want_set = {1, 2};
        CHECK(have_set == want_set);
    }
}

TEST_CASE("VectorDisjunctiveUnion", "[Utilities]")
{
    SECTION("Nothing in common")
    {
        const std::vector<int> v1 = {1, 2, 3};
        const std::vector<int> v2 = {4, 5, 6};
        const std::vector<int> have_set = VectorDisjunctiveUnion(v1, v2);
        const std::vector<int> want_set = {1, 2, 3, 4, 5, 6};
        CHECK(have_set == want_set);
    }
    SECTION("Vectors are not disjoint")
    {
        const std::vector<int> v1 = {1, 2, 3, 4};
        const std::vector<int> v2 = {3, 4, 5, 6};
        const std::vector<int> have_set = VectorDisjunctiveUnion(v1, v2);
        const std::vector<int> want_set = {1, 2, 5, 6};
        CHECK(have_set == want_set);
    }
}

TEST_CASE("JoinStringVector", "[Utilities]")
{
    SECTION("Vector is empty")
    {
        const std::vector<std::string> v;
        CHECK(JoinStringVector(v) == "");
    }
    SECTION("Vector has one string")
    {
        const std::vector<std::string> v = {"jet"};
        CHECK(JoinStringVector(v) == "jet");
    }
    SECTION("Vector has multiple strings")
    {
        const std::vector<std::string> v = {"Bra", "-", "ket"};
        CHECK(JoinStringVector(v) == "Bra-ket");
    }
}

TEST_CASE("VectorConcatenation", "[Utilities]")
{
    SECTION("Both vectors are empty")
    {
        const std::vector<int> v1;
        const std::vector<int> v2;
        const std::vector<int> have_concat = VectorConcatenation(v1, v2);
        const std::vector<int> want_concat;
        CHECK(have_concat == want_concat);
    }
    SECTION("Prefix vector is empty")
    {
        const std::vector<int> v1;
        const std::vector<int> v2 = {3, 4};
        const std::vector<int> have_concat = VectorConcatenation(v1, v2);
        const std::vector<int> want_concat = {3, 4};
        CHECK(have_concat == want_concat);
    }
    SECTION("Suffix vector is empty")
    {
        const std::vector<int> v1 = {1, 2};
        const std::vector<int> v2;
        const std::vector<int> have_concat = VectorConcatenation(v1, v2);
        const std::vector<int> want_concat = {1, 2};
        CHECK(have_concat == want_concat);
    }
    SECTION("Neither vector is empty")
    {
        const std::vector<int> v1 = {1, 2};
        const std::vector<int> v2 = {3, 4};
        const std::vector<int> have_concat = VectorConcatenation(v1, v2);
        const std::vector<int> want_concat = {1, 2, 3, 4};
        CHECK(have_concat == want_concat);
    }
}

TEST_CASE("Factorial", "[Utilities]")
{
    SECTION("Factorial of 0") { CHECK(Factorial(0) == 1); }
    SECTION("Factorial of 1") { CHECK(Factorial(1) == 1); }
    SECTION("Factorial of 4") { CHECK(Factorial(4) == 24); }
    SECTION("Factorial of 10") { CHECK(Factorial(10) == 3628800); }
}

TEST_CASE("ShapeToSize", "[Utilities]")
{
    SECTION("Shape has 0 dimensions") { CHECK(ShapeToSize({}) == 1); }
    SECTION("Shape has 1 dimension") { CHECK(ShapeToSize({3}) == 3); }
    SECTION("Shape has 2 dimensions") { CHECK(ShapeToSize({2, 3}) == 6); }
    SECTION("Shape has 3 dimensions")
    {
        CHECK(ShapeToSize({2, 3, 4}) == 24);
        CHECK(ShapeToSize({3, 4, 2}) == 24);
    }
}

TEST_CASE("UnravelIndex", "[Utilities]")
{
    using multi_index_t = std::vector<size_t>;

    SECTION("Linear index of 0 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(0, {2, 3}) == multi_index_t{0, 0});
    }
    SECTION("Linear index of 1 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(1, {2, 3}) == multi_index_t{0, 1});
    }
    SECTION("Linear index of 2 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(2, {2, 3}) == multi_index_t{0, 2});
    }
    SECTION("Linear index of 3 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(3, {2, 3}) == multi_index_t{1, 0});
    }
    SECTION("Linear index of 4 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(4, {2, 3}) == multi_index_t{1, 1});
    }
    SECTION("Linear index of 5 in (2, 3) dimensions")
    {
        CHECK(UnravelIndex(5, {2, 3}) == multi_index_t{1, 2});
    }
    // SECTION("Linear index of 6 in (2, 3) dimensions")
    // {
    //     const auto linear_index = 6;
    //     const shape_t shape = {2, 3};
    //     const multi_index_t have_index = UnravelIndex(linear_index, shape);
    //     const multi_index_t want_index = {{2, 0}};
    //     CHECK(have_index == want_index);
    // }
}

TEST_CASE("RavelIndex", "[Utilities]")
{
    SECTION("Multi-index of (0, 0) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({0, 0}, {2, 3}) == 0);
    }
    SECTION("Multi-index of (0, 1) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({0, 1}, {2, 3}) == 1);
    }
    SECTION("Multi-index of (0, 2) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({0, 2}, {2, 3}) == 2);
    }
    SECTION("Multi-index of (1, 0) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({1, 0}, {2, 3}) == 3);
    }
    SECTION("Multi-index of (1, 1) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({1, 1}, {2, 3}) == 4);
    }
    SECTION("Multi-index of (1, 2) in (2, 3) dimensions")
    {
        CHECK(RavelIndex({1, 2}, {2, 3}) == 5);
    }
}

TEST_CASE("SplitStringOnMultipleDelimiters", "[Utilities]")
{
    SECTION("Empty string")
    {
        const std::string s;
        const std::vector<std::string> delimiters = {" "};
        const auto have_tokens = SplitStringOnMultipleDelimiters(s, delimiters);
        const std::vector<std::string> want_tokens;
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Single token")
    {
        const std::string s = " jet  ";
        const std::vector<std::string> delimiters = {","};
        const auto have_tokens = SplitStringOnMultipleDelimiters(s, delimiters);
        const std::vector<std::string> want_tokens = {"jet"};
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Multiple tokens with too few delimiters")
    {
        const std::string s = "air,sea,land";
        const std::vector<std::string> delimiters = {","};
        const auto have_tokens = SplitStringOnMultipleDelimiters(s, delimiters);
        const std::vector<std::string> want_tokens = {"air", "sea,land"};
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Multiple tokens with the same number of delimiters")
    {
        const std::string s = "air, sea, , land";
        const std::vector<std::string> delimiters = {",", ",", ","};
        const auto have_tokens = SplitStringOnMultipleDelimiters(s, delimiters);
        const std::vector<std::string> want_tokens = {"air", "sea", "land"};
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Multiple tokens with too many delimiters")
    {
        const std::string s = "air, sea, land";
        const std::vector<std::string> delimiters = {",", ",", ",", ","};
        const auto have_tokens = SplitStringOnMultipleDelimiters(s, delimiters);
        const std::vector<std::string> want_tokens = {"air", "sea", "land"};
        CHECK(have_tokens == want_tokens);
    }
}

TEST_CASE("SplitStringOnDelimiterRecursively", "[Utilities]")
{
    std::vector<std::string> have_tokens;

    SECTION("Empty string")
    {
        const std::string s;
        const std::string delimiter = ",";
        const std::vector<std::string> want_tokens = {""};
        SplitStringOnDelimiterRecursively(s, delimiter, have_tokens);
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Single token")
    {
        const std::string s = "jet";
        const std::string delimiter = ",";
        const std::vector<std::string> want_tokens = {"jet"};
        SplitStringOnDelimiterRecursively(s, delimiter, have_tokens);
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Multiple tokens")
    {
        const std::string s = "air,sea,,land";
        const std::string delimiter = ",";
        const std::vector<std::string> want_tokens = {"air", "sea", "", "land"};
        SplitStringOnDelimiterRecursively(s, delimiter, have_tokens);
        CHECK(have_tokens == want_tokens);
    }
    SECTION("Overlapping delimiter")
    {
        const std::string s = "lhs === rhs";
        const std::string delimiter = "==";
        const std::vector<std::string> want_tokens = {"lhs ", "= rhs"};
        SplitStringOnDelimiterRecursively(s, delimiter, have_tokens);
        CHECK(have_tokens == want_tokens);
    }
}

TEST_CASE("ReplaceAllInString", "[Utilities]")
{
    SECTION("Empty string")
    {
        std::string s;
        const std::string from = "x";
        const std::string to = "y";
        const std::string want_replaced;
        ReplaceAllInString(s, from, to);
        CHECK(s == want_replaced);
    }
    SECTION("No replacements")
    {
        std::string s = "jet";
        const std::string from = "k";
        const std::string to = "j";
        const std::string want_replaced = "jet";
        ReplaceAllInString(s, from, to);
        CHECK(s == want_replaced);
    }
    SECTION("One replacement")
    {
        std::string s = "jet";
        const std::string from = "je";
        const std::string to = "ke";
        const std::string want_replaced = "ket";
        ReplaceAllInString(s, from, to);
        CHECK(s == want_replaced);
    }
    SECTION("Multiple replacements")
    {
        std::string s = "linear algebra";
        const std::string from = "a";
        const std::string to = "4";
        const std::string want_replaced = "line4r 4lgebr4";
        ReplaceAllInString(s, from, to);
        CHECK(s == want_replaced);
    }
    SECTION("Replacement contains string to be replaced")
    {
        std::string s = "Heyy";
        const std::string from = "y";
        const std::string to = "yy";
        const std::string want_replaced = "Heyyyy";
        ReplaceAllInString(s, from, to);
        CHECK(s == want_replaced);
    }
}

TEST_CASE("VectorInVector", "[Utilities]")
{
    SECTION("Empty Vector")
    {
        std::vector<std::string> empty = {};
        std::vector<std::string> abc = {"a", "b", "c"};

        CHECK(VectorInVector(empty, abc) == true);
        CHECK(VectorInVector(empty, empty) == true);
        CHECK(VectorInVector(abc, empty) == false);
    }

    SECTION("Single Element Vector")
    {
        std::vector<std::string> abc = {"a", "b", "c"};
        CHECK(VectorInVector({"a"}, abc) == true);
        CHECK(VectorInVector({"b"}, abc) == true);
        CHECK(VectorInVector({"c"}, abc) == true);
        CHECK(VectorInVector({"d"}, abc) == false);
        CHECK(VectorInVector({"ab"}, abc) == false);
    }
    SECTION("Multiple Element Vector")
    {
        std::vector<std::string> abc = {"a", "b", "c"};

        CHECK(VectorInVector({"a", "b"}, abc) == true);
        CHECK(VectorInVector({"b", "c"}, abc) == true);
        CHECK(VectorInVector(abc, abc) == true);
        CHECK(VectorInVector({"c", "a", "b"}, abc) == true);
        CHECK(VectorInVector({"b", "c", "a"}, abc) == true);
        CHECK(VectorInVector({"a", "b", "c", "d"}, abc) == false);
    }
}

TEST_CASE("FastCopy", "[Utilities]")
{
    std::vector<double> vec1(10);
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] = i;
    }
    std::vector<double> vec2;
    FastCopy(vec1, vec2);
    double error = 0.;
    for (size_t i = 0; i < vec2.size(); i++) {
        error += std::abs(vec2[i] - i);
    }
    REQUIRE(error < 1e-15);
}
