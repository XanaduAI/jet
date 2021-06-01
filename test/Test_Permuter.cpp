#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/permute/PermuterIncludes.hpp"

using namespace Jet;
using data_t = std::complex<float>;

std::vector<data_t> fillArray(size_t num_vals, size_t start = 0)
{
    std::vector<data_t> data(num_vals);
    for (size_t i = start; i < start + num_vals; i++) {
        data[i - start] = data_t(i, i);
    }
    return data;
}
/// Declare reused vector data here to avoid reinitialization
static const std::vector<data_t> data_pow2_dbca{
    {0, 0}, {8, 8}, {2, 2}, {10, 10}, {4, 4}, {12, 12}, {6, 6}, {14, 14},
    {1, 1}, {9, 9}, {3, 3}, {11, 11}, {5, 5}, {13, 13}, {7, 7}, {15, 15}};

static const std::vector<data_t> data_pow2_cbad{
    {0, 0}, {1, 1}, {8, 8},   {9, 9},   {4, 4}, {5, 5}, {12, 12}, {13, 13},
    {2, 2}, {3, 3}, {10, 10}, {11, 11}, {6, 6}, {7, 7}, {14, 14}, {15, 15}};

static const std::vector<data_t> data_pow2_bacd{
    {0, 0}, {1, 1}, {2, 2}, {3, 3}, {8, 8},   {9, 9},   {10, 10}, {11, 11},
    {4, 4}, {5, 5}, {6, 6}, {7, 7}, {12, 12}, {13, 13}, {14, 14}, {15, 15}};

static const std::vector<data_t> data_pow2_badc{
    {0, 0}, {2, 2}, {1, 1}, {3, 3}, {8, 8},   {10, 10}, {9, 9},   {11, 11},
    {4, 4}, {6, 6}, {5, 5}, {7, 7}, {12, 12}, {14, 14}, {13, 13}, {15, 15}};

static const std::vector<data_t> data_pow2_4x4_ba{
    {0, 0}, {4, 4}, {8, 8},   {12, 12}, {1, 1}, {5, 5}, {9, 9},   {13, 13},
    {2, 2}, {6, 6}, {10, 10}, {14, 14}, {3, 3}, {7, 7}, {11, 11}, {15, 15}};

static const std::vector<data_t> data_pow2_4x2x2_bac{
    {0, 0}, {1, 1}, {8, 8},   {9, 9},   {2, 2}, {3, 3}, {10, 10}, {11, 11},
    {4, 4}, {5, 5}, {12, 12}, {13, 13}, {6, 6}, {7, 7}, {14, 14}, {15, 15}};

static const std::vector<data_t> data_pow2_2x4x2_bac{
    {0, 0}, {1, 1}, {4, 4}, {5, 5}, {8, 8},   {9, 9},   {12, 12}, {13, 13},
    {2, 2}, {3, 3}, {6, 6}, {7, 7}, {10, 10}, {11, 11}, {14, 14}, {15, 15}};

static const std::vector<data_t> data_pow2_2x4x2_acb{
    {0, 0}, {4, 4},   {1, 1}, {5, 5},   {2, 2},   {6, 6},   {3, 3},   {7, 7},
    {8, 8}, {12, 12}, {9, 9}, {13, 13}, {10, 10}, {14, 14}, {11, 11}, {15, 15}};

static const std::vector<data_t> data_pow2_4x2x2_acb{
    {0, 0}, {2, 2},   {1, 1}, {3, 3},   {4, 4},   {6, 6},   {5, 5},   {7, 7},
    {8, 8}, {10, 10}, {9, 9}, {11, 11}, {12, 12}, {14, 14}, {13, 13}, {15, 15}};

static const std::vector<data_t> data_pow2_4x2x2_cba{
    {0, 0}, {8, 8},   {4, 4}, {12, 12}, {1, 1}, {9, 9},   {5, 5}, {13, 13},
    {2, 2}, {10, 10}, {6, 6}, {14, 14}, {3, 3}, {11, 11}, {7, 7}, {15, 15}};

static const std::vector<data_t> data_pow2_2x2x4_acb{
    {0, 0}, {2, 2},   {4, 4},   {6, 6},   {1, 1}, {3, 3},   {5, 5},   {7, 7},
    {8, 8}, {10, 10}, {12, 12}, {14, 14}, {9, 9}, {11, 11}, {13, 13}, {15, 15}};

static const std::vector<data_t> data_pow2_2x2x4_cba{
    {0, 0}, {4, 4}, {8, 8}, {12, 12}, {2, 2}, {6, 6}, {10, 10}, {14, 14},
    {1, 1}, {5, 5}, {9, 9}, {13, 13}, {3, 3}, {7, 7}, {11, 11}, {15, 15}};

static const std::vector<data_t> data_non_pow2_bca{
    {0, 0},   {15, 15}, {1, 1},   {16, 16}, {2, 2},   {17, 17},
    {3, 3},   {18, 18}, {4, 4},   {19, 19}, {5, 5},   {20, 20},
    {6, 6},   {21, 21}, {7, 7},   {22, 22}, {8, 8},   {23, 23},
    {9, 9},   {24, 24}, {10, 10}, {25, 25}, {11, 11}, {26, 26},
    {12, 12}, {27, 27}, {13, 13}, {28, 28}, {14, 14}, {29, 29}};

static const std::vector<data_t> data_non_pow2_cba{
    {0, 0}, {15, 15}, {5, 5}, {20, 20}, {10, 10}, {25, 25},
    {1, 1}, {16, 16}, {6, 6}, {21, 21}, {11, 11}, {26, 26},
    {2, 2}, {17, 17}, {7, 7}, {22, 22}, {12, 12}, {27, 27},
    {3, 3}, {18, 18}, {8, 8}, {23, 23}, {13, 13}, {28, 28},
    {4, 4}, {19, 19}, {9, 9}, {24, 24}, {14, 14}, {29, 29}};

static const std::vector<data_t> data_non_pow2_cab{
    {0, 0}, {5, 5}, {10, 10}, {15, 15}, {20, 20}, {25, 25},
    {1, 1}, {6, 6}, {11, 11}, {16, 16}, {21, 21}, {26, 26},
    {2, 2}, {7, 7}, {12, 12}, {17, 17}, {22, 22}, {27, 27},
    {3, 3}, {8, 8}, {13, 13}, {18, 18}, {23, 23}, {28, 28},
    {4, 4}, {9, 9}, {14, 14}, {19, 19}, {24, 24}, {29, 29}};

TEMPLATE_TEST_CASE("Permuter<TestType>::Transpose Power-of-2 data", "[Permute]",
                   //DefaultPermuter<>, DefaultPermuter<64>, DefaultPermuter<128>,
                   //DefaultPermuter<256>, DefaultPermuter<512>,
                   //DefaultPermuter<2048>, QFlexPermuter<>, QFlexPermuter<64>,
                   //QFlexPermuter<128>, QFlexPermuter<256>, QFlexPermuter<512>,
                   //QFlexPermuter<2048>, (QFlexPermuter<1024, 64>),
                   //(QFlexPermuter<1024, 128>), (QFlexPermuter<1024, 256>),
                   //(QFlexPermuter<1024, 512>), (QFlexPermuter<2048, 1024>), 
                   (QFlexPermuter<1024, 2>))
{
    Permuter<TestType> permuter;

    SECTION("2x2x2x2")
    {
        std::vector<size_t> shape{2, 2, 2, 2};
        std::vector<std::string> index_expected{"a", "b", "c", "d"};
        std::vector<data_t> data_expected = fillArray(16);

        SECTION("{d,b,c,a} - > {a,b,c,d}")
        {
            std::vector<data_t> data_out(data_pow2_dbca);
            CHECK(permuter.Transpose(data_pow2_dbca, shape,
                                     {"d", "b", "c", "a"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_pow2_dbca, shape, data_out,
                               {"d", "b", "c", "a"}, index_expected);
            CHECK(data_out == data_expected);
        }
        SECTION("{c,b,a,d} - > {a,b,c,d}")
        {
            std::vector<data_t> data_out(data_pow2_cbad);
            CHECK(permuter.Transpose(data_pow2_cbad, shape,
                                     {"c", "b", "a", "d"},
                                     index_expected) == data_expected);
            permuter.Transpose(data_pow2_cbad, shape, data_out,
                               {"c", "b", "a", "d"}, index_expected);
            CHECK(data_out == data_expected);
        }
        SECTION("{b,a,c,d} - > {a,b,c,d}")
        {
            std::vector<data_t> data_out(data_pow2_bacd);
            CHECK(permuter.Transpose(data_pow2_bacd, shape,
                                     {"b", "a", "c", "d"},
                                     index_expected) == data_expected);
            permuter.Transpose(data_pow2_bacd, shape, data_out,
                               {"b", "a", "c", "d"}, index_expected);
            CHECK(data_out == data_expected);
        }
        SECTION("{b,a,d,c} - > {a,b,c,d}")
        {
            std::vector<data_t> data_out(data_pow2_badc);
            CHECK(permuter.Transpose(data_pow2_badc, shape,
                                     {"b", "a", "d", "c"},
                                     index_expected) == data_expected);
            permuter.Transpose(data_pow2_badc, shape, data_out,
                               {"b", "a", "d", "c"}, index_expected);
            CHECK(data_out == data_expected);
        }
    }
    SECTION("4x4")
    {
        std::vector<size_t> shape{4, 4};
        std::vector<std::string> index_expected{"a", "b"};
        std::vector<data_t> data_expected = fillArray(16);

        SECTION("{b, a} - > {a,b}")
        {
            const std::vector<data_t> data_out(data_pow2_4x4_ba);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"b", "a"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"b", "a"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
    }
    SECTION("2x4x2")
    {
        std::vector<size_t> shape{2, 4, 2};
        std::vector<std::string> index_expected{"a", "b", "c"};
        std::vector<data_t> data_expected = fillArray(16);

        SECTION("{b, a, c} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_2x4x2_bac);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"b", "a", "c"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"b", "a", "c"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{a, c, b} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_2x4x2_acb);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"a", "c", "b"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"a", "c", "b"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{c, b, a} - > {a,b,c}")
        {
            /// 2x4x2 cba == 2x2x2x2 dbca
            static const std::vector<data_t> data_out(data_pow2_dbca);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"c", "b", "a"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"c", "b", "a"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
    }
    SECTION("4x2x2")
    {
        std::vector<size_t> shape{4, 2, 2};
        std::vector<std::string> index_expected{"a", "b", "c"};
        std::vector<data_t> data_expected = fillArray(16);

        SECTION("{b, a, c} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_4x2x2_bac);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"b", "a", "c"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"b", "a", "c"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{a, c, b} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_4x2x2_acb);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"a", "c", "b"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"a", "c", "b"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{c, b, a} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_4x2x2_cba);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"c", "b", "a"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"c", "b", "a"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
    }
    SECTION("2x2x4")
    {
        std::vector<size_t> shape{2, 2, 4};
        std::vector<std::string> index_expected{"a", "b", "c"};
        std::vector<data_t> data_expected = fillArray(16);

        SECTION("{b, a, c} - > {a,b,c}")
        {
            // 2x2x4 cba == 2x2x2x2 bacd
            static const std::vector<data_t> data_out(data_pow2_bacd);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"b", "a", "c"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"b", "a", "c"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{a, c, b} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_2x2x4_acb);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"a", "c", "b"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"a", "c", "b"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
        SECTION("{c, b, a} - > {a,b,c}")
        {
            static const std::vector<data_t> data_out(data_pow2_2x2x4_cba);

            std::vector<data_t> data_out_ref(data_out);

            CHECK(permuter.Transpose(data_out, shape, {"c", "b", "a"},
                                     index_expected) == data_expected);

            permuter.Transpose(data_out, shape, data_out_ref, {"c", "b", "a"},
                               index_expected);
            CHECK(data_out_ref == data_expected);
        }
    }
}

TEST_CASE("DefaultPermuter<>::Transpose Non power-of-2 data",
                   "[Permute]", )
{
    Permuter<DefaultPermuter<>> permuter;

    std::vector<size_t> shape{2, 3, 5};
    std::vector<std::string> index_expected{"a", "b", "c"};
    std::vector<data_t> data_expected = fillArray(30);

    SECTION("{b,c,a} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_bca);
        CHECK(permuter.Transpose(data_non_pow2_bca, {3, 5, 2}, {"b", "c", "a"},
                                 index_expected) == data_expected);

        permuter.Transpose(data_non_pow2_bca, {3, 5, 2}, data_out,
                           {"b", "c", "a"}, index_expected);
        CHECK(data_out == data_expected);
    }
    SECTION("{c,b,a} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_cba);
        CHECK(permuter.Transpose(data_non_pow2_cba, {5, 3, 2}, {"c", "b", "a"},
                                 index_expected) == data_expected);
        permuter.Transpose(data_non_pow2_cba, {5, 3, 2}, data_out,
                           {"c", "b", "a"}, index_expected);
        CHECK(data_out == data_expected);
    }
    SECTION("{c,a,b} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_cab);
        CHECK(permuter.Transpose(data_non_pow2_cab, {5, 2, 3}, {"c", "a", "b"},
                                 index_expected) == data_expected);
        permuter.Transpose(data_non_pow2_cab, {5, 2, 3}, data_out,
                           {"c", "a", "b"}, index_expected);
        CHECK(data_out == data_expected);
    }
}

TEST_CASE("QFlexPermuter<>::Transpose Non power-of-2 data",
                   "[Permute]", )
{
    using namespace Catch::Matchers;

    Permuter<QFlexPermuter<>> permuter;

    std::vector<size_t> shape{2, 3, 5};
    std::vector<std::string> index_expected{"a", "b", "c"};
    std::vector<data_t> data_expected = fillArray(30);

    SECTION("{b,c,a} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_bca);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_bca, {3, 5, 2},
                                           {"b", "c", "a"}, index_expected),
                        Jet::Exception);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_bca, {3, 5, 2},
                                           data_out, {"b", "c", "a"},
                                           index_expected),
                        Jet::Exception);
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_bca, {3, 5, 2},
                                             {"b", "c", "a"}, index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_bca, {3, 5, 2},
                                             data_out, {"b", "c", "a"},
                                             index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
    }
    SECTION("{c,b,a} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_cba);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_cba, {5, 3, 2},
                                           {"c", "b", "a"}, index_expected),
                        Jet::Exception);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_cba, {5, 3, 2},
                                           data_out, {"c", "b", "a"},
                                           index_expected),
                        Jet::Exception);
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_cba, {5, 3, 2},
                                             {"c", "b", "a"}, index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_cba, {5, 3, 2},
                                             data_out, {"c", "b", "a"},
                                             index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
    }
    SECTION("{c,a,b} - > {a,b,c}")
    {
        std::vector<data_t> data_out(data_non_pow2_cab);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_cab, {5, 2, 3},
                                           {"c", "a", "b"}, index_expected),
                        Jet::Exception);
        CHECK_THROWS_AS(permuter.Transpose(data_non_pow2_cab, {5, 2, 3},
                                           data_out, {"c", "a", "b"},
                                           index_expected),
                        Jet::Exception);
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_cab, {5, 2, 3},
                                             {"c", "a", "b"}, index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
        CHECK_THROWS_WITH(permuter.Transpose(data_non_pow2_cab, {5, 2, 3},
                                             data_out, {"c", "a", "b"},
                                             index_expected),
                          Contains("Fast transpose expects power-of-2 data"));
    }
}
