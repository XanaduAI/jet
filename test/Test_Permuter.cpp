#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/permute/PermuterIncludes.hpp"

using namespace Jet;
using data_t = std::complex<float>;

namespace {
std::vector<data_t> FillArray(size_t num_vals, size_t start = 0)
{
    std::vector<data_t> data(num_vals);
    for (size_t i = start; i < start + num_vals; i++) {
        data[i - start] = data_t(i, i);
    }
    return data;
}
} // namespace

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

static const std::vector<data_t> large_example{
    {0, 0},     {16, 16},   {64, 64},   {80, 80},   {8, 8},     {24, 24},
    {72, 72},   {88, 88},   {256, 256}, {272, 272}, {320, 320}, {336, 336},
    {264, 264}, {280, 280}, {328, 328}, {344, 344}, {1, 1},     {17, 17},
    {65, 65},   {81, 81},   {9, 9},     {25, 25},   {73, 73},   {89, 89},
    {257, 257}, {273, 273}, {321, 321}, {337, 337}, {265, 265}, {281, 281},
    {329, 329}, {345, 345}, {128, 128}, {144, 144}, {192, 192}, {208, 208},
    {136, 136}, {152, 152}, {200, 200}, {216, 216}, {384, 384}, {400, 400},
    {448, 448}, {464, 464}, {392, 392}, {408, 408}, {456, 456}, {472, 472},
    {129, 129}, {145, 145}, {193, 193}, {209, 209}, {137, 137}, {153, 153},
    {201, 201}, {217, 217}, {385, 385}, {401, 401}, {449, 449}, {465, 465},
    {393, 393}, {409, 409}, {457, 457}, {473, 473}, {2, 2},     {18, 18},
    {66, 66},   {82, 82},   {10, 10},   {26, 26},   {74, 74},   {90, 90},
    {258, 258}, {274, 274}, {322, 322}, {338, 338}, {266, 266}, {282, 282},
    {330, 330}, {346, 346}, {3, 3},     {19, 19},   {67, 67},   {83, 83},
    {11, 11},   {27, 27},   {75, 75},   {91, 91},   {259, 259}, {275, 275},
    {323, 323}, {339, 339}, {267, 267}, {283, 283}, {331, 331}, {347, 347},
    {130, 130}, {146, 146}, {194, 194}, {210, 210}, {138, 138}, {154, 154},
    {202, 202}, {218, 218}, {386, 386}, {402, 402}, {450, 450}, {466, 466},
    {394, 394}, {410, 410}, {458, 458}, {474, 474}, {131, 131}, {147, 147},
    {195, 195}, {211, 211}, {139, 139}, {155, 155}, {203, 203}, {219, 219},
    {387, 387}, {403, 403}, {451, 451}, {467, 467}, {395, 395}, {411, 411},
    {459, 459}, {475, 475}, {4, 4},     {20, 20},   {68, 68},   {84, 84},
    {12, 12},   {28, 28},   {76, 76},   {92, 92},   {260, 260}, {276, 276},
    {324, 324}, {340, 340}, {268, 268}, {284, 284}, {332, 332}, {348, 348},
    {5, 5},     {21, 21},   {69, 69},   {85, 85},   {13, 13},   {29, 29},
    {77, 77},   {93, 93},   {261, 261}, {277, 277}, {325, 325}, {341, 341},
    {269, 269}, {285, 285}, {333, 333}, {349, 349}, {132, 132}, {148, 148},
    {196, 196}, {212, 212}, {140, 140}, {156, 156}, {204, 204}, {220, 220},
    {388, 388}, {404, 404}, {452, 452}, {468, 468}, {396, 396}, {412, 412},
    {460, 460}, {476, 476}, {133, 133}, {149, 149}, {197, 197}, {213, 213},
    {141, 141}, {157, 157}, {205, 205}, {221, 221}, {389, 389}, {405, 405},
    {453, 453}, {469, 469}, {397, 397}, {413, 413}, {461, 461}, {477, 477},
    {6, 6},     {22, 22},   {70, 70},   {86, 86},   {14, 14},   {30, 30},
    {78, 78},   {94, 94},   {262, 262}, {278, 278}, {326, 326}, {342, 342},
    {270, 270}, {286, 286}, {334, 334}, {350, 350}, {7, 7},     {23, 23},
    {71, 71},   {87, 87},   {15, 15},   {31, 31},   {79, 79},   {95, 95},
    {263, 263}, {279, 279}, {327, 327}, {343, 343}, {271, 271}, {287, 287},
    {335, 335}, {351, 351}, {134, 134}, {150, 150}, {198, 198}, {214, 214},
    {142, 142}, {158, 158}, {206, 206}, {222, 222}, {390, 390}, {406, 406},
    {454, 454}, {470, 470}, {398, 398}, {414, 414}, {462, 462}, {478, 478},
    {135, 135}, {151, 151}, {199, 199}, {215, 215}, {143, 143}, {159, 159},
    {207, 207}, {223, 223}, {391, 391}, {407, 407}, {455, 455}, {471, 471},
    {399, 399}, {415, 415}, {463, 463}, {479, 479}, {32, 32},   {48, 48},
    {96, 96},   {112, 112}, {40, 40},   {56, 56},   {104, 104}, {120, 120},
    {288, 288}, {304, 304}, {352, 352}, {368, 368}, {296, 296}, {312, 312},
    {360, 360}, {376, 376}, {33, 33},   {49, 49},   {97, 97},   {113, 113},
    {41, 41},   {57, 57},   {105, 105}, {121, 121}, {289, 289}, {305, 305},
    {353, 353}, {369, 369}, {297, 297}, {313, 313}, {361, 361}, {377, 377},
    {160, 160}, {176, 176}, {224, 224}, {240, 240}, {168, 168}, {184, 184},
    {232, 232}, {248, 248}, {416, 416}, {432, 432}, {480, 480}, {496, 496},
    {424, 424}, {440, 440}, {488, 488}, {504, 504}, {161, 161}, {177, 177},
    {225, 225}, {241, 241}, {169, 169}, {185, 185}, {233, 233}, {249, 249},
    {417, 417}, {433, 433}, {481, 481}, {497, 497}, {425, 425}, {441, 441},
    {489, 489}, {505, 505}, {34, 34},   {50, 50},   {98, 98},   {114, 114},
    {42, 42},   {58, 58},   {106, 106}, {122, 122}, {290, 290}, {306, 306},
    {354, 354}, {370, 370}, {298, 298}, {314, 314}, {362, 362}, {378, 378},
    {35, 35},   {51, 51},   {99, 99},   {115, 115}, {43, 43},   {59, 59},
    {107, 107}, {123, 123}, {291, 291}, {307, 307}, {355, 355}, {371, 371},
    {299, 299}, {315, 315}, {363, 363}, {379, 379}, {162, 162}, {178, 178},
    {226, 226}, {242, 242}, {170, 170}, {186, 186}, {234, 234}, {250, 250},
    {418, 418}, {434, 434}, {482, 482}, {498, 498}, {426, 426}, {442, 442},
    {490, 490}, {506, 506}, {163, 163}, {179, 179}, {227, 227}, {243, 243},
    {171, 171}, {187, 187}, {235, 235}, {251, 251}, {419, 419}, {435, 435},
    {483, 483}, {499, 499}, {427, 427}, {443, 443}, {491, 491}, {507, 507},
    {36, 36},   {52, 52},   {100, 100}, {116, 116}, {44, 44},   {60, 60},
    {108, 108}, {124, 124}, {292, 292}, {308, 308}, {356, 356}, {372, 372},
    {300, 300}, {316, 316}, {364, 364}, {380, 380}, {37, 37},   {53, 53},
    {101, 101}, {117, 117}, {45, 45},   {61, 61},   {109, 109}, {125, 125},
    {293, 293}, {309, 309}, {357, 357}, {373, 373}, {301, 301}, {317, 317},
    {365, 365}, {381, 381}, {164, 164}, {180, 180}, {228, 228}, {244, 244},
    {172, 172}, {188, 188}, {236, 236}, {252, 252}, {420, 420}, {436, 436},
    {484, 484}, {500, 500}, {428, 428}, {444, 444}, {492, 492}, {508, 508},
    {165, 165}, {181, 181}, {229, 229}, {245, 245}, {173, 173}, {189, 189},
    {237, 237}, {253, 253}, {421, 421}, {437, 437}, {485, 485}, {501, 501},
    {429, 429}, {445, 445}, {493, 493}, {509, 509}, {38, 38},   {54, 54},
    {102, 102}, {118, 118}, {46, 46},   {62, 62},   {110, 110}, {126, 126},
    {294, 294}, {310, 310}, {358, 358}, {374, 374}, {302, 302}, {318, 318},
    {366, 366}, {382, 382}, {39, 39},   {55, 55},   {103, 103}, {119, 119},
    {47, 47},   {63, 63},   {111, 111}, {127, 127}, {295, 295}, {311, 311},
    {359, 359}, {375, 375}, {303, 303}, {319, 319}, {367, 367}, {383, 383},
    {166, 166}, {182, 182}, {230, 230}, {246, 246}, {174, 174}, {190, 190},
    {238, 238}, {254, 254}, {422, 422}, {438, 438}, {486, 486}, {502, 502},
    {430, 430}, {446, 446}, {494, 494}, {510, 510}, {167, 167}, {183, 183},
    {231, 231}, {247, 247}, {175, 175}, {191, 191}, {239, 239}, {255, 255},
    {423, 423}, {439, 439}, {487, 487}, {503, 503}, {431, 431}, {447, 447},
    {495, 495}, {511, 511}};

TEMPLATE_TEST_CASE("Permuter<TestType>::Transpose Power-of-2 data", "[Permute]",
                   DefaultPermuter<>, DefaultPermuter<64>, DefaultPermuter<128>,
                   DefaultPermuter<256>, DefaultPermuter<512>,
                   DefaultPermuter<2048>, QFlexPermuter<>, QFlexPermuter<64>,
                   QFlexPermuter<128>, QFlexPermuter<256>, QFlexPermuter<512>,
                   QFlexPermuter<2048>, (QFlexPermuter<1024, 64>),
                   (QFlexPermuter<1024, 128>), (QFlexPermuter<1024, 256>),
                   (QFlexPermuter<1024, 512>), (QFlexPermuter<2048, 1024>),
                   (QFlexPermuter<1024, 2>))
{
    Permuter<TestType> permuter;

    SECTION("2x2x2x2")
    {
        std::vector<size_t> shape{2, 2, 2, 2};
        std::vector<std::string> index_expected{"a", "b", "c", "d"};
        std::vector<data_t> data_expected = FillArray(16);

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
        std::vector<data_t> data_expected = FillArray(16);

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
        std::vector<data_t> data_expected = FillArray(16);

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
        std::vector<data_t> data_expected = FillArray(16);

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
        std::vector<data_t> data_expected = FillArray(16);

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
    } /*
     SECTION("Large test: 2x2x2x2x2x2x2x2x2")
     {
         std::vector<size_t> shape{2, 2, 2, 2, 2, 2, 2, 2, 2};
         std::vector<std::string> index_expected{"a", "b", "c", "d", "e", "f",
     "g", "h", "i"}; std::vector<data_t> data_expected = FillArray(512);

         SECTION("{d, g, h, b, i, a, f, c, e} - > {a,b,c,d,e,f,g,h,i}")
         {
             std::vector<data_t> data_out(large_example);
             CHECK(permuter.Transpose(large_example, shape,
                                      {"d", "g", "h", "b", "i", "a", "f", "c",
     "e"}, index_expected) == data_expected);

             permuter.Transpose(large_example, shape, data_out,
                                {"d", "g", "h", "b", "i", "a", "f", "c", "e"},
     index_expected); CHECK(data_out == data_expected);
         }
     }*/
}

TEST_CASE("DefaultPermuter<>::Transpose Non power-of-2 data", "[Permute]", )
{
    Permuter<DefaultPermuter<>> permuter;

    std::vector<size_t> shape{2, 3, 5};
    std::vector<std::string> index_expected{"a", "b", "c"};
    std::vector<data_t> data_expected = FillArray(30);

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

TEST_CASE("QFlexPermuter<>::Transpose Non power-of-2 data", "[Permute]", )
{
    using namespace Catch::Matchers;

    Permuter<QFlexPermuter<>> permuter;

    std::vector<size_t> shape{2, 3, 5};
    std::vector<std::string> index_expected{"a", "b", "c"};
    std::vector<data_t> data_expected = FillArray(30);

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

TEMPLATE_TEST_CASE("LARGE TEST",
                   "[Permute1]", // DefaultPermuter<>, DefaultPermuter<64>,
                                 // DefaultPermuter<128>, DefaultPermuter<256>,
                                 // DefaultPermuter<512>,
                   DefaultPermuter<2048>, QFlexPermuter<>, QFlexPermuter<64>,
                   QFlexPermuter<128>, QFlexPermuter<256>, QFlexPermuter<512>,
                   QFlexPermuter<2048>, (QFlexPermuter<1024, 64>),
                   (QFlexPermuter<1024, 128>), (QFlexPermuter<1024, 256>),
                   (QFlexPermuter<1024, 512>), (QFlexPermuter<2048, 1024>),
                   (QFlexPermuter<1024, 2>))
{
    Permuter<TestType> permuter;

    SECTION("2x2x2x2x2x2x2x2x2")
    {
        std::vector<size_t> shape{2, 2, 2, 2, 2, 2, 2, 2, 2};
        std::vector<std::string> index_expected{"a", "b", "c", "d", "e",
                                                "f", "g", "h", "i"};
        std::vector<data_t> data_expected = FillArray(512);

        SECTION("{d, g, h, b, i, a, f, c, e} - > {a,b,c,d,e,f,g,h,i}")
        {
            std::vector<data_t> data_out(large_example);
            CHECK(permuter.Transpose(
                      large_example, shape,
                      {"d", "g", "h", "b", "i", "a", "f", "c", "e"},
                      index_expected) == data_expected);

            permuter.Transpose(large_example, shape, data_out,
                               {"d", "g", "h", "b", "i", "a", "f", "c", "e"},
                               index_expected);
            CHECK(data_out == data_expected);
        }
    }
}
