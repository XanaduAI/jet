#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/Permute/Permute.hpp"

using namespace Jet;
using namespace Jet::Utilities;
using data_t = std::complex<float>;


TEST_CASE("NaivePermute::Transpose", "[Permute]")
{
    NaivePermute permuter;
}


TEST_CASE("QFlexPermute::GenerateBinaryReorderingMap", "[Permute]")
{
    QFlexPermute qfp;

    std::vector<size_t> data_pos_map_out(16, 0);
    
    std::vector<size_t> data_pos_map_expect_03 
    { 0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15 };
    std::vector<size_t> data_pos_map_expect_02 
    { 0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15 };
    std::vector<size_t> data_pos_map_expect_01
    { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15 };
    std::vector<size_t> data_pos_map_expect_01_23
    { 0, 2, 1, 3, 8, 10, 9, 11, 4, 6, 5, 7, 12, 14, 13, 15 };

    qfp.GenerateBinaryReorderingMap({3,1,2,0}, data_pos_map_out);
    CHECK( data_pos_map_out == data_pos_map_expect_03 );
    qfp.GenerateBinaryReorderingMap({2,1,0,3}, data_pos_map_out);
    CHECK( data_pos_map_out == data_pos_map_expect_02 );
    qfp.GenerateBinaryReorderingMap({1,0,2,3}, data_pos_map_out);
    CHECK( data_pos_map_out == data_pos_map_expect_01 );
    qfp.GenerateBinaryReorderingMap({1,0,3,2}, data_pos_map_out);
    CHECK( data_pos_map_out == data_pos_map_expect_01_23 );
}