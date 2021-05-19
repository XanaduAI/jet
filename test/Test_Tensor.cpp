#include <algorithm>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/Tensor.hpp"

using c_fp64 = std::complex<double>;
using c_fp32 = std::complex<float>;
using data_t = std::vector<c_fp32>;

using namespace Jet;

TEMPLATE_TEST_CASE("Tensor::Tensor", "[Tensor]", c_fp32, c_fp64)
{
    SECTION("Tensor") { REQUIRE(std::is_constructible<Tensor<>>::value); }
    SECTION("Tensor<TestType> {}")
    {
        REQUIRE(std::is_constructible<Tensor<TestType>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<size_t>}")
    {
        REQUIRE(std::is_constructible<Tensor<TestType>,
                                      std::vector<size_t>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<std::string>, std::vector<size_t>}")
    {
        REQUIRE(
            std::is_constructible<Tensor<TestType>, std::vector<std::string>,
                                  std::vector<size_t>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<std::string>, "
            "std::vector<size_t>, std::vector<TestType>}")
    {
        REQUIRE(
            std::is_constructible<Tensor<TestType>, std::vector<std::string>,
                                  std::vector<size_t>,
                                  std::vector<TestType>>::value);
    }
    SECTION("Require fail: Tensor<TestType> {std::vector<size_t>, "
            "std::vector<std::string>, std::vector<TestType>}")
    {
        REQUIRE_FALSE(
            std::is_constructible<Tensor<TestType>, std::vector<size_t>,
                                  std::vector<std::string>,
                                  std::vector<TestType>>::value);
    }
}

TEST_CASE("Tensor instantiation", "[Tensor]")
{
    SECTION("Default constructor")
    {
        Tensor tensor, tensor_cmp;
        CHECK(tensor == tensor_cmp);
    }
    SECTION("Copy constructor, Size: {2,3}")
    {
        Tensor tensor({2, 3});
        Tensor tensor_copy(tensor);
        CHECK(tensor == tensor_copy);
    }
    SECTION("Copy assignment, Size: {2,3}")
    {
        Tensor tensor({2, 3});
        Tensor tensor_copy{};
        tensor_copy = tensor;
        CHECK(tensor == tensor_copy);
    }
    SECTION("Move constructor, Size: {2,3}")
    {
        Tensor tensor({2, 3});
        Tensor tensor_move(std::move(tensor));
        CHECK(tensor != tensor_move);
    }
    SECTION("Move assignment, Size: {2,3}")
    {
        Tensor tensor({2, 3});
        Tensor tensor_move{};
        tensor_move = std::move(tensor);
        CHECK(tensor != tensor_move);
    }
}

TEST_CASE("Tensor::GetShape", "[Tensor]")
{
    SECTION("Default size")
    {
        Tensor tensor;
        CHECK(tensor.GetShape().empty());
    }
    SECTION("Shape: {2,3}")
    {
        std::vector<size_t> shape{2, 3};
        Tensor tensor(shape);
        CHECK(tensor.GetShape() == shape);
    }
    SECTION("Unequal shape: {2,3} and {3,2}")
    {
        std::vector<size_t> shape_23{2, 3};
        std::vector<size_t> shape_32{3, 2};

        Tensor tensor_23(shape_23);
        Tensor tensor_32(shape_32);

        CHECK(tensor_23.GetShape() == shape_23);
        CHECK(tensor_32.GetShape() == shape_32);
        CHECK(tensor_23.GetShape() != tensor_32.GetShape());
    }
}

TEST_CASE("Tensor::GetSize", "[Tensor]")
{
    SECTION("Default size")
    {
        Tensor tensor;
        CHECK(tensor.GetSize() == 1);
    }
    SECTION("Shape: {2,3}")
    {
        Tensor tensor({2, 3});
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Equal size: {2,3} and {3,2}")
    {
        Tensor tensor_23({2, 3});
        Tensor tensor_32({3, 2});
        CHECK(tensor_23.GetSize() == tensor_32.GetSize());
    }
    SECTION("Unequal size: {2,3} and {3,3}")
    {
        Tensor tensor_23({2, 3});
        Tensor tensor_33({3, 3});
        CHECK(tensor_23.GetSize() != tensor_33.GetSize());
    }
}

TEST_CASE("Tensor::GetIndices", "[Tensor]")
{
    SECTION("Default size")
    {
        Tensor tensor;
        CHECK(tensor.GetIndices().empty());
    }
    SECTION("Size: {2,3}, Indices: default")
    {
        Tensor tensor({2, 3});
        CHECK(tensor.GetIndices() == std::vector<std::string>{"?a", "?b"});
    }
    SECTION("Size: {2,3}, Indices: {x, y}")
    {
        Tensor tensor({"x", "y"}, {2, 3});
        CHECK(tensor.GetIndices() == std::vector<std::string>{"x", "y"});
    }
}

TEST_CASE("Tensor::GetData", "[Tensor]")
{
    SECTION("Data: default")
    {
        Tensor tensor;
        CHECK(tensor.GetScalar() == c_fp32(0, 0));
        CHECK(tensor.GetData() == std::vector<c_fp32>{{0, 0}});
        CHECK(tensor.GetValue({}) == c_fp32(0, 0));
    }
    SECTION("Size: {2,3}, Indices: default, Data: default")
    {
        Tensor tensor({2, 3});

        std::vector<c_fp32> data(2 * 3, c_fp32(0.0, 0.0));

        CHECK(tensor.GetData() == data);
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: default")
    {
        Tensor tensor({"x", "y"}, {2, 3});
        std::vector<c_fp32> data(2 * 3, c_fp32(0.0, 0.0));

        CHECK(tensor.GetData() == data);
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: [0.5+0.25i]*6")
    {
        std::vector<c_fp32> data(6, c_fp32(0.5, 0.25));
        Tensor tensor({"x", "y"}, {2, 3}, data);
        CHECK(tensor.GetData() == data);
    }
}

TEST_CASE("Tensor::GetIndexToDimension", "[Tensor]")
{
    SECTION("Default")
    {
        Tensor tensor;
        CHECK(tensor.GetIndexToDimension().empty());
    }
    SECTION("Size: {2,3}, Indices: default, Data: default")
    {
        Tensor tensor({2, 3});
        std::unordered_map<std::string, size_t> i2d{{"?a", 2}, {"?b", 3}};
        CHECK(tensor.GetIndexToDimension() == i2d);
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: default")
    {
        Tensor tensor({"x", "y"}, {3, 2});
        std::unordered_map<std::string, size_t> i2d{{"x", 3}, {"y", 2}};
        CHECK(tensor.GetIndexToDimension() == i2d);
    }
}

TEST_CASE("Tensor::IsScalar", "[Tensor]")
{
    SECTION("Default")
    {
        Tensor tensor;
        CHECK(tensor.IsScalar());
    }
    SECTION("Size: {1}, Indices: {x}, Data: {0.5+0.25i}")
    {
        Tensor tensor({"x"}, {1}, {{0.5, 0.25}});
        CHECK(tensor.IsScalar());
        CHECK(tensor.GetSize() == 1);
        CHECK(tensor.GetIndices().size() == 1);
        CHECK(tensor.GetShape() == std::vector<size_t>{1});
    }
    SECTION("Size: {2,3}, Indices: default, Data: default")
    {
        std::vector<size_t> shape{2, 3};
        Tensor tensor(shape);
        CHECK(!tensor.IsScalar());
        CHECK(tensor.GetSize() == 6);
        CHECK(tensor.GetIndices().size() == 2);
        CHECK(tensor.GetShape() == shape);
    }
}

TEST_CASE("Tensor::FillRandom", "[Tensor]")
{
    std::vector<std::size_t> t_shape{3, 2, 3};
    std::vector<std::string> t_indices{"a", "b", "c"};
    Tensor tensor1(t_indices, t_shape);
    Tensor tensor2(t_indices, t_shape);

    SECTION("Size: {2,3}, Indices: {a,b,c}, Data: FillRandom()")
    {
        tensor1.FillRandom();
        tensor2.FillRandom();
        CHECK(tensor1 != tensor2);
    }
    SECTION("Size: {2,3}, Indices: {a,b,c}, Data: FillRandom(7)")
    {
        tensor1.FillRandom(7);
        tensor2.FillRandom(7);
        CHECK(tensor1 == tensor2);
    }
    SECTION(
        "Size: {2,3}, Indices: {a,b,c}, Data: FillRandom() and FillRandom(5)")
    {
        tensor1.FillRandom();
        tensor2.FillRandom(5);
        CHECK(tensor1 != tensor2);
    }
    SECTION(
        "Size: {2,3}, Indices: {a,b,c}, Data: FillRandom(3) and FillRandom(5)")
    {
        tensor1.FillRandom(3);
        tensor2.FillRandom(5);
        CHECK(tensor1 != tensor2);
    }
}

TEST_CASE("Tensor::GetValue", "[Tensor]")
{
    std::vector<std::size_t> t_shape{3, 2};
    std::vector<std::string> t_indices{"a", "b"};
    std::vector<c_fp32> data{{0, 0.5}, {1, 0.5}, {2, 0.5},
                             {3, 0.5}, {4, 0.5}, {5, 0.5}};

    Tensor tensor(t_indices, t_shape, data);

    CHECK(tensor.GetValue({0, 0}) == data[0]);
    CHECK(tensor.GetValue({1, 0}) == data[1]);
    CHECK(tensor.GetValue({2, 0}) == data[2]);
    CHECK(tensor.GetValue({0, 1}) == data[3]);
    CHECK(tensor.GetValue({1, 1}) == data[4]);
    CHECK(tensor.GetValue({2, 1}) == data[5]);
}

TEST_CASE("Tensor::RenameIndex", "[Tensor]")
{
    std::vector<std::size_t> t_shape{3, 2};
    std::vector<std::string> t_indices{"a", "b"};
    std::vector<std::string> t_indices_expected{"a", "z"};
    Tensor tensor(t_indices, t_shape);

    CHECK(tensor.GetIndices() == t_indices);
    tensor.RenameIndex(1, "z");
    CHECK(tensor.GetIndices() == t_indices_expected);
}

TEST_CASE("Tensor::SetValue", "[Tensor]")
{
    std::vector<std::size_t> t_shape{3, 2};
    std::vector<std::string> t_indices{"a", "b"};
    std::vector<c_fp32> data(6, c_fp32(0, 0));

    std::vector<c_fp32> data_expected{{0, 0}, {0, 0}, {0, 0},
                                      {0, 0}, {0, 0}, {1, 1}};

    Tensor tensor(t_indices, t_shape, data);

    tensor.SetValue({2, 1}, c_fp32(1, 1));
    CHECK(tensor.GetData() == data_expected);
}

TEST_CASE("Inline helper ShapeToSize", "[Tensor]")
{
    std::vector<std::size_t> t_shape_1{2, 3, 4};
    std::vector<std::size_t> t_shape_2{3, 4, 2};
    std::vector<std::size_t> t_shape_3{2, 2};

    CHECK(TensorHelpers::ShapeToSize(t_shape_1) == 24);
    CHECK(TensorHelpers::ShapeToSize(t_shape_2) == 24);
    CHECK(TensorHelpers::ShapeToSize(t_shape_3) == 4);
}

TEST_CASE("Inline helper MultiplyTensorData", "[Tensor]")
{
    SECTION("Matrix-vector product")
    {
        std::vector<c_fp32> t_data_left{c_fp32(0.5, 0.0), c_fp32(0.5, 0.0),
                                        c_fp32(0.5, 0.0), c_fp32(-0.5, 0.0)};
        std::vector<c_fp32> t_data_right{c_fp32(1.0, 0.0), c_fp32(0.0, 0.0)};
        std::vector<c_fp32> t_out(2, c_fp32(0, 0));

        TensorHelpers::MultiplyTensorData(t_data_left, t_data_right, t_out,
                                          {"i"}, {}, 2, 1, 2);

        CHECK(t_out[0].real() == Approx(0.5));
        CHECK(t_out[0].imag() == Approx(0.0));
        CHECK(t_out[1].real() == Approx(0.5));
        CHECK(t_out[1].imag() == Approx(0.0));
    }
    SECTION("Matrix-matrix product")
    {
        std::vector<std::size_t> t_shape_left{2, 3, 4};
        std::vector<std::size_t> t_shape_right{3, 4, 2};
        std::vector<std::size_t> t_shape_out{2, 2};

        std::vector<std::string> t_indices_left{"a", "b", "c"};
        std::vector<std::string> t_indices_left_unique{"a"};

        std::vector<std::string> t_indices_right{"b", "c", "d"};
        std::vector<std::string> t_indices_right_unique{"d"};

        std::vector<std::string> t_indices_common{"b", "c"};
        std::vector<std::string> t_indices_symdiff{"a", "d"};

        std::vector<c_fp32> t_data_left(2 * 3 * 4, c_fp32(0.5, 0.25));
        std::vector<c_fp32> t_data_right(3 * 4 * 2, c_fp32(0.5, 0.25));

        const Tensor tensor_left(t_indices_left, t_shape_left, t_data_left);
        const Tensor tensor_right(t_indices_right, t_shape_right, t_data_right);
        Tensor tensor_out(t_indices_symdiff, t_shape_out);

        size_t rows_a = 2, cols_b = 2, rows_cols_ab = 12;

        TensorHelpers::MultiplyTensorData(
            tensor_left.GetData(), tensor_right.GetData(), tensor_out.GetData(),
            t_indices_left_unique, t_indices_right_unique, rows_a, cols_b,
            rows_cols_ab);
        CHECK(tensor_out.GetData() ==
              std::vector<c_fp32>{c_fp32(2.25, 3.0), c_fp32(2.25, 3.0),
                                  c_fp32(2.25, 3.0), c_fp32(2.25, 3.0)});
    }
}

TEMPLATE_TEST_CASE("ContractTensors", "[Tensor]", c_fp32, c_fp64)
{
    SECTION("Random 2x2 (i,j) with 2x1 (i): all permutations")
    {
        Tensor<TestType> r_ij({"i", "j"}, {2, 2});
        r_ij.FillRandom();

        Tensor<TestType> r_ji({"j", "i"}, {2, 2}, r_ij.GetData());

        Tensor<TestType> s_i({"i"}, {2});
        s_i.FillRandom();

        Tensor<TestType> con_si_rij = ContractTensors(s_i, r_ij);
        Tensor<TestType> con_si_rji = ContractTensors(s_i, r_ji);
        Tensor<TestType> con_rij_si = ContractTensors(r_ij, s_i);
        Tensor<TestType> con_rji_si = ContractTensors(r_ji, s_i);

        Tensor<TestType> expected_rij_si(
            {"j"}, {2},
            {
                r_ji.GetValue({0, 0}) * s_i.GetValue({0}) +
                    r_ji.GetValue({0, 1}) * s_i.GetValue({1}),
                r_ji.GetValue({1, 0}) * s_i.GetValue({0}) +
                    r_ji.GetValue({1, 1}) * s_i.GetValue({1}),
            });
        // R_{j,i} S_i == S_i R_{i,j}
        Tensor<TestType> expected_rji_si(
            {"j"}, {2},
            {
                r_ji.GetValue({0, 0}) * s_i.GetValue({0}) +
                    r_ji.GetValue({1, 0}) * s_i.GetValue({1}),
                r_ji.GetValue({0, 1}) * s_i.GetValue({0}) +
                    r_ji.GetValue({1, 1}) * s_i.GetValue({1}),
            });

        CHECK(con_rij_si.GetData()[0].real() ==
              Approx(expected_rij_si.GetData()[0].real()));
        CHECK(con_rij_si.GetData()[0].imag() ==
              Approx(expected_rij_si.GetData()[0].imag()));
        CHECK(con_rij_si.GetData()[1].real() ==
              Approx(expected_rij_si.GetData()[1].real()));
        CHECK(con_rij_si.GetData()[1].imag() ==
              Approx(expected_rij_si.GetData()[1].imag()));

        CHECK(con_rji_si.GetData()[0].real() ==
              Approx(expected_rji_si.GetData()[0].real()));
        CHECK(con_rji_si.GetData()[0].imag() ==
              Approx(expected_rji_si.GetData()[0].imag()));
        CHECK(con_rji_si.GetData()[1].real() ==
              Approx(expected_rji_si.GetData()[1].real()));
        CHECK(con_rji_si.GetData()[1].imag() ==
              Approx(expected_rji_si.GetData()[1].imag()));

        CHECK(con_si_rij.GetData()[0].real() ==
              Approx(expected_rij_si.GetData()[0].real()));
        CHECK(con_si_rij.GetData()[0].imag() ==
              Approx(expected_rij_si.GetData()[0].imag()));
        CHECK(con_si_rij.GetData()[1].real() ==
              Approx(expected_rij_si.GetData()[1].real()));
        CHECK(con_si_rij.GetData()[1].imag() ==
              Approx(expected_rij_si.GetData()[1].imag()));

        CHECK(con_si_rji.GetData()[0].real() ==
              Approx(expected_rji_si.GetData()[0].real()));
        CHECK(con_si_rji.GetData()[0].imag() ==
              Approx(expected_rji_si.GetData()[0].imag()));
        CHECK(con_si_rji.GetData()[1].real() ==
              Approx(expected_rji_si.GetData()[1].real()));
        CHECK(con_si_rji.GetData()[1].imag() ==
              Approx(expected_rji_si.GetData()[1].imag()));
    }

    SECTION("Contract T0(a,b) and T1(b) -> T2(a)")
    {
        std::vector<std::size_t> t_shape1{2, 2};
        std::vector<std::size_t> t_shape2{2};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"b"};

        std::vector<TestType> t_data1{TestType(0.0, 0.0), TestType(1.0, 0.0),
                                      TestType(1.0, 0.0), TestType(0.0, 0.0)};
        std::vector<TestType> t_data2{TestType(1.0, 0.0), TestType(0.0, 0.0)};
        std::vector<TestType> t_data_expect{TestType(0.0, 0.0),
                                            TestType(1.0, 0.0)};

        Tensor<TestType> tensor1(t_indices1, t_shape1, t_data1);
        Tensor<TestType> tensor2(t_indices2, t_shape2, t_data2);

        Tensor<TestType> tensor3 = ContractTensors(tensor1, tensor2);
        Tensor<TestType> tensor4({"a"}, {2}, t_data_expect);

        CHECK(tensor3 == tensor4);
    }

    SECTION("Contract T0(a) and T1(a,b) -> T2(b)")
    {
        std::vector<std::size_t> t_shape1{2};
        std::vector<std::size_t> t_shape2{2, 2};

        std::vector<std::string> t_indices1{"a"};
        std::vector<std::string> t_indices2{"a", "b"};

        std::vector<TestType> t_data1{TestType(0.0, 0.0), TestType(1.0, 0.0)};
        std::vector<TestType> t_data2{TestType(0.0, 0.0), TestType(1.0, 0.0),
                                      TestType(2.0, 0.0), TestType(3.0, 0.0)};
        std::vector<TestType> t_data_expect{TestType(2.0, 0.0),
                                            TestType(3.0, 0.0)};

        Tensor<TestType> tensor1(t_indices1, t_shape1, t_data1);
        Tensor<TestType> tensor2(t_indices2, t_shape2, t_data2);

        Tensor<TestType> tensor3 = ContractTensors(tensor1, tensor2);
        Tensor<TestType> tensor4({"b"}, {2}, t_data_expect);

        CHECK(tensor3 == tensor4);
    }

    SECTION("Contract T0(a,b,c) and T1(b,c,d) -> T2(a,d)")
    {
        std::vector<std::size_t> t_shape1{2, 3, 4};
        std::vector<std::size_t> t_shape2{3, 4, 2};
        std::vector<std::size_t> t_shape3{2, 2};

        std::vector<std::string> t_indices1{"a", "b", "c"};
        std::vector<std::string> t_indices2{"b", "c", "d"};
        std::vector<std::string> t_indices3{"a", "d"};

        std::vector<TestType> t_data1(2 * 3 * 4, TestType(0.5, 0.25));
        std::vector<TestType> t_data2(3 * 4 * 2, TestType(0.5, 0.25));

        Tensor<TestType> tensor1(t_indices1, t_shape1, t_data1);
        Tensor<TestType> tensor2(t_indices2, t_shape2, t_data2);

        Tensor<TestType> tensor3 = ContractTensors(tensor1, tensor2);
        Tensor<TestType> tensor4(t_indices3, t_shape3,
                                 {TestType(2.25, 3.0), TestType(2.25, 3.0),
                                  TestType(2.25, 3.0), TestType(2.25, 3.0)});

        CHECK(tensor3 == tensor4);
    }
    SECTION("Contract T0(a,b) and T1(a,b) -> scalar")
    {
        std::vector<std::size_t> t_shape1{2, 2};
        std::vector<std::size_t> t_shape2{2, 2};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"a", "b"};

        std::vector<TestType> t_data1(2 * 2, TestType(1.0, 0.0));
        std::vector<TestType> t_data2(2 * 2, TestType(1.0, 0.0));

        Tensor<TestType> tensor1(t_indices1, t_shape1, t_data1);
        Tensor<TestType> tensor2(t_indices2, t_shape2, t_data2);

        Tensor<TestType> tensor3 = ContractTensors(tensor1, tensor2);
        Tensor<TestType> tensor4({}, {}, {TestType(4.0, 0.0)});

        CHECK(tensor3 == tensor4);
        CHECK(tensor3.IsScalar());
    }
}

TEST_CASE("Conj", "[Tensor]")
{
    std::vector<std::size_t> t_shape{2, 3};
    std::vector<std::string> t_indices{"x", "y"};
    std::vector<c_fp32> t_data(2 * 3, c_fp32(0.5, 0.25));
    std::vector<c_fp32> t_data_conj(2 * 3, c_fp32(0.5, -0.25));

    Tensor tensor(t_indices, t_shape, t_data);
    tensor = Conj(tensor);

    CHECK(tensor.GetData() == t_data_conj);
}

TEST_CASE("SliceIndex", "[Tensor]")
{
    std::vector<std::size_t> t_shape{2, 3};
    std::vector<std::string> t_indices{"x", "y"};
    std::vector<c_fp32> t_data{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

    Tensor tensor(t_indices, t_shape, t_data);

    Tensor t_x0({"y"}, {3}, std::vector<c_fp32>{{1, 0}, {2, 0}, {3, 0}});
    Tensor t_x1({"y"}, {3}, std::vector<c_fp32>{{4, 0}, {5, 0}, {6, 0}});
    Tensor t_y0({"x"}, {2}, std::vector<c_fp32>{{1, 0}, {4, 0}});
    Tensor t_y1({"x"}, {2}, std::vector<c_fp32>{{2, 0}, {5, 0}});
    Tensor t_y2({"x"}, {2}, std::vector<c_fp32>{{3, 0}, {6, 0}});

    CHECK(t_x0 == SliceIndex(tensor, "x", 0));
    CHECK(t_x1 == SliceIndex(tensor, "x", 1));

    CHECK(t_y0 == SliceIndex(tensor, "y", 0));
    CHECK(t_y1 == SliceIndex(tensor, "y", 1));
    CHECK(t_y2 == SliceIndex(tensor, "y", 2));
}

TEST_CASE("Transpose", "[Tensor]")
{
    using namespace Catch::Matchers;

    std::vector<std::size_t> t_shape{2, 3};
    std::vector<std::string> t_indices{"x", "y"};
    std::vector<c_fp32> t_data{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

    Tensor tensor(t_indices, t_shape, t_data);
    Tensor tensor_t({"y", "x"}, {3, 2},
                    {{1, 0}, {4, 0}, {2, 0}, {5, 0}, {3, 0}, {6, 0}});

    CHECK(tensor_t == Transpose(tensor, std::vector<std::string>{"y", "x"}));
    CHECK(tensor_t == Transpose(tensor, std::vector<std::size_t>{1, 0}));


    CHECK_THROWS_WITH(
        Transpose(Tensor<c_fp32>(), std::vector<std::string>{"y", "x"}),
        Contains("Number of indices cannot be zero."));
    CHECK_THROWS_WITH(
        Transpose(Tensor<c_fp32>(), std::vector<std::size_t>{1, 0}),
        Contains("Size of ordering must match number of tensor indices."));
}

TEST_CASE("AddTensors", "[Tensor]")
{
    SECTION("Scalars")
    {
        Tensor lhs({}, {}, {1});
        Tensor rhs({}, {}, {{2, 4}});

        const Tensor have_tensor = AddTensors(lhs, rhs);
        const Tensor want_tensor({}, {}, {{3, 4}});
        CHECK(have_tensor == want_tensor);
    }

    SECTION("Vectors")
    {
        Tensor lhs({"i"}, {3}, {1, {0, 2}, {3, 0}});
        Tensor rhs({"i"}, {3}, {1, {0, 2}, {0, 3}});

        const Tensor have_tensor = AddTensors(lhs, rhs);
        const Tensor want_tensor({"i"}, {3}, {2, {0, 4}, {3, 3}});
        CHECK(have_tensor == want_tensor);
    }

    SECTION("Matrices")
    {
        Tensor lhs({"i", "j"}, {2, 2}, {1, 2, 3, 4});
        Tensor rhs({"i", "j"}, {2, 2}, {{0, 1}, {0, 2}, {0, 3}, {0, 4}});

        const Tensor have_tensor = AddTensors(lhs, rhs);
        const Tensor want_tensor({"i", "j"}, {2, 2},
                                 {{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        CHECK(have_tensor == want_tensor);
    }

    SECTION("Matrices with swapped indices")
    {
        Tensor lhs({"i", "j"}, {2, 2}, {1, 2, 3, 4});
        Tensor rhs({"j", "i"}, {2, 2}, {{0, 1}, {0, 2}, {0, 3}, {0, 4}});

        const Tensor have_tensor = AddTensors(lhs, rhs);
        const Tensor want_tensor({"i", "j"}, {2, 2},
                                 {{1, 1}, {2, 3}, {3, 2}, {4, 4}});
        CHECK(have_tensor == want_tensor);
    }
}

TEST_CASE("Reshape", "[Tensor]")
{
    using namespace Catch::Matchers;

    SECTION("Equal data size")
    {
        std::vector<std::size_t> t_shape{2, 3};
        std::vector<std::string> t_indices{"x", "y"};
        std::vector<c_fp32> t_data{{1, 0}, {2, 0}, {3, 0},
                                   {4, 0}, {5, 0}, {6, 0}};

        Tensor tensor(t_indices, t_shape, t_data);
        Tensor tensor_r({"?a", "?b"}, {3, 2}, t_data);

        CHECK(tensor_r == Reshape(tensor, {3, 2}));
    }
    SECTION("Unequal data size")
    {
        std::vector<std::size_t> t_shape{2, 3};
        std::vector<std::string> t_indices{"x", "y"};
        std::vector<c_fp32> t_data{{1, 0}, {2, 0}, {3, 0},
                                   {4, 0}, {5, 0}, {6, 0}};

        Tensor tensor(t_indices, t_shape, t_data);
        Tensor tensor_r({"?a", "?b"}, {3, 2}, t_data);
        CHECK_THROWS_WITH(Reshape(tensor, {3, 3}),
                          Contains("Size is inconsistent between tensors."));
        CHECK(tensor_r.GetSize() != TensorHelpers::ShapeToSize({3, 3}));
    }
}
