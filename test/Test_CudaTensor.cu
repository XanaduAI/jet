#include <algorithm>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/CudaTensor.hpp"
#include "jet/Tensor.hpp"

using c_fp64_dev = cuDoubleComplex;
using c_fp32_dev = cuComplex;

using c_fp64_host = std::complex<double>;
using c_fp32_host = std::complex<float>;

TEMPLATE_TEST_CASE("CudaTensor::CudaTensor", "[CudaTensor]", c_fp32_dev, c_fp64_dev)
{
    using namespace Jet;

    SECTION("Tensor") { REQUIRE(std::is_constructible<CudaTensor<>>::value); }
    SECTION("Tensor<TestType> {}")
    {
        REQUIRE(std::is_constructible<CudaTensor<TestType>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<size_t>}")
    {
        REQUIRE(std::is_constructible<CudaTensor<TestType>,
                                      std::vector<size_t>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<std::string>, std::vector<size_t>}")
    {
        REQUIRE(std::is_constructible<CudaTensor<TestType>,
                                      std::vector<std::string>,
                                      std::vector<size_t>>::value);
    }
    SECTION("Tensor<TestType> {std::vector<std::string>, "
            "std::vector<size_t>, std::vector<TestType>}")
    {
        REQUIRE(
            std::is_constructible<CudaTensor<TestType>,
                                  std::vector<std::string>, std::vector<size_t>,
                                  std::vector<TestType>>::value);
    }
    SECTION("Require fail: Tensor<TestType> {std::vector<size_t>, "
            "std::vector<std::string>, std::vector<TestType>}")
    {
        REQUIRE_FALSE(
            std::is_constructible<CudaTensor<TestType>, std::vector<size_t>,
                                  std::vector<std::string>,
                                  std::vector<TestType>>::value);
    }
}


TEST_CASE("CudaTensor::GetShape", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Default size")
    {
        CudaTensor tensor;
        CHECK(tensor.GetShape().empty());
    }
    SECTION("Shape: {2,3}")
    {
        std::vector<size_t> shape{2, 3};
        CudaTensor tensor(shape);
        CHECK(tensor.GetShape() == shape);
    }
    SECTION("Unequal shape: {2,3} and {3,2}")
    {
        std::vector<size_t> shape_23{2, 3};
        std::vector<size_t> shape_32{3, 2};

        CudaTensor tensor_23(shape_23);
        CudaTensor tensor_32(shape_32);

        CHECK(tensor_23.GetShape() == shape_23);
        CHECK(tensor_32.GetShape() == shape_32);
        CHECK(tensor_23.GetShape() != tensor_32.GetShape());
    }
}

TEST_CASE("CudaTensor::GetSize", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Default size")
    {
        CudaTensor tensor;
        CHECK(tensor.GetSize() == 1);
    }
    SECTION("Shape: {2,3}")
    {
        CudaTensor tensor({2, 3});
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Equal size: {2,3} and {3,2}")
    {
        CudaTensor tensor_23({2, 3});
        CudaTensor tensor_32({3, 2});
        CHECK(tensor_23.GetSize() == tensor_32.GetSize());
    }
    SECTION("Unequal size: {2,3} and {3,3}")
    {
        CudaTensor tensor_23({2, 3});
        CudaTensor tensor_33({3, 3});
        CHECK(tensor_23.GetSize() != tensor_33.GetSize());
    }
}

TEST_CASE("CudaTensor::GetIndices", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Default size")
    {
        CudaTensor tensor;
        CHECK(tensor.GetIndices().empty());
    }
    SECTION("Size: {2,3}, Indices: default")
    {
        CudaTensor tensor({2, 3});
        CHECK(tensor.GetIndices() == std::vector<std::string>{"?a", "?b"});
    }
    SECTION("Size: {2,3}, Indices: {x, y}")
    {
        CudaTensor tensor({"x", "y"}, {2, 3});
        CHECK(tensor.GetIndices() == std::vector<std::string>{"x", "y"});
    }
}

TEST_CASE("CudaTensor::GetData", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Data: default")
    {
        CudaTensor tensor;
        std::vector<c_fp32_host> host_data_buffer(1);
        tensor.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(host_data_buffer.data()));
        CHECK(host_data_buffer == std::vector<c_fp32_host>{{0, 0}});
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: [0.5+0.25i]*6")
    {
        std::vector<c_fp32_dev> data(6, c_fp32_dev{.x=0.5, .y=0.25});
        CudaTensor tensor({"x", "y"}, {2, 3}, data.data());

        std::vector<c_fp32_host> host_data_complex(6, {{0.5, 0.25}});
        std::vector<c_fp32_host> host_data_buffer(6);
        tensor.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(host_data_buffer.data()));
        CHECK(host_data_buffer == host_data_complex);
    }
}

TEST_CASE("CudaTensor::GetIndexToDimension", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Default")
    {
        CudaTensor tensor;
        CHECK(tensor.GetIndexToDimension().empty());
    }
    SECTION("Size: {2,3}, Indices: default, Data: default")
    {
        CudaTensor tensor({2, 3});
        std::unordered_map<std::string, size_t> i2d{{"?a", 2}, {"?b", 3}};
        CHECK(tensor.GetIndexToDimension() == i2d);
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: default")
    {
        CudaTensor tensor({"x", "y"}, {3, 2});
        std::unordered_map<std::string, size_t> i2d{{"x", 3}, {"y", 2}};
        CHECK(tensor.GetIndexToDimension() == i2d);
    }
}


TEST_CASE("CudaTensor::FillRandom", "[CudaTensor]")
{
    using namespace Jet;

    std::vector<std::size_t> t_shape{3, 2, 3};
    std::vector<std::string> t_indices{"a", "b", "c"};
    CudaTensor tensor1(t_indices, t_shape);
    CudaTensor tensor2(t_indices, t_shape);

    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom()")
    {
        tensor1.FillRandom();
        tensor2.FillRandom();
        Tensor<c_fp32_host> tensor1_host = static_cast<Tensor<c_fp32_host>>(tensor1);
        Tensor<c_fp32_host> tensor2_host = static_cast<Tensor<c_fp32_host>>(tensor2);
        CHECK(tensor1_host != tensor2_host);
    }
    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom(7)")
    {
        tensor1.FillRandom(7);
        tensor2.FillRandom(7);
        Tensor<c_fp32_host> tensor1_host = static_cast<Tensor<c_fp32_host>>(tensor1);
        Tensor<c_fp32_host> tensor2_host = static_cast<Tensor<c_fp32_host>>(tensor2);
        CHECK(tensor1_host == tensor2_host);
    }
    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom() and FillRandom(5)")
    {
        tensor1.FillRandom();
        tensor2.FillRandom(5);
        Tensor<c_fp32_host> tensor1_host = static_cast<Tensor<c_fp32_host>>(tensor1);
        Tensor<c_fp32_host> tensor2_host = static_cast<Tensor<c_fp32_host>>(tensor2);
        CHECK(tensor1_host != tensor2_host);    
    }
    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom(3) and FillRandom(5)")
    {
        tensor1.FillRandom(3);
        tensor2.FillRandom(5);
        Tensor<c_fp32_host> tensor1_host = static_cast<Tensor<c_fp32_host>>(tensor1);
        Tensor<c_fp32_host> tensor2_host = static_cast<Tensor<c_fp32_host>>(tensor2);
        CHECK(tensor1_host != tensor2_host);   
    }
}

TEST_CASE("CudaTensor instantiation", "[CudaTensor]")
{
    using namespace Jet;
    SECTION("Default constructor")
    {
        CudaTensor tensor;
        CHECK(tensor.GetIndices().empty());
        CHECK(tensor.GetShape().empty());
        CHECK(tensor.GetSize() == 1);
        std::vector<c_fp32_host> host_data(1);
        std::vector<c_fp32_host> host_data_expect(1,{{0,0}});

        tensor.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(host_data.data()));
        CHECK(host_data == host_data_expect);
    }
    SECTION("Constructor, Size: {2,3}")
    {
        std::vector<size_t> shape {2,3};
        std::vector<std::string> expect_indices {"?a","?b"};
        CudaTensor tensor(shape);
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == expect_indices);
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Constructor, Size: {3,2}, Indices: {i,j}")
    {
        std::vector<size_t> shape {2,3};
        std::vector<std::string> indices {"i","j"};
        CudaTensor tensor(indices, shape);
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == indices);
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Constructor, Size: {2,2}, Indices: {i,j}, data: {{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape {2,2};
        std::vector<std::string> indices {"i","j"};
        std::vector<c_fp32_dev> data {{1,2},{3,4},{5,6},{7,8}};
        std::vector<c_fp32_host> data_expected {{1,2},{3,4},{5,6},{7,8}};

        CudaTensor tensor(indices, shape, data.data());
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == indices);
        CHECK(tensor.GetSize() == 4);

        std::vector<c_fp32_host> data_buffer(tensor.GetSize(), {0,0});

        tensor.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(data_buffer.data()));
        CHECK(data_buffer == data_expected);
    }
    SECTION("Copy constructor, Size: {2,2}, Indices: {i,j}, data: {{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape {2,2};
        std::vector<std::string> indices {"i","j"};
        std::vector<c_fp32_dev> data {{1,2},{3,4},{5,6},{7,8}};
        std::vector<c_fp32_host> data_expected {{1,2},{3,4},{5,6},{7,8}};

        CudaTensor tensor1(indices, shape, data.data());
        CudaTensor tensor2(tensor1);
        CHECK(tensor1.GetShape() == shape);
        CHECK(tensor2.GetShape() == shape);

        CHECK(tensor1.GetIndices() == indices);
        CHECK(tensor2.GetIndices() == indices);

        CHECK(tensor1.GetSize() == 4);
        CHECK(tensor2.GetSize() == 4);

        std::vector<c_fp32_host> data_buffer1(tensor1.GetSize(), {0,0});
        std::vector<c_fp32_host> data_buffer2(tensor2.GetSize(), {0,0});

        tensor1.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(data_buffer1.data()));
        tensor2.CopyGpuDataToHost(reinterpret_cast<c_fp32_dev*>(data_buffer2.data()));

        CHECK(data_buffer1 == data_expected);
        CHECK(data_buffer2 == data_expected);
    }

}

TEST_CASE("CudaTensor conversion to Tensor", "[CudaTensor]"){
    using namespace Jet;
    SECTION("CudaTensor<cuComplex> to Tensor<complex<float>>"){
        CudaTensor<c_fp32_dev> tensor_dev({"i"}, {2}, {{2,0},{0,1}});
        Tensor<c_fp32_host> tensor_host({"i"}, {2}, {{2,0},{0,1}});
        auto tensor_cast = static_cast<Tensor<std::complex<float>>>(tensor_dev);
        CHECK(tensor_host == tensor_cast);
    }
}

TEST_CASE("CudaTensor::RenameIndex", "[CudaTensor]")
{
    using namespace Jet;

    std::vector<std::size_t> t_shape{3, 2};
    std::vector<std::string> t_indices{"a", "b"};
    std::vector<std::string> t_indices_expected{"a", "z"};
    CudaTensor tensor(t_indices, t_shape);

    CHECK(tensor.GetIndices() == t_indices);
    tensor.RenameIndex(1, "z");
    CHECK(tensor.GetIndices() == t_indices_expected);
}


TEST_CASE("ContractTensors", "[CudaTensor]")
{
    using namespace Jet;

    SECTION("Random 2x2 (i,j) with 2x1 (i): all permutations")
    {

        CudaTensor r_ij({"i", "j"}, {2, 2});
        r_ij.FillRandom();
        Tensor<c_fp32_host> r_ij_host = static_cast<Tensor<c_fp32_host>>(r_ij);

        CudaTensor r_ji({"j", "i"}, {2, 2}, r_ij.GetData());
        Tensor<c_fp32_host> r_ji_host = static_cast<Tensor<c_fp32_host>>(r_ji);


        CudaTensor s_i({"i"}, {2});
        s_i.FillRandom();
        Tensor s_i_host = static_cast<Tensor<c_fp32_host>>(s_i);

        CudaTensor con_si_rij = ContractTensors(s_i, r_ij);
        CudaTensor con_si_rji = ContractTensors(s_i, r_ji);
        CudaTensor con_rij_si = ContractTensors(r_ij, s_i);
        CudaTensor con_rji_si = ContractTensors(r_ji, s_i);

        Tensor<c_fp32_host> con_si_rij_host = static_cast<Tensor<c_fp32_host>>(con_si_rij);
        Tensor<c_fp32_host> con_si_rji_host = static_cast<Tensor<c_fp32_host>>(con_si_rji);
        Tensor<c_fp32_host> con_rij_si_host = static_cast<Tensor<c_fp32_host>>(con_rij_si);
        Tensor<c_fp32_host> con_rji_si_host = static_cast<Tensor<c_fp32_host>>(con_rji_si);

        Tensor<c_fp32_host> expected_rij_si_host(
            {"j"}, {2},
            {
                r_ji_host.GetValue({0, 0}) * s_i_host.GetValue({0}) +
                    r_ji_host.GetValue({0, 1}) * s_i_host.GetValue({1}),
                r_ji_host.GetValue({1, 0}) * s_i_host.GetValue({0}) +
                    r_ji_host.GetValue({1, 1}) * s_i_host.GetValue({1}),
            });
        // R_{j,i} S_i == S_i R_{i,j}
        Tensor expected_rji_si_host(
            {"j"}, {2},
            {
                r_ji_host.GetValue({0, 0}) * s_i_host.GetValue({0}) +
                    r_ji_host.GetValue({1, 0}) * s_i_host.GetValue({1}),
                r_ji_host.GetValue({0, 1}) * s_i_host.GetValue({0}) +
                    r_ji_host.GetValue({1, 1}) * s_i_host.GetValue({1}),
            });

        std::cout << "#####################" << std::endl;
        std::cout << con_si_rij_host << std::endl;
        std::cout << con_si_rji_host << std::endl;
        std::cout << con_rij_si_host << std::endl;
        std::cout << con_rji_si_host << std::endl;
        std::cout << "#####################" << std::endl;
        std::cout << expected_rij_si_host << std::endl;
        std::cout << expected_rji_si_host << std::endl;
        std::cout << "#####################" << std::endl;

        CHECK(con_rij_si_host.GetData()[0].real() ==
              Approx(expected_rij_si_host.GetData()[0].real()));
        CHECK(con_rij_si_host.GetData()[0].imag() ==
              Approx(expected_rij_si_host.GetData()[0].imag()));
        CHECK(con_rij_si_host.GetData()[1].real() ==
              Approx(expected_rij_si_host.GetData()[1].real()));
        CHECK(con_rij_si_host.GetData()[1].imag() ==
              Approx(expected_rij_si_host.GetData()[1].imag()));

        CHECK(con_rji_si_host.GetData()[0].real() ==
              Approx(expected_rji_si_host.GetData()[0].real()));
        CHECK(con_rji_si_host.GetData()[0].imag() ==
              Approx(expected_rji_si_host.GetData()[0].imag()));
        CHECK(con_rji_si_host.GetData()[1].real() ==
              Approx(expected_rji_si_host.GetData()[1].real()));
        CHECK(con_rji_si_host.GetData()[1].imag() ==
              Approx(expected_rji_si_host.GetData()[1].imag()));

        CHECK(con_si_rij_host.GetData()[0].real() ==
              Approx(expected_rji_si_host.GetData()[0].real()));
        CHECK(con_si_rij_host.GetData()[0].imag() ==
              Approx(expected_rji_si_host.GetData()[0].imag()));
        CHECK(con_si_rij_host.GetData()[1].real() ==
              Approx(expected_rji_si_host.GetData()[1].real()));
        CHECK(con_si_rij_host.GetData()[1].imag() ==
              Approx(expected_rji_si_host.GetData()[1].imag()));

        CHECK(con_si_rji_host.GetData()[0].real() ==
              Approx(expected_rij_si_host.GetData()[0].real()));
        CHECK(con_si_rji_host.GetData()[0].imag() ==
              Approx(expected_rij_si_host.GetData()[0].imag()));
        CHECK(con_si_rji_host.GetData()[1].real() ==
              Approx(expected_rij_si_host.GetData()[1].real()));
        CHECK(con_si_rji_host.GetData()[1].imag() ==
              Approx(expected_rij_si_host.GetData()[1].imag()));
    }

    SECTION("Contract T0(a,b) and T1(b) -> T2(a)")
    {
        std::vector<std::size_t> t_shape1{2, 2};
        std::vector<std::size_t> t_shape2{2};
        std::vector<std::size_t> t_shape3{2};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"b"};

        std::vector<c_fp32_dev> t_data1{ 
                                    c_fp32_dev{ .x=0.0, .y=0.0}, c_fp32_dev{ .x=1.0, .y=0.0},
                                    c_fp32_dev{ .x=1.0, .y=0.0}, c_fp32_dev{ .x=0.0, .y=0.0}
                                };
        std::vector<c_fp32_dev> t_data2{ c_fp32_dev{.x=1.0, .y=0.0}, c_fp32_dev{.x=0.0, .y=0.0}};
        std::vector<c_fp32_host> t_data_expect{c_fp32_host(0.0, 0.0),
                                            c_fp32_host(1.0, 0.0)};

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);
        CudaTensor tensor3 = ContractTensors(tensor1, tensor2);

        Tensor<c_fp32_host> tensor3_host = static_cast<Tensor<c_fp32_host>>(tensor3);
        Tensor<c_fp32_host> tensor4_host({"a"}, {2}, t_data_expect);

        CHECK(tensor3_host == tensor4_host);
    }

    SECTION("Contract T0(a,b,c) and T1(b,c,d) -> T2(a,d)")
    {
        std::vector<std::size_t> t_shape1{2, 3, 4};
        std::vector<std::size_t> t_shape2{3, 4, 2};
        std::vector<std::size_t> t_shape3{2, 2};

        std::vector<std::string> t_indices1{"a", "b", "c"};
        std::vector<std::string> t_indices2{"b", "c", "d"};
        std::vector<std::string> t_indices3{"a", "d"};

        std::vector<c_fp32_dev> t_data1(2 * 3 * 4, c_fp32_dev{.x=0.5, .y=0.25});
        std::vector<c_fp32_dev> t_data2(3 * 4 * 2, c_fp32_dev{.x=0.5, .y=0.25});

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);

        CudaTensor tensor3 = ContractTensors(tensor1, tensor2);

        Tensor<c_fp32_host> tensor3_host = static_cast<Tensor<c_fp32_host>>(tensor3);
        Tensor<c_fp32_host> tensor4_host(t_indices3, t_shape3,
                                 {c_fp32_host(2.25, 3.0), c_fp32_host(2.25, 3.0),
                                  c_fp32_host(2.25, 3.0), c_fp32_host(2.25, 3.0)});

        CHECK(tensor3_host == tensor4_host);
    }
    SECTION("Contract T0(a,b) and T1(a,b) -> scalar")
    {
        std::vector<std::size_t> t_shape1{2, 2};
        std::vector<std::size_t> t_shape2{2, 2};
        std::vector<std::size_t> t_shape3{1};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"a", "b"};

        std::vector<c_fp32_dev> t_data1(2 * 2, c_fp32_dev{.x=1.0, .y=0.0});
        std::vector<c_fp32_dev> t_data2(2 * 2, c_fp32_dev{.x=1.0, .y=0.0});

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);

        CudaTensor tensor3 = ContractTensors(tensor1, tensor2);
        auto tensor3_host = static_cast<Tensor<c_fp32_host>>(tensor3);
        Tensor<c_fp32_host> tensor4_host({}, {}, {c_fp32_host(4.0, 0.0)});

        CHECK(tensor3_host == tensor4_host);
    }
}