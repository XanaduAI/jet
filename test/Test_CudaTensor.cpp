#include <algorithm>
#include <complex>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "jet/CudaTensor.hpp"
#include "jet/Tensor.hpp"

using c128_dev = cuDoubleComplex;
using c64_dev = cuComplex;

using c128_host = std::complex<double>;
using c64_host = std::complex<float>;

using namespace Jet;

TEMPLATE_TEST_CASE("CudaTensor::CudaTensor", "[CudaTensor]", c64_dev, c128_dev)
{

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
}

TEST_CASE("CudaTensor::GetShape", "[CudaTensor]")
{
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
    SECTION("Data: default")
    {
        CudaTensor tensor;
        std::vector<c64_host> host_data_buffer(1);
        auto ptr = reinterpret_cast<c64_dev *>(host_data_buffer.data());
        tensor.CopyGpuDataToHost(ptr);
        CHECK(host_data_buffer == std::vector<c64_host>{{0, 0}});
    }
    SECTION("Size: {2,3}, Indices: {x, y}, Data: [0.5+0.25i]*6")
    {
        std::vector<c64_dev> data(6, c64_dev{.x = 0.5, .y = 0.25});
        CudaTensor tensor({"x", "y"}, {2, 3}, data.data());

        std::vector<c64_host> host_data_complex(6, {0.5, 0.25});
        std::vector<c64_host> host_data_buffer(6);
        tensor.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(host_data_buffer.data()));
        CHECK(host_data_buffer == host_data_complex);
    }
    SECTION("CudaTensor::GetHostDataVector(), Data: default")
    {
        CudaTensor tensor;
        CHECK(tensor.GetHostDataVector() == std::vector<c64_host>{{0, 0}});
    }
    SECTION("CudaTensor::GetHostDataVector(), Size: {2,3}, Indices: {x, y}, "
            "Data: [0.5+0.25i]*6")
    {
        std::vector<c64_dev> data(6, c64_dev{.x = 0.5, .y = 0.25});
        std::vector<c64_host> host_data(6, {0.5, 0.25});

        CudaTensor tensor({"x", "y"}, {2, 3}, data.data());
        CHECK(tensor.GetHostDataVector() == host_data);
    }
}

TEST_CASE("CudaTensor::GetIndexToDimension", "[CudaTensor]")
{
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
    std::vector<size_t> t_shape{3, 2, 3};
    std::vector<std::string> t_indices{"a", "b", "c"};
    CudaTensor tensor1(t_indices, t_shape);
    CudaTensor tensor2(t_indices, t_shape);

    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom()")
    {
        tensor1.FillRandom();
        tensor2.FillRandom();
        Tensor<c64_host> tensor1_host(tensor1);
        Tensor<c64_host> tensor2_host(tensor2);
        CHECK(tensor1_host != tensor2_host);
    }
    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom(7)")
    {
        tensor1.FillRandom(7);
        tensor2.FillRandom(7);
        Tensor<c64_host> tensor1_host = static_cast<Tensor<c64_host>>(tensor1);
        Tensor<c64_host> tensor2_host = static_cast<Tensor<c64_host>>(tensor2);
        CHECK(tensor1_host == tensor2_host);
    }
    SECTION(
        "Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom() and FillRandom(5)")
    {
        tensor1.FillRandom();
        tensor2.FillRandom(5);
        Tensor<c64_host> tensor1_host = static_cast<Tensor<c64_host>>(tensor1);
        Tensor<c64_host> tensor2_host = static_cast<Tensor<c64_host>>(tensor2);
        CHECK(tensor1_host != tensor2_host);
    }
    SECTION("Size: {3,2,3}, Indices: {a,b,c}, Data: FillRandom(3) and "
            "FillRandom(5)")
    {
        tensor1.FillRandom(3);
        tensor2.FillRandom(5);
        Tensor<c64_host> tensor1_host = static_cast<Tensor<c64_host>>(tensor1);
        Tensor<c64_host> tensor2_host = static_cast<Tensor<c64_host>>(tensor2);
        CHECK(tensor1_host != tensor2_host);
    }
}

TEST_CASE("CudaTensor::CudaTensor(...)", "[CudaTensor]")
{
    SECTION("Default constructor")
    {
        CudaTensor tensor;
        CHECK(tensor.GetIndices().empty());
        CHECK(tensor.GetShape().empty());
        CHECK(tensor.GetSize() == 1);
        std::vector<c64_host> host_data(1);
        std::vector<c64_host> host_data_expect(1, {0, 0});

        tensor.CopyGpuDataToHost(reinterpret_cast<c64_dev *>(host_data.data()));
        CHECK(host_data == host_data_expect);
    }
    SECTION("Constructor, Size: {2,3}")
    {
        std::vector<size_t> shape{2, 3};
        std::vector<std::string> expect_indices{"?a", "?b"};
        CudaTensor tensor(shape);
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == expect_indices);
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Constructor, Size: {3,2}, Indices: {i,j}")
    {
        std::vector<size_t> shape{2, 3};
        std::vector<std::string> indices{"i", "j"};
        CudaTensor tensor(indices, shape);
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == indices);
        CHECK(tensor.GetSize() == 6);
    }
    SECTION("Constructor, Size: {2,2}, Indices: {i,j}, data: "
            "{{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape{2, 2};
        std::vector<std::string> indices{"i", "j"};
        std::vector<c64_dev> data{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        std::vector<c64_host> data_expected{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        CudaTensor tensor(indices, shape, data.data());
        CHECK(tensor.GetShape() == shape);
        CHECK(tensor.GetIndices() == indices);
        CHECK(tensor.GetSize() == 4);

        std::vector<c64_host> data_buffer(tensor.GetSize(), {0, 0});

        tensor.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer.data()));
        CHECK(data_buffer == data_expected);
    }
    SECTION("Copy constructor, Size: {2,2}, Indices: {i,j}, data: "
            "{{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape{2, 2};
        std::vector<std::string> indices{"i", "j"};
        std::vector<c64_dev> data{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        std::vector<c64_host> data_expected{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        CudaTensor tensor1(indices, shape, data.data());
        CudaTensor tensor2(tensor1);
        CHECK(tensor1.GetShape() == shape);
        CHECK(tensor2.GetShape() == shape);

        CHECK(tensor1.GetIndices() == indices);
        CHECK(tensor2.GetIndices() == indices);

        std::vector<c64_host> data_buffer1(tensor1.GetSize(), {0, 0});
        std::vector<c64_host> data_buffer2(tensor2.GetSize(), {0, 0});

        tensor1.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer1.data()));
        tensor2.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer2.data()));

        CHECK(data_buffer1 == data_expected);
        CHECK(data_buffer2 == data_expected);
    }
    SECTION("Copy assignment, Size: {2,2}, Indices: {i,j}, data: "
            "{{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape{2, 2};
        std::vector<std::string> indices{"i", "j"};
        std::vector<c64_dev> data{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        std::vector<c64_host> data_expected{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        CudaTensor tensor1(indices, shape, data.data());
        CudaTensor tensor2 = tensor1;
        CHECK(tensor1.GetShape() == shape);
        CHECK(tensor2.GetShape() == shape);

        CHECK(tensor1.GetIndices() == indices);
        CHECK(tensor2.GetIndices() == indices);

        std::vector<c64_host> data_buffer1(tensor1.GetSize(), {0, 0});
        std::vector<c64_host> data_buffer2(tensor2.GetSize(), {0, 0});

        tensor1.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer1.data()));
        tensor2.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer2.data()));

        CHECK(data_buffer1 == data_expected);
        CHECK(data_buffer2 == data_expected);
    }
    SECTION("Copy constructor from Jet::Tensor: {2,2}, Indices: {i,j}, data: "
            "{{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape{2, 2};
        std::vector<std::string> indices{"i", "j"};
        std::vector<c64_host> data_expected{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        std::vector<c64_host> data{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        Tensor tensor1(indices, shape, data);
        CudaTensor tensor2(tensor1);
        CHECK(tensor1.GetShape() == shape);
        CHECK(tensor2.GetShape() == shape);

        CHECK(tensor1.GetIndices() == indices);
        CHECK(tensor2.GetIndices() == ReverseVector(indices));

        std::vector<c64_host> data_buffer2(tensor2.GetSize(), {0, 0});

        tensor2.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer2.data()));

        CHECK(data_buffer2 == data_expected);
    }
    SECTION("Copy assignment from Jet::Tensor: {2,2}, Indices: {i,j}, data: "
            "{{1,2},{3,4},{5,6},{7,8}}")
    {
        std::vector<size_t> shape{2, 2};
        std::vector<std::string> indices{"i", "j"};
        std::vector<c64_host> data_expected{{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        std::vector<c64_host> data{{1, 2}, {3, 4}, {5, 6}, {7, 8}};

        Tensor tensor1(indices, shape, data);
        CudaTensor tensor2 = tensor1;
        CHECK(tensor1.GetShape() == shape);
        CHECK(tensor2.GetShape() == shape);

        CHECK(tensor1.GetIndices() == indices);
        CHECK(tensor2.GetIndices() == ReverseVector(indices));

        std::vector<c64_host> data_buffer2(tensor2.GetSize(), {0, 0});

        tensor2.CopyGpuDataToHost(
            reinterpret_cast<c64_dev *>(data_buffer2.data()));

        CHECK(data_buffer2 == data_expected);
    }
}

TEST_CASE("CudaTensor conversion to Tensor", "[CudaTensor]")
{
    SECTION("CudaTensor<cuComplex> to Tensor<complex<float>>")
    {
        CudaTensor<c64_dev> tensor_dev({"i"}, {2}, {{2, 0}, {0, 1}});
        Tensor<c64_host> tensor_host({"i"}, {2}, {{2, 0}, {0, 1}});
        auto tensor_cast = static_cast<Tensor<std::complex<float>>>(tensor_dev);
        CHECK(tensor_host == tensor_cast);
    }
}

TEST_CASE("CudaTensor::RenameIndex", "[CudaTensor]")
{
    std::vector<size_t> t_shape{3, 2};
    std::vector<std::string> t_indices{"a", "b"};
    std::vector<std::string> t_indices_expected{"a", "z"};
    CudaTensor tensor(t_indices, t_shape);

    SECTION("Rename with new index")
    {
        CHECK(tensor.GetIndices() == t_indices);
        tensor.RenameIndex(1, "z");
        CHECK(tensor.GetIndices() == t_indices_expected);
        CHECK(tensor.GetIndexToDimension().at("z") == 2);
    }
    SECTION("Rename with same index")
    {
        CHECK(tensor.GetIndices() == t_indices);
        tensor.RenameIndex(0, "a");
        CHECK(tensor.GetIndices() == t_indices);
        CHECK(tensor.GetIndexToDimension().at("a") == 3);
    }
    SECTION("Rename with existing index")
    {
        using namespace Catch::Matchers;
        CHECK(tensor.GetIndices() == t_indices);
        CHECK_THROWS_AS(tensor.RenameIndex(1, "a"), Jet::Exception);
        CHECK_THROWS_WITH(
            tensor.RenameIndex(1, "a"),
            Contains(
                "Renaming index to already existing value is not allowed."));
    }
}

TEST_CASE("CudaTensor::Reshape", "[CudaTensor]")
{
    using namespace Catch::Matchers;

    SECTION("Equal data size")
    {
        std::vector<std::size_t> t_shape{2, 3};
        std::vector<std::string> t_indices{"x", "y"};
        std::vector<c64_dev> t_data{{1, 0}, {2, 0}, {3, 0},
                                    {4, 0}, {5, 0}, {6, 0}};

        CudaTensor tensor(t_indices, t_shape, t_data);
        CudaTensor tensor_r({"?a", "?b"}, {3, 2}, t_data);
        CHECK(tensor_r.GetShape() == tensor.Reshape({3, 2}).GetShape());
        CHECK(tensor_r.GetIndices() == tensor.Reshape({3, 2}).GetIndices());
    }
    SECTION("Unequal data size")
    {
        std::vector<std::size_t> t_shape{2, 3};
        std::vector<std::string> t_indices{"x", "y"};
        std::vector<c64_dev> t_data{{1, 0}, {2, 0}, {3, 0},
                                    {4, 0}, {5, 0}, {6, 0}};

        CudaTensor tensor(t_indices, t_shape, t_data);
        CudaTensor tensor_r({"?a", "?b"}, {3, 2}, t_data);
        CHECK_THROWS_WITH(CudaTensor<>::Reshape(tensor, {3, 3}),
                          Contains("Size is inconsistent between tensors."));
        CHECK(tensor_r.GetSize() != Jet::Utilities::ShapeToSize({3, 3}));
    }
}

TEST_CASE("CudaTensor::SliceIndex", "[CudaTensor]")
{
    std::vector<std::size_t> t_shape{2, 3};
    std::vector<std::string> t_indices{"x", "y"};
    std::vector<c64_dev> t_data{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

    CudaTensor tensor(t_indices, t_shape, t_data);

    CudaTensor t_x0({"y"}, {3}, std::vector<c64_dev>{{1, 0}, {3, 0}, {5, 0}});
    CudaTensor t_x1({"y"}, {3}, std::vector<c64_dev>{{2, 0}, {4, 0}, {6, 0}});
    CudaTensor t_y0({"x"}, {2}, std::vector<c64_dev>{{1, 0}, {2, 0}});
    CudaTensor t_y1({"x"}, {2}, std::vector<c64_dev>{{3, 0}, {4, 0}});
    CudaTensor t_y2({"x"}, {2}, std::vector<c64_dev>{{5, 0}, {6, 0}});

    CHECK(static_cast<Tensor<c64_host>>(t_x0) ==
          static_cast<Tensor<c64_host>>(tensor.SliceIndex("x", 0)));
    CHECK(static_cast<Tensor<c64_host>>(t_x1) ==
          static_cast<Tensor<c64_host>>(tensor.SliceIndex("x", 1)));

    CHECK(static_cast<Tensor<c64_host>>(t_y0) ==
          static_cast<Tensor<c64_host>>(tensor.SliceIndex("y", 0)));
    CHECK(static_cast<Tensor<c64_host>>(t_y1) ==
          static_cast<Tensor<c64_host>>(tensor.SliceIndex("y", 1)));
    CHECK(static_cast<Tensor<c64_host>>(t_y2) ==
          static_cast<Tensor<c64_host>>(tensor.SliceIndex("y", 2)));
}

TEST_CASE("ContractTensors", "[CudaTensor]")
{
    SECTION("Contract T0(a,b) and T1(b) -> T2(a)")
    {
        std::vector<size_t> t_shape1{2, 2};
        std::vector<size_t> t_shape2{2};
        std::vector<size_t> t_shape3{2};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"b"};

        std::vector<c64_dev> t_data1{
            c64_dev{.x = 0.0, .y = 0.0}, c64_dev{.x = 1.0, .y = 0.0},
            c64_dev{.x = 1.0, .y = 0.0}, c64_dev{.x = 0.0, .y = 0.0}};
        std::vector<c64_dev> t_data2{c64_dev{.x = 1.0, .y = 0.0},
                                     c64_dev{.x = 0.0, .y = 0.0}};
        std::vector<c64_host> t_data_expect{c64_host(0.0, 0.0),
                                            c64_host(1.0, 0.0)};

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);

        CudaTensor tensor3 =
            CudaTensor<>::ContractTensors<c64_dev>(tensor1, tensor2);

        Tensor<c64_host> tensor3_host = static_cast<Tensor<c64_host>>(tensor3);
        Tensor<c64_host> tensor4_host({"a"}, {2}, t_data_expect);

        CHECK(tensor3_host == tensor4_host);
    }

    SECTION("Contract T0(a,b,c) and T1(b,c,d) -> T2(a,d)")
    {
        using namespace Jet::CudaTensorHelpers;

        std::vector<size_t> t_shape1{2, 3, 4};
        std::vector<size_t> t_shape2{3, 4, 2};
        std::vector<size_t> t_shape3{2, 2};

        std::vector<std::string> t_indices1{"a", "b", "c"};
        std::vector<std::string> t_indices2{"b", "c", "d"};
        std::vector<std::string> t_indices3{"a", "d"};

        std::vector<c64_dev> t_data1(2 * 3 * 4, c64_dev{.x = 0.5, .y = 0.25});
        std::vector<c64_dev> t_data2(3 * 4 * 2, c64_dev{.x = 0.5, .y = 0.25});

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);

        CudaTensor tensor3 = tensor1.ContractTensors(tensor2);

        Tensor<c64_host> tensor3_host = static_cast<Tensor<c64_host>>(tensor3);
        Tensor<c64_host> tensor4_host(
            ReverseVector(t_indices3), ReverseVector(t_shape3),
            {c64_host(2.25, 3.0), c64_host(2.25, 3.0), c64_host(2.25, 3.0),
             c64_host(2.25, 3.0)});

        CHECK(tensor3_host == tensor4_host);
    }
    SECTION("Contract T0(a,b) and T1(a,b) -> scalar")
    {
        std::vector<size_t> t_shape1{2, 2};
        std::vector<size_t> t_shape2{2, 2};
        std::vector<size_t> t_shape3{1};

        std::vector<std::string> t_indices1{"a", "b"};
        std::vector<std::string> t_indices2{"a", "b"};

        std::vector<c64_dev> t_data1(2 * 2, c64_dev{.x = 1.0, .y = 0.0});
        std::vector<c64_dev> t_data2(2 * 2, c64_dev{.x = 1.0, .y = 0.0});

        CudaTensor tensor1(t_indices1, t_shape1, t_data1);
        CudaTensor tensor2(t_indices2, t_shape2, t_data2);

        CudaTensor tensor3 =
            CudaTensor<>::ContractTensors<c64_dev>(tensor1, tensor2);

        auto tensor3_host = static_cast<Tensor<c64_host>>(tensor3);
        Tensor<c64_host> tensor4_host({}, {}, {c64_host(4.0, 0.0)});

        CHECK(tensor3_host == tensor4_host);
    }

    SECTION("Compare CudaTensor and Tensor random tensor contraction")
    {
        using namespace Jet::CudaTensorHelpers;

        std::vector<size_t> t1_shape{2, 3, 5};
        std::vector<size_t> t2_shape{5, 3, 4};
        std::vector<std::string> t1_idx{"a", "b", "c"};
        std::vector<std::string> t2_idx{"c", "b", "d"};

        CudaTensor tensor1_dev(t1_idx, t1_shape);
        CudaTensor tensor2_dev(t2_idx, t2_shape);

        Tensor tensor1_host(ReverseVector(t1_idx), ReverseVector(t1_shape));
        Tensor tensor2_host(ReverseVector(t2_idx), ReverseVector(t2_shape));

        tensor1_dev.FillRandom(7);
        tensor2_dev.FillRandom(7);
        Tensor<c64_host> tensor1_host_conv(tensor1_dev);
        Tensor<c64_host> tensor2_host_conv(tensor2_dev);

        CHECK(tensor1_host.GetIndices() == tensor1_host_conv.GetIndices());
        CHECK(tensor1_host.GetShape() == tensor1_host_conv.GetShape());

        CHECK(tensor2_host.GetIndices() == tensor2_host_conv.GetIndices());
        CHECK(tensor2_host.GetShape() == tensor2_host_conv.GetShape());

        auto tensor3_host = Tensor<c64_host>::ContractTensors(
            tensor1_host_conv, tensor2_host_conv);

        auto tensor3_dev = tensor1_dev.ContractTensors(tensor2_dev);
        Tensor<c64_host> tensor3_host_conv(tensor3_dev);

        CHECK(tensor3_host_conv.GetIndices() ==
              ReverseVector(tensor3_host.GetIndices()));
        CHECK(tensor3_host_conv.GetShape() ==
              ReverseVector(tensor3_host.GetShape()));

        const auto &data1 = tensor3_host_conv.GetData();
        const auto &data2 = tensor3_host.GetData();

        // 2x4 RowMaj to ColMaj mapping
        CHECK(data1[0].real() == Approx(data2[0].real()));
        CHECK(data1[0].imag() == Approx(data2[0].imag()));

        CHECK(data1[1].real() == Approx(data2[4].real()));
        CHECK(data1[1].imag() == Approx(data2[4].imag()));

        CHECK(data1[2].real() == Approx(data2[1].real()));
        CHECK(data1[2].imag() == Approx(data2[1].imag()));

        CHECK(data1[3].real() == Approx(data2[5].real()));
        CHECK(data1[3].imag() == Approx(data2[5].imag()));

        CHECK(data1[4].real() == Approx(data2[2].real()));
        CHECK(data1[4].imag() == Approx(data2[2].imag()));

        CHECK(data1[5].real() == Approx(data2[6].real()));
        CHECK(data1[5].imag() == Approx(data2[6].imag()));

        CHECK(data1[6].real() == Approx(data2[3].real()));
        CHECK(data1[6].imag() == Approx(data2[3].imag()));

        CHECK(data1[7].real() == Approx(data2[7].real()));
        CHECK(data1[7].imag() == Approx(data2[7].imag()));
    }
}

TEST_CASE("CudaTensor::AddTensors", "[CudaTensor]")
{
    SECTION("Scalars")
    {
        using namespace Jet::CudaTensorHelpers;

        CudaTensor lhs({}, {}, {{1, 0}});
        CudaTensor rhs({}, {}, {{2, 4}});

        const CudaTensor have_tensor = CudaTensor<>::AddTensors(lhs, rhs);
        const CudaTensor want_tensor({}, {}, {{3, 4}});

        CHECK(static_cast<Tensor<c64_host>>(have_tensor) ==
              static_cast<Tensor<c64_host>>(want_tensor));
    }

    SECTION("Vectors")
    {
        CudaTensor lhs({"i"}, {3}, {{1, 0}, {0, 2}, {3, 0}});
        CudaTensor rhs({"i"}, {3}, {{1, 0}, {0, 2}, {0, 3}});

        const CudaTensor have_tensor = lhs.AddTensor(rhs);
        const CudaTensor want_tensor({"i"}, {3}, {{2, 0}, {0, 4}, {3, 3}});
        CHECK(static_cast<Tensor<c64_host>>(have_tensor) ==
              static_cast<Tensor<c64_host>>(want_tensor));
    }

    SECTION("Matrices")
    {
        CudaTensor lhs({"i", "j"}, {2, 2}, {{1, 0}, {2, 0}, {3, 0}, {4, 0}});
        CudaTensor rhs({"i", "j"}, {2, 2}, {{0, 1}, {0, 2}, {0, 3}, {0, 4}});

        const CudaTensor have_tensor = lhs.AddTensor(rhs);
        const CudaTensor want_tensor({"i", "j"}, {2, 2},
                                     {{1, 1}, {2, 2}, {3, 3}, {4, 4}});
        CHECK(static_cast<Tensor<c64_host>>(have_tensor) ==
              static_cast<Tensor<c64_host>>(want_tensor));
    }

    SECTION("Matrices with swapped indices")
    {
        CudaTensor lhs({"i", "j"}, {2, 2}, {{1, 0}, {2, 0}, {3, 0}, {4, 0}});
        CudaTensor rhs({"j", "i"}, {2, 2}, {{0, 1}, {0, 2}, {0, 3}, {0, 4}});

        const CudaTensor have_tensor = CudaTensor<>::AddTensors(lhs, rhs);
        const CudaTensor want_tensor({"i", "j"}, {2, 2},
                                     {{1, 1}, {2, 3}, {3, 2}, {4, 4}});
        CHECK(static_cast<Tensor<c64_host>>(have_tensor) ==
              static_cast<Tensor<c64_host>>(want_tensor));
    }
}
