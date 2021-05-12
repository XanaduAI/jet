#include <pybind11/pybind11.h>

#include "PathInfo.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "Version.hpp"

PYBIND11_MODULE(jet, m)
{
    m.doc() = "Jet is a library for simulating quantum circuits using tensor "
              "network contractions.";

    using c_fp32 = std::complex<float>;
    using c_fp64 = std::complex<double>;

    AddBindingsForTensor<c_fp32>(m, "Tensor32");
    AddBindingsForTensor<c_fp64>(m, "Tensor64");

    AddBindingsForTensorNetwork<Jet::Tensor<c_fp32>>(m, "TensorNetwork32");
    AddBindingsForTensorNetwork<Jet::Tensor<c_fp64>>(m, "TensorNetwork64");

    AddBindingsForPathInfo<Jet::Tensor<c_fp32>, Jet::Tensor<c_fp64>>(m);

    AddBindingsForVersion(m);
}
