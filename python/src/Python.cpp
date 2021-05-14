#include <pybind11/pybind11.h>

#include <Jet.hpp>

#include "PathInfo.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"
#include "Version.hpp"

PYBIND11_MODULE(jet, m)
{
    m.doc() = "Jet is a library for simulating quantum circuits using tensor "
              "network contractions.";

    using c_fp32_t = std::complex<float>;
    using c_fp64_t = std::complex<double>;

    AddBindingsForTensor<c_fp32_t>(m, "Tensor32");
    AddBindingsForTensor<c_fp64_t>(m, "Tensor64");

    using tensor32_t = Jet::Tensor<c_fp32_t>;
    using tensor64_t = Jet::Tensor<c_fp64_t>;

    AddBindingsForTensorNetwork<tensor32_t>(m, "TensorNetwork32");
    AddBindingsForTensorNetwork<tensor64_t>(m, "TensorNetwork64");

    AddBindingsForTensorNetworkIO<tensor32_t>(m, "TensorNetworkFile32",
                                              "TensorNetworkSerializer32");
    AddBindingsForTensorNetworkIO<tensor64_t>(m, "TensorNetworkFile64",
                                              "TensorNetworkSerializer64");

    AddBindingsForPathInfo<tensor32_t, tensor64_t>(m);

    AddBindingsForVersion(m);
}
