#include <pybind11/pybind11.h>

#include <Jet.hpp>

#include "PathInfo.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"
#include "Version.hpp"

PYBIND11_MODULE(bindings, m)
{
    m.doc() = "Python bindings for the C++ tensor network contraction headers.";

    using c64_t = std::complex<float>;
    using c128_t = std::complex<double>;

    using TensorC64 = Jet::Tensor<c64_t>;
    using TensorC128 = Jet::Tensor<c128_t>;

    AddBindingsForTensor<c64_t>(m, "TensorC64");
    AddBindingsForTensor<c128_t>(m, "TensorC128");

    AddBindingsForTensorNetwork<TensorC64>(m, "TensorNetworkC64");
    AddBindingsForTensorNetwork<TensorC128>(m, "TensorNetworkC128");

    AddBindingsForTensorNetworkIO<TensorC64>(m, "TensorNetworkFileC64",
                                             "TensorNetworkSerializerC64");
    AddBindingsForTensorNetworkIO<TensorC128>(m, "TensorNetworkFileC128",
                                              "TensorNetworkSerializerC128");

    AddBindingsForPathInfo<TensorC64, TensorC128>(m);

    AddBindingsForVersion(m);
}
