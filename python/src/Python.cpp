#include <pybind11/pybind11.h>

#include <Jet.hpp>

#include "PathInfo.hpp"
#include "TaskBasedCpuContractor.hpp"
#include "Tensor.hpp"
#include "TensorNetwork.hpp"
#include "TensorNetworkIO.hpp"
#include "Version.hpp"

PYBIND11_MODULE(bindings, m)
{
    m.doc() = "Python bindings for the C++ tensor network contraction headers.";

    using c64_t = std::complex<float>;
    using c128_t = std::complex<double>;

    AddBindingsForPathInfo<c64_t, c128_t>(m);

    AddBindingsForTensor<c64_t>(m);
    AddBindingsForTensor<c128_t>(m);

    AddBindingsForTensorNetwork<c64_t>(m);
    AddBindingsForTensorNetwork<c128_t>(m);

    AddBindingsForTensorNetworkIO<c64_t>(m);
    AddBindingsForTensorNetworkIO<c128_t>(m);

    AddBindingsForVersion(m);
}
