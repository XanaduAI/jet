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

    using f32_t = float;
    using f64_t = double;
    using c64_t = std::complex<float>;
    using c128_t = std::complex<double>;

    AddBindingsForPathInfo<f32_t, f64_t, c64_t, c128_t>(m);

    AddBindingsForTaskBasedCpuContractor<f32_t>(m);
    AddBindingsForTaskBasedCpuContractor<f64_t>(m);
    AddBindingsForTaskBasedCpuContractor<c64_t>(m);
    AddBindingsForTaskBasedCpuContractor<c128_t>(m);

    AddBindingsForTensor<f32_t>(m);
    AddBindingsForTensor<f64_t>(m);
    AddBindingsForTensor<c64_t>(m);
    AddBindingsForTensor<c128_t>(m);

    AddBindingsForTensorNetwork<f32_t>(m);
    AddBindingsForTensorNetwork<f64_t>(m);
    AddBindingsForTensorNetwork<c64_t>(m);
    AddBindingsForTensorNetwork<c128_t>(m);

    AddBindingsForTensorNetworkIO<c64_t>(m);
    AddBindingsForTensorNetworkIO<c128_t>(m);

    AddBindingsForVersion(m);
}
