#include <pybind11/pybind11.h>

#include <Jet.hpp>

#include "PathInfo.hpp"
#include "TaskBasedCpuContractor.hpp"
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

    using Tensor32 = Jet::Tensor<c_fp32_t>;
    using Tensor64 = Jet::Tensor<c_fp64_t>;

    AddBindingsForTaskBasedCpuContractor<Tensor32>(m,
                                                   "TaskBasedCpuContractorC64");
    AddBindingsForTaskBasedCpuContractor<Tensor64>(
        m, "TaskBasedCpuContractorC128");

    AddBindingsForTensor<c_fp32_t>(m, "Tensor32");
    AddBindingsForTensor<c_fp64_t>(m, "Tensor64");

    AddBindingsForTensorNetwork<Tensor32>(m, "TensorNetwork32");
    AddBindingsForTensorNetwork<Tensor64>(m, "TensorNetwork64");

    AddBindingsForTensorNetworkIO<Tensor32>(m, "TensorNetworkFile32",
                                            "TensorNetworkSerializer32");
    AddBindingsForTensorNetworkIO<Tensor64>(m, "TensorNetworkFile64",
                                            "TensorNetworkSerializer64");

    AddBindingsForPathInfo<Tensor32, Tensor64>(m);

    AddBindingsForVersion(m);
}
