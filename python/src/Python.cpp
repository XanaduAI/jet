#include <pybind11/pybind11.h>

#include "python/Tensor.hpp"
#include "python/Version.hpp"

PYBIND11_MODULE(jet, m)
{
    m.doc() = "Jet is a library for simulating quantum circuits using tensor "
              "network contractions.";

    using c_fp32 = std::complex<float>;
    using c_fp64 = std::complex<double>;

    AddBindingsForTensor<c_fp32>(m, "Tensor32");
    AddBindingsForTensor<c_fp64>(m, "Tensor64");
    AddBindingsForVersion(m);
}