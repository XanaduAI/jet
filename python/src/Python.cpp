#include <pybind11/pybind11.h>

#include "Tensor.hpp"
#include "Version.hpp"

PYBIND11_MODULE(jet, m)
{
    m.doc() = "Jet is a library for simulating quantum circuits using tensor "
              "network contractions.";

    using c_fp32_t = std::complex<float>;
    using c_fp64_t = std::complex<double>;

    AddBindingsForTensor<c_fp32_t>(m, "Tensor32");
    AddBindingsForTensor<c_fp64_t>(m, "Tensor64");

    AddBindingsForVersion(m);
}