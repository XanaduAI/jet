#include <pybind11/pybind11.h>

#include "python/Version.hpp"

PYBIND11_MODULE(jet, m)
{
    m.doc() = "Jet is a library for simulating quantum circuits using tensor "
              "network contractions.";

    AddBindingsForVersion(m);
}