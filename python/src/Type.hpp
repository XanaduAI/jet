#pragma once

#include <complex>
#include <string>

/**
 * @brief `%Type` holds the names associated with the data type of a `Tensor`.
 */
template <class> struct Type;

template <> struct Type<std::complex<float>> {
    inline static const std::string suffix = "C64";
    inline static const std::string dtype = "complex64";
};

template <> struct Type<std::complex<double>> {
    inline static const std::string suffix = "C128";
    inline static const std::string dtype = "complex128";
};