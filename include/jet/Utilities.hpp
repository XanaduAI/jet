#pragma once

#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Abort.hpp"

namespace Jet {
namespace Utilities {

/**
 * @brief Determines if an integral value is a power of 2.
 * @param value Number to check.
 * @return True if `value` is a power of 2.
 */
constexpr inline bool is_pow_2(size_t value)
{
    return static_cast<bool>(value && !(value & (value - 1)));
}

/**
 * @brief Finds the log2 value of a known power of 2, otherwise finds the floor
 * of log2 of the operand.
 *
 * This works by counting the highest set bit in a size_t by examining the
 * number of leading zeros. This value can then be subtracted from the
 * number of bits in the size_t value to yield the log2 value.
 *
 * @param value Value to calculate log2 of. If 0, the result is undefined
 * @return size_t log2 result of value. If value is a non power-of-2, returns
 * the floor of the log2 operation.
 */
constexpr inline size_t fast_log2(size_t value);

#if defined(__GNUC__)

constexpr inline size_t fast_log2(size_t value)
{
    return static_cast<size_t>(std::numeric_limits<size_t>::digits -
                               __builtin_clzll((value)) - 1ULL);
}

#elif defined(_MSC_VER)

#include <intrin.h>

#if defined(_M_X64)

constexpr inline size_t fast_log2(size_t value)
{
    unsigned long idx;
    _BitScanReverse64(&idx, value);

    return static_cast<size_t>(idx);
}

#else

constexpr inline size_t fast_log2(size_t value)
{
    unsigned long idx;
    _BitScanReverse(&idx, value);

    return static_cast<size_t>(idx);
}

#endif // defined(_M_X64)

#endif // defined(__GNUC__)

/**
 * Streams a pair of elements to an output stream.
 *
 * @tparam T1 Type of the first element in the pair.
 * @tparam T2 Type of the second element in the pair.
 * @param os Output stream to be modified.
 * @param p Pair to be inserted.
 * @return Reference to the given output stream.
 */
template <class T1, class T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p)
{
    return os << '{' << p.first << ',' << p.second << '}';
}

/**
 * Streams a vector to an output stream.
 *
 * @tparam T Type of the elements in the vector.
 * @param os Output stream to be modified.
 * @param v Vector to be inserted.
 * @return Reference to the given output stream.
 */
template <class T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << '{';
    for (size_t i = 0; i < v.size(); i++) {
        if (i != 0) {
            os << "  ";
        }
        os << v[i];
    }
    os << '}';
    return os;
}

/**
 * Converts an ID into a unique string index of the form [a-zA-Z][0-9]*.
 *
 * @param id ID to be converted.
 * @return String index associated with the ID.
 */
inline std::string GenerateStringIndex(size_t id)
{
    static const std::vector<std::string> alphabet = {
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    const size_t div_id = id / alphabet.size();
    const std::string prefix = alphabet[id % alphabet.size()];
    const std::string suffix = (div_id == 0) ? "" : std::to_string(div_id - 1);
    return prefix + suffix;
}

/**
 * Computes the order (i.e., number of rows and columns) of the given square
 * matrix.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @return Order of the matrix.
 */
template <class scalar_type_t>
inline size_t Order(const std::vector<std::complex<scalar_type_t>> &mat)
{
    // If sqrt() returns a value just under the true square root, increment n.
    size_t n = static_cast<size_t>(sqrt(mat.size()));
    n += n * n != mat.size();
    return n;
}

/**
 * Returns the `n` x `n` complex-valued identity matrix.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param n Order of the desired identity matrix.
 * @return Vector representing the desired matrix, encoded in row-major order.
 */
template <class scalar_type_t>
inline std::vector<std::complex<scalar_type_t>> Eye(size_t n)
{
    std::vector<std::complex<scalar_type_t>> eye(n * n, 0);
    for (size_t i = 0; i < n; i++) {
        eye[i * n + i] = 1;
    }
    return eye;
}

/**
 * Multiplies the two given square matrices.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param m1 Matrix on the LHS of the multiplication.
 * @param m2 Matrix on the RHS of the multiplication.
 * @param n Order of the two matrices.
 * @return Matrix representing the product of `m1` and `m2`.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
MultiplySquareMatrices(const std::vector<std::complex<scalar_type_t>> &m1,
                       const std::vector<std::complex<scalar_type_t>> &m2,
                       size_t n)
{
    JET_ABORT_IF_NOT(m1.size() == n * n, "LHS matrix has the wrong order");
    JET_ABORT_IF_NOT(m2.size() == n * n, "RHS matrix has the wrong order");

    std::vector<std::complex<scalar_type_t>> product(n * n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                product[i * n + j] += m1[i * n + k] * m2[k * n + j];
            }
        }
    }
    return product;
}

/**
 * Raises the given matrix to the specified exponent.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param mat Matrix at the base of the power.
 * @param k Exponent of the power.
 * @return Matrix representing `mat` raised to the power of `k`.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
Pow(const std::vector<std::complex<scalar_type_t>> &mat, size_t k)
{
    if (k == 1) {
        return mat;
    }

    const auto n = Order(mat);
    if (k == 0) {
        return Eye<scalar_type_t>(n);
    }

    std::vector<std::complex<scalar_type_t>> power = mat;
    for (size_t i = 2; i <= k; i++) {
        power = MultiplySquareMatrices(power, mat, n);
    }
    return power;
}

/**
 * Adds the given matrices.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param m1 Matrix on the LHS of the addition.
 * @param m2 Matrix on the RHS of the addition.
 * @return Matrix representing the sum of `m1` and `m2`.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
operator+(const std::vector<std::complex<scalar_type_t>> &m1,
          const std::vector<std::complex<scalar_type_t>> &m2)
{
    JET_ABORT_IF_NOT(m1.size() == m2.size(), "Matrices have different sizes");

    std::vector<std::complex<scalar_type_t>> sum(m1.size());
    for (size_t i = 0; i < m1.size(); i++) {
        sum[i] = m1[i] + m2[i];
    }
    return sum;
}

/**
 * Subtracts the given matrices.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param m1 Matrix on the LHS of the subtraction.
 * @param m2 Matrix on the RHS of the subtraction.
 * @return Matrix representing `m2` subtracted from `m1`.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
operator-(const std::vector<std::complex<scalar_type_t>> &m1,
          const std::vector<std::complex<scalar_type_t>> &m2)
{
    JET_ABORT_IF_NOT(m1.size() == m2.size(), "Matrices have different sizes");

    std::vector<std::complex<scalar_type_t>> diff(m1.size());
    for (size_t i = 0; i < m1.size(); i++) {
        diff[i] = m1[i] - m2[i];
    }
    return diff;
}

/**
 * Returns the product of the given scalar and matrix.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param mat Matrix to be scaled.
 * @param c Scalar to be applied to the matrix.
 * @return Matrix representing the scalar product of `c` and `mat`.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
operator*(const std::vector<std::complex<scalar_type_t>> &mat,
          std::complex<scalar_type_t> c)
{
    std::vector<std::complex<scalar_type_t>> product = mat;
    for (size_t i = 0; i < product.size(); i++) {
        product[i] *= c;
    }
    return product;
}

/**
 * Returns a diagonal matrix with the same dimensions of the given matrix where
 * each entry along the main diagonal is derived by applying std::exp() to the
 * corresponding entry in the given matrix.
 *
 * @tparam scalar_type_t Template parameter of std::complex.
 * @param mat Matrix to be converted into a diagonal matrix.
 * @return Matrix representing the diagonal exponentation of the given matrix.
 */
template <typename scalar_type_t>
inline std::vector<std::complex<scalar_type_t>>
DiagExp(const std::vector<std::complex<scalar_type_t>> &mat)
{
    const auto n = Order(mat);
    std::vector<std::complex<scalar_type_t>> diag(mat.size(), 0);
    for (size_t i = 0; i < n; i++) {
        diag[i * n + i] = std::exp(mat[i * n + i]);
    }
    return diag;
}

/**
 * Returns a diagonal tensor with the given main diagonal.
 *
 * @tparam Tensor Type of the tensor.
 * @tparam T Type of the diagonal entries.
 * @param vec Entries to be copied to the main diagonal of the tensor.
 * @return Tensor with the given main diagonal.
 */
template <typename Tensor, typename T>
inline Tensor DiagMatrix(const std::vector<T> &vec)
{
    const size_t n = vec.size();
    Tensor tens({n, n});
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            tens.SetValue({i, j}, (i == j) ? vec[i] : 0.0);
        }
    }
    return tens;
}

/**
 * Reports whether the given element is in the provided vector.
 *
 * @tparam T Type of the elements in the vector.
 * @param e Element being searched for.
 * @param v Vector to be searched.
 * @return True if the element is in the vector.
 */
template <class T> inline bool InVector(const T &e, const std::vector<T> &v)
{
    return std::find(v.cbegin(), v.cend(), e) != v.cend();
}

/**
 * Computes the intersection of two small vectors.
 *
 * @tparam T Type of the elements in the vectors.
 * @param v1 Vector on the LHS of the intersection.
 * @param v2 Vector on the RHS of the intersection.
 * @return Vector containing the elements in both vectors.
 */
template <typename T>
inline std::vector<T> VectorIntersection(const std::vector<T> &v1,
                                         const std::vector<T> &v2)
{
    std::vector<T> result;
    for (const auto &value : v1) {
        if (InVector(value, v2)) {
            result.emplace_back(value);
        }
    }
    return result;
}

/**
 * Computes the union of two small vectors.
 *
 * @tparam T Type of the elements in the vectors.
 * @param v1 Vector on the LHS of the union.
 * @param v2 Vector on the RHS of the union.
 * @return Vector containing the elements in at least one vector.
 */
template <typename T>
inline std::vector<T> VectorUnion(const std::vector<T> &v1,
                                  const std::vector<T> &v2)
{
    std::vector<T> result = v1;
    for (const auto &value : v2) {
        if (!InVector(value, v1)) {
            result.emplace_back(value);
        }
    }
    return result;
}

/**
 * Computes the difference of two small vectors.
 *
 * @tparam T Type of the elements in the vectors.
 * @param v1 Vector on the LHS of the difference.
 * @param v2 Vector on the RHS of the difference.
 * @return Vector containing the elements only in the first vector.
 */
template <typename T>
inline std::vector<T> VectorSubtraction(const std::vector<T> &v1,
                                        const std::vector<T> &v2)
{
    std::vector<T> result;
    for (const auto &value : v1) {
        if (!InVector(value, v2)) {
            result.emplace_back(value);
        }
    }
    return result;
}

/**
 * Computes the disjunctive union of two small vectors.
 *
 * @tparam T Type of the elements in the vectors.
 * @param v1 Vector on the LHS of the disjunctive union.
 * @param v2 Vector on the RHS of the disjunctive union.
 * @return Vector containing the elements in exactly one of the vectors.
 */
template <typename T>
inline std::vector<T> VectorDisjunctiveUnion(const std::vector<T> &v1,
                                             const std::vector<T> &v2)
{
    return VectorSubtraction(VectorUnion(v1, v2), VectorIntersection(v1, v2));
}

/**
 * Concatenates the strings in the given vector (using "" as the separator).
 *
 * @param v Vector to be concatenated.
 * @return String representing the contatentation of the vector contents.
 */
inline std::string JoinStringVector(const std::vector<std::string> &v)
{
    return std::accumulate(v.begin(), v.end(), std::string(""));
}

/**
 * Concatenates the elements in the two given vectors.
 *
 * @tparam T Type of the elements in the vectors.
 * @param v1 Prefix of the concatenation.
 * @param v2 Suffix of the concatenation.
 * @return Vector representing the contatentation of `v1` and `v2`.
 */
template <typename T>
inline std::vector<T> VectorConcatenation(const std::vector<T> &v1,
                                          const std::vector<T> &v2)
{
    std::vector<T> concat = v1;
    concat.insert(concat.end(), v2.begin(), v2.end());
    return concat;
}

/**
 * Returns the factorial of the given number.
 *
 * @warning This function is susceptible to overflow errors for large values of
 *          `n`.
 *
 * @param n Number whose factorial is to be computed.
 * @return Factorial of the given number.
 */
inline size_t Factorial(size_t n)
{
    size_t prod = 1;
    for (size_t i = 2; i <= n; i++) {
        prod *= i;
    }
    return prod;
}

/**
 * @brief Returns the size of a shape.
 *
 * @param shape Index dimensions.
 * @return Product of the index dimensions in the shape.
 */
inline size_t ShapeToSize(const std::vector<size_t> &shape)
{
    size_t size = 1;
    for (const auto &dim : shape) {
        size *= dim;
    }
    return size;
}

/**
 * @brief Converts a linear index into a multi-dimensional index.
 *
 * The multi-dimensional index is written in row-major order.
 *
 * Example: To compute the multi-index (i, j) of an element in a 2x2 matrix
 *          given a linear index of 2, `shape` would be {2, 2} and the result
 *          would be `{1, 0}`.
 *          \code{.cpp}
 *     std::vector<size_t> multi_index = UnravelIndex(2, {2, 2});  // {1, 0}
 *          \endcode
 *
 * @param index Linear index to be unraveled.
 * @param shape Size of each index dimension.
 * @return Multi-index associated with the linear index.
 */
inline std::vector<size_t> UnravelIndex(unsigned long long index,
                                        const std::vector<size_t> &shape)
{
    const size_t size = ShapeToSize(shape);
    JET_ABORT_IF(size <= index, "Linear index does not fit in the shape.");

    std::vector<size_t> multi_index(shape.size());
    for (int i = multi_index.size() - 1; i >= 0; i--) {
        multi_index[i] = index % shape[i];
        index /= shape[i];
    }
    return multi_index;
}

/**
 * @brief Converts a multi-dimensional index into a linear index.
 *
 * @note This function is the inverse of UnravelIndex().
 *
 * @param index Multi-index to be raveled, expressed in row-major order.
 * @param shape Size of each index dimension.
 * @return Linear index associated with the multi-index.
 */
inline unsigned long long RavelIndex(const std::vector<size_t> &index,
                                     const std::vector<size_t> &shape)
{
    JET_ABORT_IF_NOT(index.size() == shape.size(),
                     "Number of index and shape dimensions must match.");

    size_t multiplier = 1;

    unsigned long long linear_index = 0;
    for (int i = index.size() - 1; i >= 0; i--) {
        JET_ABORT_IF(index[i] >= shape[i], "Index does not fit in the shape.");
        linear_index += index[i] * multiplier;
        multiplier *= shape[i];
    }
    return linear_index;
}

/**
 * Splits `s` (at most once) on the given delimiter and stores the result in the
 * provided vector.  If an instance of the delimiter is found, the contents of
 * `s` up to (and including) the first occurrence of the delimiter are erased.
 *
 * @param s String to be split.
 * @param delimiter Delimeter separating each part of the split string.
 * @param tokens Vector to store the result of the split.
 */
inline void SplitStringOnDelimiter(std::string &s, const std::string &delimiter,
                                   std::vector<std::string> &tokens)
{
    const size_t pos = s.find(delimiter);
    if (pos == std::string::npos) {
        tokens.emplace_back(s);
        return;
    }

    const auto token = s.substr(0, pos);
    tokens.emplace_back(token);

    s.erase(0, pos + delimiter.length());
    tokens.emplace_back(s);
}

/**
 * Splits `s` (at most once) on each of the given delimiters (in order).
 *
 * @warning All spaces are removed from `s` prior to performing the split.
 *
 * @param s String to be split.
 * @param delimiters Delimiters separating each part of the split string.
 * @return Vector containing the result of the split (excluding empty tokens).
 */
inline std::vector<std::string>
SplitStringOnMultipleDelimiters(std::string s,
                                const std::vector<std::string> &delimiters)
{
    // Remove spaces.
    s.erase(std::remove_if(s.begin(), s.end(), isspace), s.end());

    std::vector<std::string> tokens = {s};
    for (std::size_t i = 0; i < delimiters.size(); i++) {
        tokens.pop_back();
        SplitStringOnDelimiter(s, delimiters[i], tokens);
    }

    // Remove empty tokens.
    const auto empty = [](const std::string &token) { return token.empty(); };
    tokens.erase(std::remove_if(tokens.begin(), tokens.end(), empty),
                 tokens.end());

    return tokens;
}

/**
 * Splits `s` on the given delimiter (as many times as possible) and stores the
 * result in the provided vector.
 *
 * @param s String to be split.
 * @param delimiter Delimeter separating each part of the split string.
 * @param tokens Vector to store the result of the split.
 */
inline void SplitStringOnDelimiterRecursively(const std::string &s,
                                              const std::string &delimiter,
                                              std::vector<std::string> &tokens)
{
    const size_t pos = s.find(delimiter);
    if (pos == std::string::npos) {
        tokens.emplace_back(s);
    }
    else {
        tokens.emplace_back(s.begin(), s.begin() + pos);
        const auto remaining = s.substr(pos + delimiter.length());
        SplitStringOnDelimiterRecursively(remaining, delimiter, tokens);
    }
}

/**
 * Replaces each occurrence of `from` with `to` in the given string.
 *
 * @param s String to be searched for occurrences of `from`.
 * @param from Substring to be replaced in `s`.
 * @param to Replacement string for `from`.
 */
inline void ReplaceAllInString(std::string &s, const std::string &from,
                               const std::string &to)
{
    JET_ABORT_IF(from.empty(), "Cannot replace occurrences of an empty string");

    size_t pos = s.find(from, 0);
    while (pos != std::string::npos) {
        s.replace(pos, from.length(), to);
        // Skip over `to` in case part of it matches `from`.
        pos = s.find(from, pos + to.length());
    }
}

/**
 * Returns the total amount of system memory as reported by /proc/meminfo.
 *
 * Adapted from
 * https://github.com/Russellislam08/RAMLogger/blob/master/readproc.h
 *
 * @return Total amount of system memory (in kB).
 *         If an error occurs, -1 is returned.
 */
inline int GetTotalMemory()
{
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == NULL) {
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
        int memTotal;
        if (sscanf(line, "MemTotal: %d kB", &memTotal) == 1) {
            fclose(meminfo);
            return memTotal;
        }
    }

    // Getting here means we were not able to find what we were looking for
    fclose(meminfo);
    return -1;
}

/**
 * Returns the amount of available system memory as reported by /proc/meminfo.
 *
 * @see GetTotalMemory()
 *
 * @return Amount of available system memory (in kB).
 *         If an error occurs, -1 is returned.
 */
inline int GetAvailableMemory()
{
    /* Same function as above but it parses the meminfo file
       in order to obtain the current amount of physical memory available
    */
    FILE *meminfo = fopen("/proc/meminfo", "r");
    if (meminfo == NULL) {
        return -1;
    }

    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
        int memAvail;
        if (sscanf(line, "MemAvailable: %d kB", &memAvail) == 1) {
            fclose(meminfo);
            return memAvail;
        }
    }

    fclose(meminfo);
    return -1;
}

/**
 * Use OpenMP when available to copy
 *
 * @param a vector to copy
 * @param b vector to copy to
 */
template <typename T> void FastCopy(const std::vector<T> &a, std::vector<T> &b)
{
#ifdef _OPENMP
    size_t max_right_dim = 1024;
    size_t size = a.size();
    if (b.size() != size)
        b.resize(size);
#pragma omp parallel for schedule(static, max_right_dim)
    for (std::size_t p = 0; p < size; ++p) {
        b[p] = a[p];
    }
#else
    b = a;
#endif
}

/**
 * Determine if vector w contains v
 *
 * @param v
 * @param w
 *
 * @return true if every element of v is in w
 */

template <typename T>
bool VectorInVector(const std::vector<T> &v, const std::vector<T> &w)
{
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (!InVector(v[i], w))
            return false;
    }
    return true;
}

}; // namespace Utilities
}; // namespace Jet
