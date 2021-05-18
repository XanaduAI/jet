#pragma once

#include <exception>
#include <iostream>
#include <sstream>

/**
 * @brief Macro that prints error message and source location to stderr
 * and calls `std::terminate()`
 *
 * @param message string literal describing error
 */
#define JET_ABORT(message) Jet::Abort(message, __FILE__, __LINE__, __func__)
/**
 * @brief Macro that prints error message and source location to stderr
 * and calls `std::terminate()` if expression evaluates to true
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define JET_ABORT_IF(expression, message)                                      \
    if ((expression)) {                                                        \
        JET_ABORT(message);                                                    \
    }
/**
 * @brief Macro that prints error message and source location to stderr
 * and calls `std::terminate()` if expression evaluates to false
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define JET_ABORT_IF_NOT(expression, message)                                  \
    if (!(expression)) {                                                       \
        JET_ABORT(message);                                                    \
    }

/**
 * @brief Macro that prints expression and source location to stderr
 * and calls `std::terminate()` if expression evaluates to false
 *
 * @param expression an expression
 */
#define JET_ASSERT(expression)                                                 \
    JET_ABORT_IF_NOT(expression, "Assertion failed: " #expression)

namespace Jet {

class JetException : public std::exception {
  public:
    JetException(const std::string &err_msg) noexcept : err_msg(err_msg) {}
    virtual ~JetException() = default;
    const char *what() const noexcept { return err_msg.c_str(); }

  private:
    std::string err_msg;
};

/**
 * @brief Prints an error message to stderr and calls `std::terminate()`.
 *
 * This function should not be called directly - use one of the `JET_ASSERT()`
 * or `JET_ABORT()` macros, which provide the source location at compile time.
 *
 * @param message string literal describing the error
 * @param file_name source file where error occured
 * @param line line of source file
 * @param function_name function in which error occured
 */
inline void Abort(const char *message, const char *file_name, int line,
                  const char *function_name)
{
    std::stringstream err_msg;
    err_msg << "[" << file_name << "][Line:" << line
            << "][Method:" << function_name
            << "]: Fatal error in Jet: " << message;
    throw JetException(err_msg.str());
}

}; // namespace Jet
