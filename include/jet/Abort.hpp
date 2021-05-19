#pragma once

#include <exception>
#include <iostream>
#include <sstream>

/**
 * @brief Macro that throws `%JetException` with given message.
 *
 * @param message string literal describing error
 */
#define JET_ABORT(message) Jet::Abort(message, __FILE__, __LINE__, __func__)
/**
 * @brief Macro that throws `%JetException` if expression evaluates to true.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define JET_ABORT_IF(expression, message)                                      \
    if ((expression)) {                                                        \
        JET_ABORT(message);                                                    \
    }
/**
 * @brief Macro that throws `%JetException` with error message if expression
 * evaluates to false.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define JET_ABORT_IF_NOT(expression, message)                                  \
    if (!(expression)) {                                                       \
        JET_ABORT(message);                                                    \
    }

/**
 * @brief Macro that throws `%JetException` with the given expression and source
 * location if expression evaluates to false.
 *
 * @param expression an expression
 */
#define JET_ASSERT(expression)                                                 \
    JET_ABORT_IF_NOT(expression, "Assertion failed: " #expression)

namespace Jet {

/**
 * @brief `%JetException` is the general exception thrown by Jet for runtime
 * errors.
 *
 */
class JetException : public std::exception {
  public:
    /**
     * @brief Constructs a new `%JetException` exception.
     *
     * @param err_msg Error message explaining the exception condition.
     */
    explicit JetException(const std::string &err_msg) noexcept
        : err_msg(err_msg)
    {
    }

    /**
     * @brief Destroy the `%JetException` object.
     */
    virtual ~JetException() = default;

    /**
     * @brief Returns string containing exception message. Overloaded
     * std::exception method.
     *
     * @return const char* Exception message
     */
    const char *what() const noexcept { return err_msg.c_str(); }

  private:
    std::string err_msg;
};

/**
 * @brief Throws a `%JetException` with the given error message.
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
            << "][Method:" << function_name << "]: Error in Jet: " << message;
    throw JetException(err_msg.str());
}

}; // namespace Jet