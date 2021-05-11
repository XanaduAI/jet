#pragma once

#include <string>

namespace Jet {

/// Major version number of Jet.
constexpr size_t MAJOR_VERSION = 0;

/// Minor version number of Jet.
constexpr size_t MINOR_VERSION = 1;

/// Patch version number of Jet.
constexpr size_t PATCH_VERSION = 0;

/**
 * @brief Returns the current Jet version.
 *
 * @return String representation of the current Jet version.
 */
std::string Version()
{
    const auto major = std::to_string(MAJOR_VERSION);
    const auto minor = std::to_string(MINOR_VERSION);
    const auto patch = std::to_string(PATCH_VERSION);
    return major + "." + minor + "." + patch;
}
} // namespace Jet