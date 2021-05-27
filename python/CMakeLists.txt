Include(FetchContent)

FetchContent_Declare(
    Pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
)

FetchContent_MakeAvailable(Pybind11)

# The following command also creates a CMake target called "bindings".
pybind11_add_module(bindings src/Python.cpp)

target_link_libraries(bindings PRIVATE Jet)
