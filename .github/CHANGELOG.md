## Release 0.2.0 (development release)

### New features since last release

* Exceptions now favoured in-place of `std::terminate`, with `JetException` as base type. Replaces `std::invalid_argument` base type of `TensorNetworkIO`. [(#3)](https://github.com/XanaduAI/jet/pull/3)

* Python bindings are now available for the `Tensor` class. [(#2)](https://github.com/XanaduAI/jet/pull/2)

* Running CMake with `-DBUILD_PYTHON=ON` now generates Python bindings within a `jet` package. [(#1)](https://github.com/XanaduAI/jet/pull/1)

### Improvements

* `TaskBasedCpuContractor` now stores `Tensor` results. [(#8)](https://github.com/XanaduAI/jet/pull/8)

* `Tensor` class now checks data type at compile-time. [(#4)](https://github.com/XanaduAI/jet/pull/4)

### Breaking Changes

* Indices are now specified in row-major order. [(#9)](https://github.com/XanaduAI/jet/pull/9)

### Bug Fixes

* The output of `TensorNetwork::Contract()` and `TaskBasedCpuContractor::Contract()` now agree with one another. [(#6)](https://github.com/XanaduAI/jet/pull/6)

### Documentation

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Lee J. O'Riordan](https://github.com/mlxd), [Trevor Vincent](https://github.com/trevor-vincent).

## Release 0.1.0 (current release)

### New features since last release

* This is the initial public release.

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Lee J. O'Riordan](https://github.com/mlxd), [Trevor Vincent](https://github.com/trevor-vincent).
