## Release 0.2.0 (development release)

### New features since last release

* Python bindings are now available for the `TensorNetworkSerializer` class. [(#5)](https://github.com/XanaduAI/jet/pull/5)

* Python bindings are now available for the `TensorNetwork` and `PathInfo` classes [(#7)](https://github.com/XanaduAI/jet/pull/7)

* Python bindings are now available for the `Tensor` class. [(#2)](https://github.com/XanaduAI/jet/pull/2)

* Running CMake with `-DBUILD_PYTHON=ON` now generates Python bindings within a `jet` package. [(#1)](https://github.com/XanaduAI/jet/pull/1)

* A new intermediate representation (IR) is added, including a parser, IR representation program, and a Strawberry Fields interface. [(#11)](https://github.com/XanaduAI/jet/pull/11)

### Improvements

* Tensor network contractions are now significantly faster than initial release. [(#12)](https://github.com/XanaduAI/jet/pull/12)

* Exceptions are now favoured in place of `std::terminate` with `Exception` being the new base type for all exceptions thrown by Jet. [(#3)](https://github.com/XanaduAI/jet/pull/3)

* `TaskBasedCpuContractor` now stores `Tensor` results. [(#8)](https://github.com/XanaduAI/jet/pull/8)

* `Tensor` class now checks data type at compile-time. [(#4)](https://github.com/XanaduAI/jet/pull/4)

### Breaking Changes

* Indices are now specified in row-major order. [(#10)](https://github.com/XanaduAI/jet/pull/10)

### Bug Fixes

* The output of `TensorNetwork::Contract()` and `TaskBasedCpuContractor::Contract()` now agree with external packages. [(#12)](https://github.com/XanaduAI/jet/pull/12)

* The output of `TensorNetwork::Contract()` and `TaskBasedCpuContractor::Contract()` now agree with one another. [(#6)](https://github.com/XanaduAI/jet/pull/6)

### Documentation

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Theodor Isacsson](https://github.com/thisac), [Lee J. O'Riordan](https://github.com/mlxd), [Trevor Vincent](https://github.com/trevor-vincent).

## Release 0.1.0 (current release)

### New features since last release

* This is the initial public release.

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Lee J. O'Riordan](https://github.com/mlxd), [Trevor Vincent](https://github.com/trevor-vincent).
