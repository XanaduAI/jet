## Release 0.2.0 (development release)

### New features since last release

* Full and sliced contractions can now be run with `TaskBasedContractor`on the GPU using the `CudaTensor` class. (#29)](https://github.com/XanaduAI/jet/pull/29)

* The `TaskBasedCpuContractor` class has been renamed to `TaskBasedContractor`. (#29)](https://github.com/XanaduAI/jet/pull/29)

* An `XIRProgram` which declares the gates supported by Jet is now bundled with the `jet` Python package. [(#34)](https://github.com/XanaduAI/jet/pull/34)

* The `jet` Python package now includes an interpreter for XIR programs. [(#24)](https://github.com/XanaduAI/jet/pull/24)

* Gates may now be instantiated by name using the `GateFactory` Python class. [(#23)](https://github.com/XanaduAI/jet/pull/23)

* Quantum circuit and state models have been added to the `jet` Python package. [(#21)](https://github.com/XanaduAI/jet/pull/21)

* Quantum gate models have been added to the `jet` Python package. [(#16)](https://github.com/XanaduAI/jet/pull/16)

* Python bindings are now available for the `TaskBasedCpuContractor` class. [(#19)](https://github.com/XanaduAI/jet/pull/19)

* Python bindings now include a factory method which accepts a `dtype` parameter. [(#18)](https://github.com/XanaduAI/jet/pull/18)

* Running `make build` from the `python` directory now creates a Python distribution package. [(#13)](https://github.com/XanaduAI/jet/pull/13)

* A new intermediate representation (IR) is added, including a parser, IR representation program, and a Strawberry Fields interface. [(#11)](https://github.com/XanaduAI/jet/pull/11)

* Python bindings are now available for the `TensorNetworkSerializer` class. [(#5)](https://github.com/XanaduAI/jet/pull/5)

* Python bindings are now available for the `TensorNetwork` and `PathInfo` classes [(#7)](https://github.com/XanaduAI/jet/pull/7)

* Python bindings are now available for the `Tensor` class. [(#2)](https://github.com/XanaduAI/jet/pull/2)

* Running CMake with `-DBUILD_PYTHON=ON` now generates Python bindings within a `jet` package. [(#1)](https://github.com/XanaduAI/jet/pull/1)

### Improvements

* Tensor transposes are now significantly faster when all the dimensions are powers of two. [(#12)](https://github.com/XanaduAI/jet/pull/12)

* Use camel case for type aliases. [(#17)](https://github.com/XanaduAI/jet/pull/17)

* Exceptions are now favoured in place of `std::terminate` with `Exception` being the new base type for all exceptions thrown by Jet. [(#3)](https://github.com/XanaduAI/jet/pull/3)

* `TaskBasedCpuContractor` now stores `Tensor` results. [(#8)](https://github.com/XanaduAI/jet/pull/8)

* `Tensor` class now checks data type at compile-time. [(#4)](https://github.com/XanaduAI/jet/pull/4)

### Breaking Changes

* The Jet interpreter for XIR scripts is now case-sensitive with respect to gate names. [(#36)](https://github.com/XanaduAI/jet/pull/36)

* Python bindings for `complex<float>` and `complex<double>` specializations are now suffixed with `C64` and `C128`, respectively. [(#15)](https://github.com/XanaduAI/jet/pull/15)

* Indices are now specified in row-major order. [(#10)](https://github.com/XanaduAI/jet/pull/10)

### Bug Fixes

* An issue with the `CudaTensor` indices was fixed when converting between the CPU `Tensor` class. (#29)](https://github.com/XanaduAI/jet/pull/24)

* The Jet versions returned by `Jet::Version()` (C++) and `jet.Version()` (Python) are now correct. [(#26)](https://github.com/XanaduAI/jet/pull/26)

* The documentation build no longer emits any Doxygen warnings. [(#25)](https://github.com/XanaduAI/jet/pull/25)

* Running `make build` in the `python` directory now correctly uses the virtual environment. [(#31)](https://github.com/XanaduAI/jet/pull/31)

* The output of `TensorNetwork::Contract()` and `TaskBasedCpuContractor::Contract()` now agree with external packages. [(#12)](https://github.com/XanaduAI/jet/pull/12)

* `TaskBasedCpuContractor::AddReductionTask()` now handles the reduction of non-scalar tensors. [(#19)](https://github.com/XanaduAI/jet/pull/19)

* The output of `TensorNetwork::Contract()` and `TaskBasedCpuContractor::Contract()` now agree with one another. [(#6)](https://github.com/XanaduAI/jet/pull/6)

* `PathInfo` now correctly names intermediary tensors in a sliced tensor network [(#22)](https://github.com/XanaduAI/jet/pull/22).

### Documentation

* The Sphinx documentation now includes API documentation for the `jet` Python package. [(#40)](https://github.com/XanaduAI/jet/pull/40)

* The "Using Jet" section of the Sphinx documentation website now compiles with the latest Jet headers. [(#26)](https://github.com/XanaduAI/jet/pull/26)

* The license comment headers at the top of the IR source files have been removed. [(#14)](https://github.com/XanaduAI/jet/pull/14)

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Theodor Isacsson](https://github.com/thisac), [Josh Izaac](https://github.com/josh146), [Lee J. O'Riordan](https://github.com/mlxd), [Antal Száva](https://github.com/antalszava), [Trevor Vincent](https://github.com/trevor-vincent).

## Release 0.1.0 (current release)

### New features since last release

* This is the initial public release.

### Contributors

This release contains contributions from (in alphabetical order):

[Mikhail Andrenkov](https://github.com/Mandrenkov), [Jack Brown](https://github.com/brownj85), [Lee J. O'Riordan](https://github.com/mlxd), [Trevor Vincent](https://github.com/trevor-vincent).
