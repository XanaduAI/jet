# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *

# By default, Python uses two 64-bit floating-point numbers to represent a
# complex number. Altogether, this requires 128 bits of storage.
Tensor = TensorC128
TensorNetwork = TensorNetworkC128
TensorNetworkFile = TensorNetworkFileC128
TensorNetworkSerializer = TensorNetworkSerializerC128

# Grab the current Jet version from the C++ headers.
__version__ = version()
