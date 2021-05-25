# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *

# Python floating-point numbers, by default, are 64 bits wide.
Tensor = TensorC128
TensorNetwork = TensorNetworkC128
TensorNetworkFile = TensorNetworkFileC128
TensorNetworkSerializer = TensorNetworkSerializerC128

# Grab the current Jet version from the C++ headers.
__version__ = version()
