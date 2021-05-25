# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *

# Python floating-point numbers, by default, are 64 bits wide.
Tensor = Tensor64
TensorNetwork = TensorNetwork64
TensorNetworkFile = TensorNetworkFile64
TensorNetworkSerializer = TensorNetworkSerializer64

# Grab the current Jet version from the C++ headers.
__version__ = version()
