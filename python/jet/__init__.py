# The existence of a Python binding is proof that it is intended to be exposed.
from .bindings import *

# The default Python floating-point type occupies 64 bits.
Tensor = Tensor64
TensorNetwork = TensorNetwork64
TensorNetworkFile = TensorNetworkFile64
TensorNetworkSerializer = TensorNetworkSerializer64

# Grab the current Jet version from the C++ headers.
__version__ = version()
