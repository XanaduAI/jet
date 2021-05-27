import numpy as np

# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *


def Tensor(*args, **kwargs):
    """Contructs a tensor with the specified NumPy data type."""
    dt = kwargs.pop("dtype", "complex128")
    if np.dtype(dt) == np.complex64:
        return TensorC64(*args, **kwargs)
    elif np.dtype(dt) == np.complex128:
        return TensorC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dt}' is not supported.")


def TensorNetwork(*args, **kwargs):
    """Contructs a tensor network with the specified NumPy data type."""
    dt = kwargs.pop("dtype", "complex128")
    if np.dtype(dt) == np.complex64:
        return TensorNetworkC64(*args, **kwargs)
    elif np.dtype(dt) == np.complex128:
        return TensorNetworkC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dt}' is not supported.")


def TensorNetworkFile(*args, **kwargs):
    """Contructs a tensor network file with the specified NumPy data type."""
    dt = kwargs.pop("dtype", "complex128")
    if np.dtype(dt) == np.complex64:
        return TensorNetworkFileC64(*args, **kwargs)
    elif np.dtype(dt) == np.complex128:
        return TensorNetworkFileC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dt}' is not supported.")


def TensorNetworkSerializer(*args, **kwargs):
    """Contructs a tensor network serializer with the specified NumPy data type."""
    dt = kwargs.pop("dtype", "complex128")
    if np.dtype(dt) == np.complex64:
        return TensorNetworkSerializerC64(*args, **kwargs)
    elif np.dtype(dt) == np.complex128:
        return TensorNetworkSerializerC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dt}' is not supported.")


# Grab the current Jet version from the C++ headers.
__version__ = version()
