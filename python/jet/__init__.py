# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *


def Tensor(*args, **kwargs):
    """Contructs a tensor with the specified NumPy data type."""
    dtype = kwargs.pop("dtype", "c16")
    if dtype == "c8":
        return TensorC64(*args, **kwargs)
    elif dtype == "c16":
        return TensorC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetwork(*args, **kwargs):
    """Contructs a tensor network with the specified NumPy data type."""
    dtype = kwargs.pop("dtype", "c16")
    if dtype == "c8":
        return TensorNetworkC64(*args, **kwargs)
    elif dtype == "c16":
        return TensorNetworkC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkFile(*args, **kwargs):
    """Contructs a tensor network file with the specified NumPy data type."""
    dtype = kwargs.pop("dtype", "c16")
    if dtype == "c8":
        return TensorNetworkFileC64(*args, **kwargs)
    elif dtype == "c16":
        return TensorNetworkFileC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkSerializer(*args, **kwargs):
    """Contructs a tensor network serializer with the specified NumPy data type."""
    dtype = kwargs.pop("dtype", "c16")
    if dtype == "c8":
        return TensorNetworkSerializerC64(*args, **kwargs)
    elif dtype == "c16":
        return TensorNetworkSerializerC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


# Grab the current Jet version from the C++ headers.
__version__ = version()
