from typing import Union

import numpy as np

from .bindings import (
    TaskBasedCpuContractorC64,
    TaskBasedCpuContractorC128,
    TensorC64,
    TensorC128,
    TensorNetworkC64,
    TensorNetworkC128,
    TensorNetworkFileC64,
    TensorNetworkFileC128,
    TensorNetworkSerializerC64,
    TensorNetworkSerializerC128,
)

__all__ = [
    "TaskBasedCpuContractor",
    "Tensor",
    "TensorNetwork",
    "TensorNetworkFile",
    "TensorNetworkSerializer",
]


def TaskBasedCpuContractor(
    *args, **kwargs
) -> Union[TaskBasedCpuContractorC64, TaskBasedCpuContractorC128]:
    """Constructs a task-based CPU contractor with the specified data type. If a
    `dtype` keyword argument is not provided, a TaskBasedCpuContractorC128
    instance will be returned.
    """
    dtype = kwargs.pop("dtype", "complex128")
    if np.dtype(dtype) == np.complex64:
        return TaskBasedCpuContractorC64(*args, **kwargs)
    elif np.dtype(dtype) == np.complex128:
        return TaskBasedCpuContractorC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def Tensor(*args, **kwargs) -> Union[TensorC64, TensorC128]:
    """Constructs a tensor with the specified data type. If a `dtype` keyword
    argument is not provided, a TensorC128 instance will be returned.
    """
    dtype = kwargs.pop("dtype", "complex128")
    if np.dtype(dtype) == np.complex64:
        return TensorC64(*args, **kwargs)
    elif np.dtype(dtype) == np.complex128:
        return TensorC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetwork(*args, **kwargs) -> Union[TensorNetworkC64, TensorNetworkC128]:
    """Constructs a tensor network with the specified data type. If a `dtype`
    keyword argument is not provided, a TensorNetworkC128 instance will be
    returned.
    """
    dtype = kwargs.pop("dtype", "complex128")
    if np.dtype(dtype) == np.complex64:
        return TensorNetworkC64(*args, **kwargs)
    elif np.dtype(dtype) == np.complex128:
        return TensorNetworkC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkFile(*args, **kwargs) -> Union[TensorNetworkFileC64, TensorNetworkFileC128]:
    """Constructs a tensor network file with the specified data type. If a
    `dtype` keyword argument is not provided, a TensorNetworkFileC128 instance
    will be returned.
    """
    dtype = kwargs.pop("dtype", "complex128")
    if np.dtype(dtype) == np.complex64:
        return TensorNetworkFileC64(*args, **kwargs)
    elif np.dtype(dtype) == np.complex128:
        return TensorNetworkFileC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkSerializer(
    *args, **kwargs
) -> Union[TensorNetworkSerializerC64, TensorNetworkSerializerC128]:
    """Constructs a tensor network serializer with the specified data type. If a
    `dtype` keyword argument is not provided, a TensorNetworkSerializerC128
    instance will be returned.
    """
    dtype = kwargs.pop("dtype", "complex128")
    if np.dtype(dtype) == np.complex64:
        return TensorNetworkSerializerC64(*args, **kwargs)
    elif np.dtype(dtype) == np.complex128:
        return TensorNetworkSerializerC128(*args, **kwargs)
    else:
        raise TypeError(f"Data type '{dtype}' is not supported.")
