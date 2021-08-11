"""Module containing __init__() wrappers for the classes in jet.bindings."""
from typing import Union

import numpy as np

from .bindings import (
    TaskBasedContractorC64,
    TaskBasedContractorC128,
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
    "TaskBasedContractorType",
    "TensorType",
    "TensorNetworkType",
    "TensorNetworkFileType",
    "TensorNetworkSerializerType",
    "TaskBasedContractor",
    "Tensor",
    "TensorNetwork",
    "TensorNetworkFile",
    "TensorNetworkSerializer",
]

# Type aliases to avoid enumerating class specializations.
TaskBasedContractorType = Union[TaskBasedContractorC64, TaskBasedContractorC128]
TensorType = Union[TensorC64, TensorC128]
TensorNetworkType = Union[TensorNetworkC64, TensorNetworkC128]
TensorNetworkFileType = Union[TensorNetworkFileC64, TensorNetworkFileC128]
TensorNetworkSerializerType = Union[TensorNetworkSerializerC64, TensorNetworkSerializerC128]


def TaskBasedContractor(*args, **kwargs) -> TaskBasedContractorType:
    """Constructs a task-based contractor (TBC) with the specified data
    type. If a ``dtype`` keyword argument is not provided, a
    ``TaskBasedContractorC128`` instance will be returned.

    Args:
        *args: Positional arguments to pass to the TBC constructor.
        **kwargs: Keyword arguments to pass to the TBC constructor.

    Returns:
        Task-based contractor instance.
    """
    dtype = kwargs.pop("dtype", np.complex128)

    if np.dtype(dtype) == np.complex64:
        return TaskBasedContractorC64(*args, **kwargs)

    if np.dtype(dtype) == np.complex128:
        return TaskBasedContractorC128(*args, **kwargs)

    raise TypeError(f"Data type '{dtype}' is not supported.")


def Tensor(*args, **kwargs) -> TensorType:
    """Constructs a tensor with the specified data type. If a ``dtype`` keyword
    argument is not provided, a ``TensorC128`` instance will be returned.

    Args:
        *args: Positional arguments to pass to the tensor constructor.
        **kwargs: Keyword arguments to pass to the tensor constructor.

    Returns:
        Tensor instance.
    """
    dtype = kwargs.pop("dtype", np.complex128)

    if np.dtype(dtype) == np.complex64:
        return TensorC64(*args, **kwargs)

    if np.dtype(dtype) == np.complex128:
        return TensorC128(*args, **kwargs)

    raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetwork(*args, **kwargs) -> TensorNetworkType:
    """Constructs a tensor network with the specified data type. If a ``dtype``
    keyword argument is not provided, a ``TensorNetworkC128`` instance will be
    returned.

    Args:
        *args: Positional arguments to pass to the tensor network constructor.
        **kwargs: Keyword arguments to pass to the tensor network constructor.

    Returns:
        Tensor network instance.
    """
    dtype = kwargs.pop("dtype", np.complex128)

    if np.dtype(dtype) == np.complex64:
        return TensorNetworkC64(*args, **kwargs)

    if np.dtype(dtype) == np.complex128:
        return TensorNetworkC128(*args, **kwargs)

    raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkFile(*args, **kwargs) -> TensorNetworkFileType:
    """Constructs a tensor network file with the specified data type. If a
    ``dtype`` keyword argument is not provided, a ``TensorNetworkFileC128``
    instance will be returned.

    Args:
        *args: Positional arguments to pass to the tensor network file constructor.
        **kwargs: Keyword arguments to pass to the tensor network file constructor.

    Returns:
        Tensor network file instance.
    """
    dtype = kwargs.pop("dtype", np.complex128)

    if np.dtype(dtype) == np.complex64:
        return TensorNetworkFileC64(*args, **kwargs)

    if np.dtype(dtype) == np.complex128:
        return TensorNetworkFileC128(*args, **kwargs)

    raise TypeError(f"Data type '{dtype}' is not supported.")


def TensorNetworkSerializer(*args, **kwargs) -> TensorNetworkSerializerType:
    """Constructs a tensor network serializer with the specified data type. If a
    ``dtype`` keyword argument is not provided, a ``TensorNetworkSerializerC128``
    instance will be returned.

    Args:
        *args: Positional arguments to pass to the tensor network serializer constructor.
        **kwargs: Keyword arguments to pass to the tensor network serializer constructor.

    Returns:
        Tensor network serializer instance.
    """
    dtype = kwargs.pop("dtype", np.complex128)

    if np.dtype(dtype) == np.complex64:
        return TensorNetworkSerializerC64(*args, **kwargs)

    if np.dtype(dtype) == np.complex128:
        return TensorNetworkSerializerC128(*args, **kwargs)

    raise TypeError(f"Data type '{dtype}' is not supported.")
