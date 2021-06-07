from abc import ABC, abstractmethod
from typing import Optional, List, Sequence
import numpy as np

from .factory import Tensor, TensorType

__all__ = [
    "State",
    "Qudit",
    "QuditRegister",
    "Qubit",
    "QubitRegister",
]


class State(ABC):
    def __init__(self, name: str, num_wires: int, **kwargs):
        """Constructs a quantum state.

        Args:
            name: name of the state.
            num_wires: number of wires the state connects to.

        Kwargs:
            tensor_id (int): ID of the state tensor.
        """
        self.name = name
        self.tensor_id = kwargs.get("tensor_id", None)

        self._indices = None
        self._num_wires = num_wires

    @property
    def indices(self) -> Optional[List[str]]:
        """Returns the indices of this state for connecting tensors."""
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this state for connecting tensors."""
        # Skip the sequence property checks if `indices` is None.
        if indices is None:
            pass

        # Check that `indices` is a sequence of unique strings.
        elif (
            not isinstance(indices, Sequence)
            or not all(isinstance(idx, str) for idx in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Indices must be a sequence of unique strings.")

        # Check that `indices` has the correct length (or is None).
        elif len(indices) != self.num_wires:
            raise ValueError(
                f"States must have one index per wire. "
                f"Received {len(indices)} indices for {self.num_wires} wires."
            )

        self._indices = indices

    @property
    def num_wires(self) -> int:
        """Returns the number of wires spanned by this state."""
        return self._num_wires

    def __eq__(self, other) -> bool:
        """Reports whether this state is equivalent to the given state."""
        return np.all(self._data() == other._data())

    def __ne__(self, other) -> bool:
        """Reports whether this state is not equivalent to the given state."""
        return not (self == other)

    @abstractmethod
    def _data(self) -> np.ndarray:
        """Returns the vector representation of this state."""

    def tensor(self, dtype: type = np.complex128, adjoint: bool = False) -> TensorType:
        """Returns the tensor representation of this state."""
        if adjoint:
            data = np.conj(self._data())
        else:
            data = self._data()

        indices = self.indices
        if indices is None:
            indices = list(map(str, range(self.num_wires)))

        dimension = int(round(len(data) ** (1 / len(indices))))
        shape = [dimension] * len(indices)

        return Tensor(indices=indices, shape=shape, data=data, dtype=dtype)


class Qudit(State):
    def __init__(self, dim: int, data: Optional[np.ndarray] = None):
        """Constructs a qudit state.

        Args:
            dim: dimension of the qudit.
            data: optional state vector.
        """
        name = "Qubit" if dim == 2 else f"Qudit(d={dim})"
        super().__init__(name=name, num_wires=1)

        if data is None:
            self._state_vector = np.arange(dim) == 0
        else:
            self._state_vector = data.flatten()

    def _data(self) -> np.ndarray:
        return self._state_vector


class QuditRegister(State):
    def __init__(self, dim: int, size: int, data: Optional[np.ndarray] = None):
        """Constructs a qudit register state.

        Args:
            dim: dimension of the qudits.
            size: number of qudits.
            data: optional state vector.
        """
        name = f"Qubit[{size}]" if dim == 2 else f"Qudit(d={dim})[{size}]"
        super().__init__(name=name, num_wires=size)

        if data is None:
            self._state_vector = np.arange(dim ** size) == 0
        else:
            self._state_vector = data.flatten()

    def _data(self) -> np.ndarray:
        return self._state_vector


def Qubit(data: Optional[np.ndarray] = None) -> Qudit:
    """Constructs a qubit state using an optional state vector."""
    return Qudit(dim=2, data=data)


def QubitRegister(size: int, data: Optional[np.ndarray] = None) -> QuditRegister:
    """Constructs a qubit register state with the given size and optional state vector."""
    return QuditRegister(dim=2, size=size, data=data)
