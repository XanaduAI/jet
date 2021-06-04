from abc import ABC, abstractmethod
from typing import Optional, List, Sequence
import numpy as np

from .factory import Tensor, TensorType

__all__ = [
    "State",
    "Qubit",
    "Qudit",
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
        elif len(indices) != 1:
            raise ValueError(
                f"States must have one index per wire. "
                f"Received {len(indices)} indices for {self.num_wires} wires."
            )

        self._indices = indices

    @property
    def num_wires(self) -> int:
        """Returns the number of wires spanned by this state."""
        return self._num_wires

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


class Qubit(State):
    def __init__(self, data: Optional[np.ndarray] = None):
        """Constructs a qubit state.

        Args:
           data: optional state vector.
        """
        self._state_vector = np.array([1, 0]) if data is None else data
        super().__init__(name="Qubit", num_wires=1)

    def _data(self) -> np.ndarray:
        return self._state_vector


class Qudit(State):
    def __init__(self, data: Optional[np.ndarray] = None, dim: int = 2):
        """Constructs a qudit gate.

        Args:
            data: optional state vector.
            dim: dimension of the qudit.
        """
        self._state_vector = np.array([1] + [0] * (dim - 1)) if data is None else data
        super().__init__(name=f"Qu-{dim}-it", num_wires=1)

    def _data(self) -> np.ndarray:
        return self._state_vector
