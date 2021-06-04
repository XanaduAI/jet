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
            tensor_id (int): ID for the state tensor.
        """
        self.name = name
        self.tensor_id = kwargs.get("tensor_id", None)

        self._indices = None
        self._num_wires = num_wires

    @property
    def num_wires(self) -> int:
        """Returns the number of wires spanned by this state."""
        return self._num_wires

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

    @abstractmethod
    def _data(self) -> np.ndarray:
        """Returns the vector representation of this state."""
        raise NotImplementedError("No data available for generic state.")


class Qubit(State):
    def __init__(self, state: Optional[np.ndarray] = None):
        """Constructs a qubit state."""
        self._state = state or np.array([1, 0])
        super().__init__(name="Qubit", num_wires=1)

    def _data(self) -> np.ndarray:
        return self._state


class Qudit(State):
    def __init__(self, dim: int, state: Optional[np.ndarray] = None):
        """Constructs a qudit gate.

        Args:
            dim: dimension of the qudit.
            state: optional state vector.
        """
        self._state = state or np.array([1] + [0] * (dim - 1))
        super().__init__(name=f"Qudit(d={dim})", num_wires=1)

    def _data(self) -> np.ndarray:
        return self._state
