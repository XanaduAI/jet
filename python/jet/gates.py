import cmath
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np
from thewalrus.fock_gradients import (
    beamsplitter,
    displacement,
    squeezing,
    two_mode_squeezing,
)

from .factory import Tensor, TensorType

__all__ = [
    "Gate",
    # CV Fock gates
    "Displacement",
    "Squeezing",
    "TwoModeSqueezing",
    "Beamsplitter",
    # Qubit gates
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "SX",
    "CNOT",
    "CY",
    "CZ",
    "SWAP",
    "ISWAP",
    "CSWAP",
    "Toffoli",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "CPhaseShift",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U1",
    "U2",
    "U3",
]


INV_SQRT2 = 1 / math.sqrt(2)


class Gate(ABC):
    def __init__(
        self,
        name: str,
        num_wires: int,
        params: Optional[List] = None,
        tensor_id: Optional[int] = None,
    ):
        """Constructs a quantum gate.

        Args:
            name: name of the gate.
            num_wires: number of wires the gate is applied to.
            params: gate parameters.
            tensor_id: ID of the gate tensor.
        """
        self.name = name
        self.tensor_id = tensor_id

        self._indices = None
        self._num_wires = num_wires
        self._params = params or []

    @property
    def indices(self) -> Optional[List[str]]:
        """Returns the indices of this gate."""
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this gate."""
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
        elif len(indices) != 2 * self._num_wires:
            raise ValueError(
                f"Gates must have two indices per wire; received {len(indices)}"
                f"indices for {self._num_wires} wires."
            )

        self._indices = indices

    @property
    def num_wires(self) -> int:
        """Returns the number of wires this gate affects."""
        return self._num_wires

    @property
    def params(self) -> Optional[List]:
        """Returns the parameters of this gate."""
        return self._params

    @abstractmethod
    def _data(self) -> np.ndarray:
        """Returns the matrix representation of this gate."""

    def tensor(self, dtype: type = np.complex128, adjoint: bool = False) -> TensorType:
        """Returns the tensor representation of this gate."""
        if adjoint:
            data = np.conj(self._data()).T.flatten()
        else:
            data = self._data().flatten()

        indices = self.indices
        if indices is None:
            indices = list(map(str, range(2 * self._num_wires)))

        dimension = int(round(len(data) ** (1 / len(indices))))
        shape = [dimension] * len(indices)

        return Tensor(indices=indices, shape=shape, data=data, dtype=dtype)


####################################################################################################
# Continuous variable Fock gates
####################################################################################################


class Displacement(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a displacement gate.

        Args:
            r (float): displacement magnitude.
            phi (float): displacement angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Displacement", num_wires=1, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        return displacement(*self.params)


class Squeezing(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a squeezing gate.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Squeezing", num_wires=1, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        return squeezing(*self.params)


class TwoModeSqueezing(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a two-mode squeezing gate.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="TwoModeSqueezing", num_wires=2, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        return two_mode_squeezing(*self.params)


class Beamsplitter(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a beamsplitter gate.

        Args:
            theta (float): transmissivity angle of the beamsplitter. The transmissivity is
                           :math:`t=\\cos(\\theta)`.
            phi (float): reflection phase of the beamsplitter.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Beamsplitter", num_wires=2, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        return beamsplitter(*self.params)


####################################################################################################
# Qubit gates
####################################################################################################


class Hadamard(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a Hadamard gate."""
        super().__init__(name="Hadamard", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        """Hadamard matrix"""
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat)


class PauliX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliX gate."""
        super().__init__(name="PauliX", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, 1], [1, 0]]
        return np.array(mat)


class PauliY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliY gate."""
        super().__init__(name="PauliY", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat)


class PauliZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliZ gate."""
        super().__init__(name="PauliZ", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, -1]]
        return np.array(mat)


class S(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit phase gate."""
        super().__init__(name="S", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, 1j]]
        return np.array(mat)


class T(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit T gate."""
        super().__init__(name="T", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, cmath.exp(0.25j * np.pi)]]
        return np.array(mat)


class SX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Square-Root X gate."""
        super().__init__(name="SX", num_wires=1, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat)


class PhaseShift(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit local phase shift gate."""
        super().__init__(name="PhaseShift", num_wires=1, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]
        return np.array(mat)


class CPhaseShift(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled phase shift gate."""
        super().__init__(name="CPhaseShift", num_wires=2, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, cmath.exp(1j * phi)]]
        return np.array(mat)


class CNOT(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a CNOT gate."""
        super().__init__(name="CNOT", num_wires=2, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat)


class CY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-Y gate."""
        super().__init__(name="CY", num_wires=2, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat)


class CZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-Z gate."""
        super().__init__(name="CZ", num_wires=2, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat)


class SWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a SWAP gate."""
        super().__init__(name="SWAP", num_wires=2, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


class ISWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an ISWAP gate."""
        super().__init__(name="ISWAP", num_wires=2, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


class CSWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a CSWAP gate."""
        super().__init__(name="CSWAP", num_wires=3, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
        return np.array(mat)


class Toffoli(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a Toffoli gate."""
        super().__init__(name="Toffoli", num_wires=3, params=params, **kwargs)

        if len(params) != 0:
            raise ValueError(f"Received {len(params)} (!= 0) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
        return np.array(mat)


class RX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit X rotation gate."""
        super().__init__(name="RX", num_wires=1, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[c, js], [js, c]]
        return np.array(mat)


class RY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Y rotation gate."""
        super().__init__(name="RY", num_wires=1, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[c, -s], [s, c]]
        return np.array(mat)


class RZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Z rotation gate."""
        super().__init__(name="RZ", num_wires=1, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        p = cmath.exp(-0.5j * theta)

        mat = [[p, 0], [0, np.conj(p)]]
        return np.array(mat)


class Rot(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an arbitrary single-qubit rotation gate."""
        super().__init__(name="Rot", num_wires=1, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, theta, omega = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [cmath.exp(-0.5j * (phi + omega)) * c, -cmath.exp(0.5j * (phi - omega)) * s],
            [cmath.exp(-0.5j * (phi - omega)) * s, cmath.exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat)


class CRX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RX gate."""
        super().__init__(name="CRX", num_wires=2, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return np.array(mat)


class CRY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RY gate."""
        super().__init__(name="CRY", num_wires=2, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return np.array(mat)


class CRZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RZ gate."""
        super().__init__(name="CRZ", num_wires=2, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-0.5j * theta), 0],
            [0, 0, 0, cmath.exp(0.5j * theta)],
        ]
        return np.array(mat)


class CRot(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-rotation gate."""
        super().__init__(name="CRot", num_wires=2, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, theta, omega = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-0.5j * (phi + omega)) * c, -cmath.exp(0.5j * (phi - omega)) * s],
            [0, 0, cmath.exp(-0.5j * (phi - omega)) * s, cmath.exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat)


class U1(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a U1 gate."""
        super().__init__(name="U1", num_wires=1, params=params, **kwargs)

        if len(params) != 1:
            raise ValueError(f"Received {len(params)} (!= 1) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]
        return np.array(mat)


class U2(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a U2 gate."""
        super().__init__(name="U2", num_wires=1, params=params, **kwargs)

        if len(params) != 2:
            raise ValueError(f"Received {len(params)} (!= 2) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, lam = self.params
        mat = [
            [INV_SQRT2, -INV_SQRT2 * cmath.exp(1j * lam)],
            [INV_SQRT2 * cmath.exp(1j * phi), INV_SQRT2 * cmath.exp(1j * (phi + lam))],
        ]
        return np.array(mat)


class U3(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an arbitrary single-qubit unitary gate."""
        super().__init__(name="U3", num_wires=1, params=params, **kwargs)

        if len(params) != 3:
            raise ValueError(f"Received {len(params)} (!= 3) parameters for a {self.name} gate.")

    @lru_cache
    def _data(self) -> np.ndarray:
        theta, phi, lam = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [c, -s * cmath.exp(1j * lam)],
            [s * cmath.exp(1j * phi), c * cmath.exp(1j * (phi + lam))],
        ]
        return np.array(mat)
