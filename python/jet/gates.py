import cmath
import math
from functools import lru_cache
from typing import List, Optional, Sequence, Union

import numpy as np
from thewalrus.fock_gradients import (
    beamsplitter,
    displacement,
    squeezing,
    two_mode_squeezing,
)

from .bindings import TensorC64, TensorC128
from .factory import Tensor

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


class Gate:
    def __init__(self, name: str, num_wires: int, **kwargs):
        """Constructs a quantum gate.

        Args:
            name: name of the gate.
            num_wires: number of wires the gate is applied to.

        Kwargs:
            dtype (type): type to use in matrix representations of gates.
            params (list): gate parameters.
            tensor_id (int): identification number for the gate-tensor.
        """
        self.name = name
        self.tensor_id = kwargs.get("tensor_id", None)

        self._dtype = kwargs.get("dtype", np.complex128)
        self._indices = None
        self._num_wires = num_wires
        self._params = kwargs.get("params", [])

    def tensor(self, adjoint: bool = False) -> Union[TensorC64, TensorC128]:
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

        return Tensor(indices=indices, shape=shape, data=data, dtype=self._dtype)

    def _data(self) -> np.ndarray:
        """Returns the matrix representation of this gate."""
        raise NotImplementedError("No tensor data available for generic gate.")

    def _validate(self, want_num_params: int):
        """Throws a ValueError if the given quantity differs from the number of gate parameters."""
        have_num_params = 0 if self.params is None else len(self.params)
        if have_num_params != want_num_params:
            raise ValueError(
                f"The {self.name} gate accepts exactly {want_num_params} parameters "
                f"but {have_num_params} parameters were given."
            )

    @property
    def indices(self) -> Optional[List[str]]:
        """Returns the indices of this gate for connecting tensors."""
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this gate for connecting tensors."""
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
                f"Indices must have two indices per wire. "
                f"Received {len(indices)} indices for {self._num_wires} wires."
            )

        self._indices = indices

    @property
    def params(self) -> Optional[List]:
        """Returns the parameters of this gate."""
        return self._params


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
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        return displacement(*self.params, dtype=self._dtype)


class Squeezing(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a squeezing gate.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Squeezing", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        return squeezing(*self.params, dtype=self._dtype)


class TwoModeSqueezing(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a two-mode squeezing gate.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="TwoModeSqueezing", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        return two_mode_squeezing(*self.params, dtype=self._dtype)


class Beamsplitter(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a beamsplitter gate.

        Args:
            theta (float): transmissivity angle of the beamsplitter. The transmissivity is
                           :math:`t=\\cos(\\theta)`.
            phi (float): reflection phase of the beamsplitter.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Beamsplitter", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        return beamsplitter(*self.params, dtype=self._dtype)


####################################################################################################
# Qubit gates
####################################################################################################


class Hadamard(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a Hadamard gate."""
        super().__init__(name="Hadamard", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        """Hadamard matrix"""
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat, dtype=self._dtype)


class PauliX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliX gate."""
        super().__init__(name="PauliX", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, 1], [1, 0]]
        return np.array(mat, dtype=self._dtype)


class PauliY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliY gate."""
        super().__init__(name="PauliY", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat, dtype=self._dtype)


class PauliZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a PauliZ gate."""
        super().__init__(name="PauliZ", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, -1]]
        return np.array(mat, dtype=self._dtype)


class S(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit phase gate."""
        super().__init__(name="S", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, 1j]]
        return np.array(mat, dtype=self._dtype)


class T(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit T gate."""
        super().__init__(name="T", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, cmath.exp(0.25j * np.pi)]]
        return np.array(mat, dtype=self._dtype)


class SX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Square-Root X gate."""
        super().__init__(name="SX", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat, dtype=self._dtype)


class PhaseShift(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit local phase shift gate."""
        super().__init__(name="PhaseShift", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]
        return np.array(mat, dtype=self._dtype)


class CPhaseShift(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled phase shift gate."""
        super().__init__(name="CPhaseShift", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, cmath.exp(1j * phi)]]
        return np.array(mat, dtype=self._dtype)


class CNOT(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a CNOT gate."""
        super().__init__(name="CNOT", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat, dtype=self._dtype)


class CY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-Y gate."""
        super().__init__(name="CY", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat, dtype=self._dtype)


class CZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-Z gate."""
        super().__init__(name="CZ", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat, dtype=self._dtype)


class SWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a SWAP gate."""
        super().__init__(name="SWAP", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=self._dtype)


class ISWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an ISWAP gate."""
        super().__init__(name="ISWAP", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=0)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=self._dtype)


class CSWAP(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a CSWAP gate."""
        super().__init__(name="CSWAP", num_wires=3, params=params, **kwargs)
        self._validate(want_num_params=0)

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
        return np.array(mat, dtype=self._dtype)


class Toffoli(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a Toffoli gate."""
        super().__init__(name="Toffoli", num_wires=3, params=params, **kwargs)
        self._validate(want_num_params=0)

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
        return np.array(mat, dtype=self._dtype)


class RX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit X rotation gate."""
        super().__init__(name="RX", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[c, js], [js, c]]
        return np.array(mat, dtype=self._dtype)


class RY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Y rotation gate."""
        super().__init__(name="RY", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[c, -s], [s, c]]
        return np.array(mat, self._dtype)


class RZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a single-qubit Z rotation gate."""
        super().__init__(name="RZ", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        p = cmath.exp(-0.5j * theta)

        mat = [[p, 0], [0, np.conj(p)]]
        return np.array(mat, dtype=self._dtype)


class Rot(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an arbitrary single-qubit rotation gate."""
        super().__init__(name="Rot", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, theta, omega = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [cmath.exp(-0.5j * (phi + omega)) * c, -cmath.exp(0.5j * (phi - omega)) * s],
            [cmath.exp(-0.5j * (phi - omega)) * s, cmath.exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat, dtype=self._dtype)


class CRX(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RX gate."""
        super().__init__(name="CRX", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return np.array(mat, dtype=self._dtype)


class CRY(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RY gate."""
        super().__init__(name="CRY", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return np.array(mat, dtype=self._dtype)


class CRZ(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-RZ gate."""
        super().__init__(name="CRZ", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-0.5j * theta), 0],
            [0, 0, 0, cmath.exp(0.5j * theta)],
        ]
        return np.array(mat, dtype=self._dtype)


class CRot(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a controlled-rotation gate."""
        super().__init__(name="CRot", num_wires=2, params=params, **kwargs)
        self._validate(want_num_params=3)

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
        return np.array(mat, dtype=self._dtype)


class U1(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a U1 gate."""
        super().__init__(name="U1", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=1)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]
        return np.array(mat, dtype=self._dtype)


class U2(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs a U2 gate."""
        super().__init__(name="U2", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=2)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, lam = self.params
        mat = [
            [INV_SQRT2, -INV_SQRT2 * cmath.exp(1j * lam)],
            [INV_SQRT2 * cmath.exp(1j * phi), INV_SQRT2 * cmath.exp(1j * (phi + lam))],
        ]
        return np.array(mat, dtype=self._dtype)


class U3(Gate):
    def __init__(self, *params, **kwargs):
        """Constructs an arbitrary single-qubit unitary gate."""
        super().__init__(name="U3", num_wires=1, params=params, **kwargs)
        self._validate(want_num_params=3)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta, phi, lam = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [c, -s * cmath.exp(1j * lam)],
            [s * cmath.exp(1j * phi), c * cmath.exp(1j * (phi + lam))],
        ]
        return np.array(mat, dtype=self._dtype)
