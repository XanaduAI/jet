"""Tensor representations of quantum gates"""

import math
import cmath
from numbers import Number
from functools import lru_cache
from typing import Sequence, Union, List, Optional

import numpy as np
from thewalrus.fock_gradients import (
    displacement,
    grad_displacement,
    squeezing,
    grad_squeezing,
    two_mode_squeezing,
    grad_two_mode_squeezing,
    beamsplitter,
    grad_beamsplitter
)

import jet

__all__ = [
    "Displacement",
    "Squeezing",
    "TwoModeSqueezing",
    "Beamsplitter",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PauliRot",
    "MultiRZ",
    "S",
    "T",
    "SX",
    "CNOT",
    "CZ",
    "CY",
    "SWAP",
    "ISWAP",
    "CSWAP",
    "Toffoli",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "ControlledPhaseShift",
    "CPhase",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U1",
    "U2",
    "U3",
]

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
INV_SQRT2 = 1 / math.sqrt(2)

class Gate:
    """Gate class

    Args:
        name (str): name of the gate
        num_wires (int): the number of wires the gate is applied to

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, name: str, num_wires: int, tensor_id: int = None) -> None:
        self.name = name
        self.tensor_id = tensor_id

        self._indices = None
        self._num_wires = num_wires

    def tensor(self, adjoint: bool = False):
        """Tensor representation of gate"""
        if adjoint:
            data = np.conj(self._data()).T.flatten()
        else:
            data = self._data().flatten()

        indices = self.indices
        if indices is None:
            indices = list(ALPHABET[:2 * self._num_wires])
        shape = int(len(data) ** ((2 * self._num_wires) ** -1))

        return jet.Tensor(indices, [shape] * 2 * self._num_wires, data)

    def _data(self) -> np.ndarray:
        """Matrix representation of the gate"""
        raise NotImplementedError("No tensor data available for generic gate.")

    @property
    def indices(self) -> Optional[List[str]]:
        """Indices for connecting tensors"""
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[List[str]]) -> None:
        """Setter method for indices"""
        # validate that indices is a list of unique strings
        if (
            not isinstance(indices, Sequence)
            or not all(isinstance(idx, str) for idx in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Indices must be a sequence of unique strings.")

        # validate that indices has the correct lenght (or is None)
        if indices is None:
            self._indices = indices
        elif len(indices) == 2 * self._num_wires:
            self._indices = indices
        else:
            raise ValueError(
                f"Must have 2 indices per wire. Got {len(indices)} indices for"
                f"{self._num_wires} wires."
            )


class Displacement(Gate):
    """Displacement gate

    Args:
        r (float): displacement magnitude
        phi (float): displacement angle
        cutoff (int): Fock ladder cutoff

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "Displacement"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the displacement gate

        Returns:
            array[complex]: matrix representing the displacement gate.
        """
        return displacement(*self.params, dtype=np.complex128)


class Squeezing(Gate):
    """Squeezing gate

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "Squeezing"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the squeezing gate

        Returns:
            array[complex]: matrix representing the squeezing gate.
        """
        return squeezing(*self.params, dtype=np.complex128)


class TwoModeSqueezing(Gate):
    """TwoModeSqueezing gate

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "TwoModeSqueezing"
        num_wires = 2
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the two-mode squeezing gate

        Returns:
            array[complex]: matrix representing the two-mode squeezing gate.
        """
        return two_mode_squeezing(*self.params, dtype=np.complex128)


class Beamsplitter(Gate):
    """Beamsplitter gate

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "Beamsplitter"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the beamsplitter gate

        Returns:
            array[complex]: matrix representing the beamsplitter gate.
        """
        return beamsplitter(*self.params, dtype=np.complex128)


class CNOT(Gate):
    """CNOT gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "CNOT"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """CNOT matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat, dtype=np.complex128)


class Hadamard(Gate):
    """Hadamard gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "Hadamard"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Hadamard matrix"""
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat, dtype=np.complex128)


class PauliX(Gate):
    """PauliX gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "PauliX"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliX matrix"""
        mat = [[0, 1], [1, 0]]
        return np.array(mat, dtype=np.complex128)


class PauliY(Gate):
    """PauliX gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "PauliY"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliY matrix"""
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat, dtype=np.complex128)


class PauliZ(Gate):
    """PauliZ gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "PauliZ"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliZ matrix"""
        mat = [[1, 0], [0, -1]]
        return np.array(mat, dtype=np.complex128)


class S(Gate):
    """The single-qubit phase gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "S"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit phase gate matrix"""
        mat = [[1, 0], [0, 1j]]
        return np.array(mat, dtype=np.complex128)


class T(Gate):
    """The single-qubit T gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "T"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit T gate matrix"""
        mat = [[1, 0], [0, cmath.exp(0.25j * np.pi)]]
        return np.array(mat, dtype=np.complex128)


class SX(Gate):
    """The single-qubit Square-Root X gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "SX"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit Square-Root X operator matrix"""
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat, dtype=np.complex128)


class CZ(Gate):
    """The controlled-Z gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "CZ"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-Z gate matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat, dtype=np.complex128)


class CY(Gate):
    """The controlled-Y gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "CY"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-Y operator matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat, dtype=np.complex128)


class SWAP(Gate):
    """The swap gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "SWAP"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Swap operator matrix"""
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=np.complex128)


class ISWAP(Gate):
    """The i-swap gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "iSWAP"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """i-swap operator matrix"""
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=np.complex128)


class CSWAP(Gate):
    """The CSWAP gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "CSWAP"
        num_wires = 3
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """CSWAP operator matrix"""
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
        return np.array(mat, dtype=np.complex128)


class Toffoli(Gate):
    """The Toffoli gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: Number, tensor_id: int) -> None:
        name = "Toffoli"
        num_wires = 3
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Toffoli operator matrix"""
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
        return np.array(mat, dtype=np.complex128)


class RX(Gate):
    """The single qubit X gate

    Kwargs:
        tensor_id (int): identification number for the gate-tensor
    """

    def __init__(self, *params: float, tensor_id: int) -> None:
        name = "RX"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, tensor_id=tensor_id)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single qubit X operator matrix"""
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)
        return np.array([[c, js], [js, c]], dtype=np.complex128)
