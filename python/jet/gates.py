import cmath
import math
from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np
from thewalrus.fock_gradients import (
    beamsplitter,
    displacement,
    squeezing,
    two_mode_squeezing,
)

import jet

__all__ = [
    # CV fock gates
    "Displacement",
    "Squeezing",
    "TwoModeSqueezing",
    "Beamsplitter",
    # Qubit gates
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

INV_SQRT2 = 1 / math.sqrt(2)


class Gate:
    def __init__(self, name: str, num_wires: int, **kwargs) -> None:
        """Constructs a quantum gate.

        Args:
            name: name of the gate.
            num_wires: number of wires the gate is applied to.

        Kwargs:
            tensor_id (int): identification number for the gate-tensor.
            dtype (type): type to use in matrix representations of gates.
        """
        self.name = name
        self.tensor_id = kwargs.get("tensor_id", None)

        self._dtype = kwargs.get("dtype", np.complex128)
        self._indices = None
        self._num_wires = num_wires

    def tensor(self, adjoint: bool = False) -> jet.Tensor:
        """Returns the tensor representation of this gate."""
        if adjoint:
            data = np.conj(self._data()).T.flatten()
        else:
            data = self._data().flatten()

        indices = self.indices
        if indices is None:
            indices = list(map(str, range(2 * self._num_wires)))

        dimension = int(len(data) ** (1 / len(indices)))
        shape = [dimension] * len(indices)

        return jet.Tensor(indices=indices, shape=shape, data=data)

    def _data(self) -> np.ndarray:
        """Returns the matrix representation of this gate."""
        raise NotImplementedError("No tensor data available for generic gate.")

    @property
    def indices(self) -> Optional[List[str]]:
        """Returns the indices of this gate for connecting tensors."""
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this gate for connecting tensors."""
        # Check that `indices is a sequence of unique strings.
        if (
            not isinstance(indices, Sequence)
            or not all(isinstance(idx, str) for idx in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("Indices must be a sequence of unique strings.")

        # Check that `indices` has the correct length (or is None)
        if indices is not None and len(indices) != 2 * self._num_wires:
            raise ValueError(
                f"Indices must have two indices per wire. "
                f"Received {len(indices)} indices for {self._num_wires} wires."
            )
        self._indices = indices


##################################
# Continuous variable Fock gates
##################################


class Displacement(Gate):
    """Displacement gate

    Args:
        r (float): displacement magnitude
        phi (float): displacement angle
        cutoff (int): Fock ladder cutoff
    """

    def __init__(self, *params, **kwargs) -> None:
        name = "Displacement"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the displacement gate

        Returns:
            array[complex]: matrix representing the displacement gate.
        """
        return displacement(*self.params, dtype=self._dtype)


class Squeezing(Gate):
    """Squeezing gate

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff
    """

    def __init__(self, *params, **kwargs) -> None:
        name = "Squeezing"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the squeezing gate

        Returns:
            array[complex]: matrix representing the squeezing gate.
        """
        return squeezing(*self.params, dtype=self._dtype)


class TwoModeSqueezing(Gate):
    """TwoModeSqueezing gate

    Args:
        r (float): squeezing magnitude
        theta (float): squeezing angle
        cutoff (int): Fock ladder cutoff
    """

    def __init__(self, *params, **kwargs) -> None:
        name = "TwoModeSqueezing"
        num_wires = 2
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the two-mode squeezing gate

        Returns:
            array[complex]: matrix representing the two-mode squeezing gate.
        """
        return two_mode_squeezing(*self.params, dtype=self._dtype)


class Beamsplitter(Gate):
    """Beamsplitter gate

    Args:
        theta (float): transmissivity angle of the beamsplitter. The transmissivity is :math:`t=\cos(\theta)`
        phi (float): reflection phase of the beamsplitter
        cutoff (int): Fock ladder cutoff
    """

    def __init__(self, *params, **kwargs) -> None:
        name = "Beamsplitter"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """The matrix representation of the beamsplitter gate

        Returns:
            array[complex]: matrix representing the beamsplitter gate.
        """
        return beamsplitter(*self.params, dtype=self._dtype)


###############
# Qubit gates
###############


class CNOT(Gate):
    """CNOT gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CNOT"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """CNOT matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat, dtype=self._dtype)


class Hadamard(Gate):
    """Hadamard gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "Hadamard"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Hadamard matrix"""
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat, dtype=self._dtype)


class PauliX(Gate):
    """PauliX gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "PauliX"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliX matrix"""
        mat = [[0, 1], [1, 0]]
        return np.array(mat, dtype=self._dtype)


class PauliY(Gate):
    """PauliX gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "PauliY"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliY matrix"""
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat, dtype=self._dtype)


class PauliZ(Gate):
    """PauliZ gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "PauliZ"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """PauliZ matrix"""
        mat = [[1, 0], [0, -1]]
        return np.array(mat, dtype=self._dtype)


class S(Gate):
    """The single-qubit phase gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "S"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit phase gate matrix"""
        mat = [[1, 0], [0, 1j]]
        return np.array(mat, dtype=self._dtype)


class T(Gate):
    """The single-qubit T gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "T"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit T gate matrix"""
        mat = [[1, 0], [0, cmath.exp(0.25j * np.pi)]]
        return np.array(mat, dtype=self._dtype)


class SX(Gate):
    """The single-qubit Square-Root X gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "SX"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single-qubit Square-Root X operator matrix"""
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat, dtype=self._dtype)


class CZ(Gate):
    """The controlled-Z gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CZ"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-Z gate matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat, dtype=self._dtype)


class CY(Gate):
    """The controlled-Y gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CY"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-Y operator matrix"""
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat, dtype=self._dtype)


class SWAP(Gate):
    """The swap gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "SWAP"
        num_wires = 2
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Swap operator matrix"""
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=self._dtype)


class ISWAP(Gate):
    """The i-swap gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "iSWAP"
        num_wires = 1
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """i-swap operator matrix"""
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat, dtype=self._dtype)


class CSWAP(Gate):
    """The CSWAP gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CSWAP"
        num_wires = 3
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


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
        return np.array(mat, dtype=self._dtype)


class Toffoli(Gate):
    """The Toffoli gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "Toffoli"
        num_wires = 3
        num_params = 0

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


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
        return np.array(mat, dtype=self._dtype)


class RX(Gate):
    """The single qubit X rotation gate"""

    def __init__(self, *params: float, **kwargs) -> None:
        name = "RX"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single qubit X rotation matrix"""
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[c, js], [js, c]]
        return np.array(mat, dtype=self._dtype)


class RY(Gate):
    """The single qubit Y rotation gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "RY"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single qubit Y rotation matrix"""
        theta = self.params[0]

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[c, -s], [s, c]]
        return np.array(mat, self._dtype)


class RZ(Gate):
    """The single qubit Z rotation gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "RZ"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single qubit Z rotation matrix"""
        theta = self.params[0]
        p = cmath.exp(-0.5j * theta)

        mat = [[p, 0], [0, np.conj(p)]]
        return np.array(mat, dtype=self._dtype)


class PhaseShift(Gate):
    """The single qubit local phase shift gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "PhaseShift"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Single qubit local phase shift operator matrix"""
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]

        return np.array(mat, dtype=self._dtype)


class CPhase(Gate):
    """The controlled phase shift gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CPhase"
        num_wires = 2
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled phase shift operator matrix"""
        phi = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, cmath.exp(1j * phi)]
        ]
        return np.array(mat, dtype=self._dtype)

class Rot(Gate):
    """The arbitrary single qubit rotation gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "Rot"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Arbitrary single qubit rotation operator matrix"""
        phi, theta, omega = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat =[
            [cmath.exp(-0.5j * (phi + omega)) * c, -cmath.exp(0.5j * (phi - omega)) * s],
            [cmath.exp(-0.5j * (phi - omega)) * s, cmath.exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat, dtype=self._dtype)


class CRX(Gate):
    """The controlled-RX gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CRX"
        num_wires = 2
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-RX operator matrix"""
        theta = self.params[0]
        c = math.cos(theta / 2)
        js = 1j * math.sin(-theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return np.array(mat, dtype=self._dtype)


class CRY(Gate):
    """The controlled-RY gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CRY"
        num_wires = 2
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-RY operator matrix"""
        theta = self.params[0]
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return np.array(mat, dtype=self._dtype)


class CRZ(Gate):
    """The controlled-RZ gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CRZ"
        num_wires = 2
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-RZ operator matrix"""
        theta = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cmath.exp(-0.5j * theta), 0],
            [0, 0, 0, cmath.exp(0.5j * theta)],
        ]
        return np.array(mat, dtype=self._dtype)


class CRot(Gate):
    """The controlled-rotation gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "CRZ"
        num_wires = 2
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Controlled-rotation operator matrix"""
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
    """The U1 gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "U1"
        num_wires = 1
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """U1 operator matrix"""
        phi = self.params[0]
        mat = [[1, 0], [0, cmath.exp(1j * phi)]]
        return np.array(mat, dtype=self._dtype)


class U2(Gate):
    """The U2 gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "U2"
        num_wires = 2
        num_params = 1

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """U2 operator matrix"""
        phi, lam = self.params
        mat = [
            [INV_SQRT2, -INV_SQRT2 * cmath.exp(1j * lam)],
            [INV_SQRT2 * cmath.exp(1j * phi), INV_SQRT2 * cmath.exp(1j * (phi + lam))]
        ]
        return np.array(mat, dtype=self._dtype)


class U3(Gate):
    """The arbitrary single qubit unitary gate"""

    def __init__(self, *params, **kwargs) -> None:
        name = "U3"
        num_wires = 1
        num_params = 3

        if len(params) != num_params:
            raise ValueError(f"{len(params)} passed. The {name} gate only accepts {num_params} parameters.")
        self.params = params

        super().__init__(name, num_wires, **kwargs)


    @lru_cache
    def _data(self) -> np.ndarray:
        """Arbitrary single qubit unitary operator matrix"""
        theta, phi, lam = self.params
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        mat = [
            [c, -s * cmath.exp(1j * lam)],
            [s * cmath.exp(1j * phi), c * cmath.exp(1j * (phi + lam))],
        ]
        return np.array(mat, dtype=self._dtype)
