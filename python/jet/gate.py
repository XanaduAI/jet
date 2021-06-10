from abc import ABC, abstractmethod
from cmath import exp
from functools import lru_cache
from math import cos, sin, sqrt
from typing import Callable, Dict, List, Optional, Sequence

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
    "GateFactory",
    # CV Fock gates
    "Displacement",
    "Squeezing",
    "TwoModeSqueezing",
    "Beamsplitter",
    # Qubit gates
    "Hadamard",
    "NOT",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "SX",
    "CNOT",
    "CX",
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


INV_SQRT2 = 1 / sqrt(2)


class Gate(ABC):
    def __init__(
        self,
        name: str,
        num_wires: int,
        params: Optional[List[float]] = None,
        tensor_id: Optional[int] = None,
    ):
        """Constructs a quantum gate.

        Args:
            name (str): name of the gate.
            num_wires (int): number of wires the gate is applied to.
            params (list or None): parameters of the gate.
            tensor_id (int or None): ID of the gate tensor.
        """
        self.name = name
        self.tensor_id = tensor_id

        self._indices = None
        self._num_wires = num_wires
        self._params = params

    @property
    def indices(self) -> Optional[Sequence[str]]:
        """Returns the indices of this gate. An index is a label associated with
        an axis of the tensor representation of a gate; the indices of a tensor
        determine its connectivity in the context of a tensor network.
        """
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this gate. If the indices of a gate are not None,
        they are used to construct the tensor representation of that gate. See
        @indices.getter for more information about tensor indices.

        Raises:
            ValueError if the given indices are not a sequence of unique strings
            or the number of provided indices is invalid.

        Args:
            indices (Sequence[str] or None): new indices of the gate.
        """
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

        # Check that `indices` has the correct length.
        elif len(indices) != 2 * self._num_wires:
            raise ValueError(
                f"Gates must have two indices per wire; received {len(indices)} "
                f"indices for {self._num_wires} wires."
            )

        self._indices = indices

    @property
    def num_wires(self) -> int:
        """Returns the number of wires connected to this gate."""
        return self._num_wires

    @property
    def params(self) -> Optional[List[float]]:
        """Returns the parameters of this gate."""
        return self._params

    @abstractmethod
    def _data(self) -> np.ndarray:
        """Returns the matrix representation of this gate."""
        pass

    def tensor(self, dtype: type = np.complex128, adjoint: bool = False) -> TensorType:
        """Returns the tensor representation of this gate.

        Args:
            dtype (type): data type of the tensor.
            adjoint (bool): whether to take the adjoint of the tensor.
        """
        if adjoint:
            data = np.linalg.inv(self._data()).flatten()
        else:
            data = self._data().flatten()

        indices = self.indices
        if indices is None:
            indices = list(map(str, range(2 * self._num_wires)))

        dimension = int(round(len(data) ** (1 / len(indices))))
        shape = [dimension] * len(indices)

        return Tensor(indices=indices, shape=shape, data=data, dtype=dtype)


class GateFactory:
    """GateFactory is an implementation of the factory design pattern for the
    Gate class. The create() method constructs a Gate instance from a name that
    has been registered by a Gate subclass using the @register decorator.
    """

    """Map that associates names with concrete Gate subclasses."""
    registry: Dict[str, type] = {}

    @staticmethod
    def create(name: str, *params: float, **kwargs) -> Gate:
        """Constructs a gate by name.

        Raises:
            KeyError if there is no entry for the given name in the registry.

        Args:
            name (str): registered name of the desired gate.
            params (float): Parameters to pass to the gate constructor.
            kwargs: Keyword arguments to pass to the gate constructor.

        Returns:
            The constructed gate.
        """
        key = name.lower()
        if key not in GateFactory.registry:
            raise KeyError(f"The key '{key}' does not exist in the gate registry.")

        subclass = GateFactory.registry[key]
        return subclass(*params, **kwargs)

    @staticmethod
    def register(names: Sequence[str]) -> Callable[[type], type]:
        """Registers a set of names with a class type.

        Raises:
            ValueError if the provided class does not inherit from Gate.

            KeyError if a name has already been registered to another class.
        """

        def wrapper(subclass: type) -> type:
            if not issubclass(subclass, Gate):
                raise ValueError(f"The type '{subclass.__name__}' is not a subclass of Gate")

            # Let the caller specify duplicate keys if they wish.
            keys = set(name.lower() for name in names)

            conflicts = keys & set(GateFactory.registry)
            if conflicts:
                raise KeyError(f"The keys {conflicts} already exist in the gate registry.")

            for key in keys:
                GateFactory.registry[key] = subclass

            return subclass

        return wrapper


####################################################################################################
# Continuous variable Fock gates
####################################################################################################


@GateFactory.register(names=["Displacement", "D"])
class Displacement(Gate):
    def __init__(self, r: float, phi: float, cutoff: int, **kwargs):
        """Constructs a displacement gate.  See `thewalrus.displacement
        <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.displacement.html>`__
        for more details.

        Args:
            r (float): displacement magnitude.
            phi (float): displacement angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Displacement", num_wires=1, params=[r, phi, cutoff], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        return displacement(*self.params)


@GateFactory.register(names=["Squeezing"])
class Squeezing(Gate):
    def __init__(self, r: float, theta: float, cutoff: int, **kwargs):
        """Constructs a squeezing gate.  See `thewalrus.squeezing
        <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.squeezing.html>`__
        for more details.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Squeezing", num_wires=1, params=[r, theta, cutoff], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        return squeezing(*self.params)


@GateFactory.register(names=["TwoModeSqueezing"])
class TwoModeSqueezing(Gate):
    def __init__(self, r: float, theta: float, cutoff: int, **kwargs):
        """Constructs a two-mode squeezing gate.  See `thewalrus.two_mode_squeezing
        <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.two_mode_squeezing.html>`__
        for more details.

        Args:
            r (float): squeezing magnitude.
            theta (float): squeezing angle.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="TwoModeSqueezing", num_wires=2, params=[r, theta, cutoff], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        return two_mode_squeezing(*self.params)


@GateFactory.register(names=["Beamsplitter", "BS"])
class Beamsplitter(Gate):
    def __init__(self, theta: float, phi: float, cutoff: int, **kwargs):
        """Constructs a beamsplitter gate.  See `thewalrus.beamsplitter
        <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.beamsplitter.html>`__
        for more details.

        Args:
            theta (float): transmissivity angle of the beamsplitter. The transmissivity is
                           :math:`t=\\cos(\\theta)`.
            phi (float): reflection phase of the beamsplitter.
            cutoff (int): Fock ladder cutoff.
        """
        super().__init__(name="Beamsplitter", num_wires=2, params=[theta, phi, cutoff], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        return beamsplitter(*self.params)


####################################################################################################
# Qubit gates
####################################################################################################


@GateFactory.register(names=["Hadamard", "H"])
class Hadamard(Gate):
    def __init__(self, **kwargs):
        """Constructs a Hadamard gate."""
        super().__init__(name="Hadamard", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        """Hadamard matrix"""
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat)


@GateFactory.register(names=["PauliX", "X", "NOT"])
class PauliX(Gate):
    def __init__(self, **kwargs):
        """Constructs a Pauli-X gate."""
        super().__init__(name="PauliX", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, 1], [1, 0]]
        return np.array(mat)


@GateFactory.register(names=["PauliY", "Y"])
class PauliY(Gate):
    def __init__(self, **kwargs):
        """Constructs a Pauli-Y gate."""
        super().__init__(name="PauliY", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat)


@GateFactory.register(names=["PauliZ", "Z"])
class PauliZ(Gate):
    def __init__(self, **kwargs):
        """Constructs a Pauli-Z gate."""
        super().__init__(name="PauliZ", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, -1]]
        return np.array(mat)


@GateFactory.register(names=["S"])
class S(Gate):
    def __init__(self, **kwargs):
        """Constructs a single-qubit phase gate."""
        super().__init__(name="S", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, 1j]]
        return np.array(mat)


@GateFactory.register(names=["T"])
class T(Gate):
    def __init__(self, **kwargs):
        """Constructs a single-qubit T gate."""
        super().__init__(name="T", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0], [0, exp(0.25j * np.pi)]]
        return np.array(mat)


@GateFactory.register(names=["SX"])
class SX(Gate):
    def __init__(self, **kwargs):
        """Constructs a single-qubit Square-Root X gate."""
        super().__init__(name="SX", num_wires=1, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat)


@GateFactory.register(names=["PhaseShift"])
class PhaseShift(Gate):
    def __init__(self, phi: float, **kwargs):
        """Constructs a single-qubit local phase shift gate.

        Args:
            phi (float): phase shift.
        """
        super().__init__(name="PhaseShift", num_wires=1, params=[phi], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["CPhaseShift"])
class CPhaseShift(Gate):
    def __init__(self, phi: float, **kwargs):
        """Constructs a controlled phase shift gate.

        Args:
            phi (float): phase shift.
        """
        super().__init__(name="CPhaseShift", num_wires=2, params=[phi], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["CX", "CNOT"])
class CX(Gate):
    def __init__(self, **kwargs):
        """Constructs a controlled-X gate."""
        super().__init__(name="CX", num_wires=2, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat)


@GateFactory.register(names=["CY"])
class CY(Gate):
    def __init__(self, **kwargs):
        """Constructs a controlled-Y gate."""
        super().__init__(name="CY", num_wires=2, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat)


@GateFactory.register(names=["CZ"])
class CZ(Gate):
    def __init__(self, **kwargs):
        """Constructs a controlled-Z gate."""
        super().__init__(name="CZ", num_wires=2, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat)


@GateFactory.register(names=["SWAP"])
class SWAP(Gate):
    def __init__(self, **kwargs):
        """Constructs a SWAP gate."""
        super().__init__(name="SWAP", num_wires=2, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


@GateFactory.register(names=["ISWAP"])
class ISWAP(Gate):
    def __init__(self, **kwargs):
        """Constructs an ISWAP gate."""
        super().__init__(name="ISWAP", num_wires=2, **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


@GateFactory.register(names=["CSWAP"])
class CSWAP(Gate):
    def __init__(self, **kwargs):
        """Constructs a CSWAP gate."""
        super().__init__(name="CSWAP", num_wires=3, **kwargs)

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


@GateFactory.register(names=["Toffoli"])
class Toffoli(Gate):
    def __init__(self, **kwargs):
        """Constructs a Toffoli gate."""
        super().__init__(name="Toffoli", num_wires=3, **kwargs)

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


@GateFactory.register(names=["RX"])
class RX(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a single-qubit X rotation gate.

        Args:
            theta (float): rotation angle around the X-axis.
        """
        super().__init__(name="RX", num_wires=1, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = cos(theta / 2)
        js = 1j * sin(-theta / 2)

        mat = [[c, js], [js, c]]
        return np.array(mat)


@GateFactory.register(names=["RY"])
class RY(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a single-qubit Y rotation gate.

        Args:
            theta (float): rotation angle around the Y-axis.
        """
        super().__init__(name="RY", num_wires=1, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]

        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [[c, -s], [s, c]]
        return np.array(mat)


@GateFactory.register(names=["RZ"])
class RZ(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a single-qubit Z rotation gate.

        Args:
            theta (float): rotation angle around the Z-axis.
        """
        super().__init__(name="RZ", num_wires=1, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        p = exp(-0.5j * theta)

        mat = [[p, 0], [0, np.conj(p)]]
        return np.array(mat)


@GateFactory.register(names=["Rot"])
class Rot(Gate):
    def __init__(self, phi: float, theta: float, omega: float, **kwargs):
        """Constructs an arbitrary single-qubit rotation gate.  Each of the Pauli
        rotation gates can be recovered by fixing two of the three parameters:

        >>> assert RX(theta).tensor() == Rot(pi/2, -pi/2, theta).tensor()
        >>> assert RY(theta).tensor() == Rot(0, 0, theta).tensor()
        >>> assert RZ(theta).tensor() == Rot(theta, 0, 0).tensor()

        Args:
            phi (float): first rotation angle.
            theta (float): second rotation angle.
            omega (float): third rotation angle.
        """
        super().__init__(name="Rot", num_wires=1, params=[phi, theta, omega], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, theta, omega = self.params
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [
            [exp(-0.5j * (phi + omega)) * c, -exp(0.5j * (phi - omega)) * s],
            [exp(-0.5j * (phi - omega)) * s, exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat)


@GateFactory.register(names=["CRX"])
class CRX(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a controlled-RX gate.

        Args:
            theta (float): rotation angle around the X-axis.
        """
        super().__init__(name="CRX", num_wires=2, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = cos(theta / 2)
        js = 1j * sin(-theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return np.array(mat)


@GateFactory.register(names=["CRY"])
class CRY(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a controlled-RY gate.

        Args:
            theta (float): rotation angle around the Y-axis.
        """
        super().__init__(name="CRY", num_wires=2, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return np.array(mat)


@GateFactory.register(names=["CRZ"])
class CRZ(Gate):
    def __init__(self, theta: float, **kwargs):
        """Constructs a controlled-RZ gate.

        Args:
            theta (float): rotation angle around the Z-axis.
        """
        super().__init__(name="CRZ", num_wires=2, params=[theta], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp(-0.5j * theta), 0],
            [0, 0, 0, exp(0.5j * theta)],
        ]
        return np.array(mat)


@GateFactory.register(names=["CRot"])
class CRot(Gate):
    def __init__(self, phi: float, theta: float, omega: float, **kwargs):
        """Constructs a controlled-rotation gate.

        Args:
            phi (float): first rotation angle.
            theta (float): second rotation angle.
            omega (float): third rotation angle.
        """
        super().__init__(name="CRot", num_wires=2, params=[phi, theta, omega], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, theta, omega = self.params
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp(-0.5j * (phi + omega)) * c, -exp(0.5j * (phi - omega)) * s],
            [0, 0, exp(-0.5j * (phi - omega)) * s, exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat)


@GateFactory.register(names=["U1"])
class U1(Gate):
    def __init__(self, phi: float, **kwargs):
        """Constructs a U1 gate.

        Args:
            phi (float): rotation angle.
        """
        super().__init__(name="U1", num_wires=1, params=[phi], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi = self.params[0]
        mat = [[1, 0], [0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["U2"])
class U2(Gate):
    def __init__(self, phi: float, lam: float, **kwargs):
        """Constructs a U2 gate.

        Args:
            phi (float): first rotation angle.
            lam (float): second rotation angle.
        """
        super().__init__(name="U2", num_wires=1, params=[phi, lam], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        phi, lam = self.params
        mat = [
            [INV_SQRT2, -INV_SQRT2 * exp(1j * lam)],
            [INV_SQRT2 * exp(1j * phi), INV_SQRT2 * exp(1j * (phi + lam))],
        ]
        return np.array(mat)


@GateFactory.register(names=["U3"])
class U3(Gate):
    def __init__(self, theta: float, phi: float, lam: float, **kwargs):
        """Constructs a U3 gate.

        Args:
            theta (float): first rotation angle.
            phi (float): second rotation angle.
            lam (float): third rotation angle.
        """
        super().__init__(name="U3", num_wires=1, params=[theta, phi, lam], **kwargs)

    @lru_cache
    def _data(self) -> np.ndarray:
        theta, phi, lam = self.params
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [
            [c, -s * exp(1j * lam)],
            [s * exp(1j * phi), c * exp(1j * (phi + lam))],
        ]
        return np.array(mat)


# Some gates have different names depending on their context.
NOT = PauliX
CNOT = CX
