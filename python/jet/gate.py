"""Module containing the ``Gate`` and ``GateFactory`` classes in addition to all
``Gate`` subclasses.
"""
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
    # Decorator gates
    "Adjoint",
    "Scale",
    # CV Fock gates
    "FockGate",
    "Displacement",
    "Squeezing",
    "TwoModeSqueezing",
    "Beamsplitter",
    # Qubit gates
    "QubitGate",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "SX",
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
    """Gate represents a quantum gate.

    Args:
        name (str): Name of the gate.
        num_wires (int): Number of wires the gate is applied to.
        dim (int): Dimension of the gate. This should match the dimension of the
            qudits the gate can be applied to.
        params (List[float] or None): Parameters of the gate.

    Raises:
        ValueError: If the dimension is invalid.
    """

    def __init__(self, name: str, num_wires: int, dim: int, params: Optional[List[float]] = None):
        self.name = name

        self._indices = None
        self._num_wires = num_wires
        self._params = params

        self._validate_dimension(dim)
        self._dim = dim

    @property
    def dimension(self) -> int:
        """Returns the dimension of this gate."""
        return self._dim

    @dimension.setter
    def dimension(self, dim: int) -> None:
        """Sets the dimension of this gate.

        Args:
            dim (int): New dimension of this gate.

        Raises:
            ValueError: If the dimension is invalid.
        """
        self._validate_dimension(dim)
        self._dim = dim

    @abstractmethod
    def _validate_dimension(self, dim: int) -> None:
        """Validates a candidate dimension for this gate.

        Args:
            dim (int): Dimension to be validated.

        Raises:
            ValueError: If the dimension is invalid.
        """

    @property
    def indices(self) -> Optional[Sequence[str]]:
        """Returns the indices of this gate. An index is a label associated with
        an axis of the tensor representation of a gate; the indices of a tensor
        determine its connectivity in the context of a tensor network.
        """
        return self._indices

    @indices.setter
    def indices(self, indices: Optional[Sequence[str]]) -> None:
        """Sets the indices of this gate. If the indices of a gate are not ``None``,
        they are used to construct the tensor representation of that gate. See
        @indices.getter for more information about tensor indices.

        Raises:
            ValueError: If the given indices are not a sequence of unique strings
                or the number of provided indices is invalid.

        Args:
            indices (Sequence[str] or None): New indices of the gate.
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
        """Returns the number of wires this gate acts on."""
        return self._num_wires

    @property
    def params(self) -> Optional[List[float]]:
        """Returns the parameters of this gate."""
        return self._params

    @abstractmethod
    def _data(self) -> np.ndarray:
        """Returns the matrix representation of this gate."""

    def tensor(self, dtype: np.dtype = np.complex128) -> TensorType:
        """Returns the tensor representation of this gate.

        Args:
            dtype (np.dtype): Data type of the tensor.
        """
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

    registry: Dict[str, type] = {}
    """Map that associates names with concrete Gate subclasses."""

    @staticmethod
    def create(
        name: str, *params: float, adjoint: bool = False, scalar: float = 1, **kwargs
    ) -> Gate:
        """Constructs a gate by name.

        Raises:
            KeyError: If there is no entry for the given name in the registry.

        Args:
            name (str): Registered name of the desired gate.
            params (float): Parameters to pass to the gate constructor.
            adjoint (bool): Whether to take the adjoint of the gate.
            scalar (float): Scaling factor to apply to the gate.
            kwargs: Keyword arguments to pass to the gate constructor.

        Returns:
            Gate: The constructed gate.
        """
        if name not in GateFactory.registry:
            raise KeyError(f"The name '{name}' does not exist in the gate registry.")

        subclass = GateFactory.registry[name]
        gate = subclass(*params, **kwargs)

        if adjoint:
            gate = Adjoint(gate=gate)

        if scalar != 1:
            gate = Scale(gate=gate, scalar=scalar)

        return gate

    @staticmethod
    def register(names: Sequence[str]) -> Callable[[type], type]:
        """Registers a set of names with a class type.

        Raises:
            ValueError: If the provided class does not inherit from Gate.
            KeyError: If a name is already registered to another class.
        """

        def wrapper(subclass: type) -> type:
            if not issubclass(subclass, Gate):
                raise ValueError(f"The type '{subclass.__name__}' is not a subclass of Gate.")

            # Let the caller specify duplicate keys if they wish.
            conflicts = set(names) & set(GateFactory.registry)
            if conflicts:
                raise KeyError(f"The names {conflicts} already exist in the gate registry.")

            for name in set(names):
                GateFactory.registry[name] = subclass

            return subclass

        return wrapper

    # pylint: disable=bad-staticmethod-argument
    @staticmethod
    def unregister(cls: type) -> None:
        """Unregisters a class type.

        Args:
            cls (type): Class type to remove from the registry.
        """
        for key in {key for key, val in GateFactory.registry.items() if val == cls}:
            del GateFactory.registry[key]


####################################################################################################
# Decorator gates
####################################################################################################


class Adjoint(Gate):
    """Adjoint is a decorator which computes the conjugate transpose of an existing ``Gate``.

    Args:
        gate (Gate): Gate to take the adjoint of.
    """

    def __init__(self, gate: Gate):
        self._gate = gate

        super().__init__(
            name=gate.name, num_wires=gate.num_wires, dim=gate.dimension, params=gate.params
        )

    def _data(self):
        # pylint: disable=protected-access
        return self._gate._data().conj().T

    def _validate_dimension(self, dim):
        # pylint: disable=protected-access
        self._gate._validate_dimension(dim)


class Scale(Gate):
    """Scale is a decorator which linearly scales an existing ``Gate``.

    Args:
        gate (Gate): Gate to scale.
        scalar (float): Scaling factor.
    """

    def __init__(self, gate: Gate, scalar: float):
        self._gate = gate
        self._scalar = scalar

        super().__init__(
            name=gate.name, num_wires=gate.num_wires, dim=gate.dimension, params=gate.params
        )

    def _data(self):
        # pylint: disable=protected-access
        return self._scalar * self._gate._data()

    def _validate_dimension(self, dim):
        # pylint: disable=protected-access
        self._gate._validate_dimension(dim)


####################################################################################################
# Continuous variable Fock gates
####################################################################################################


class FockGate(Gate):
    """FockGate represents a (continuous variable) Fock gate.

    Args:
        name (str): Name of the gate.
        num_wires (int): Number of wires the gate is applied to.
        cutoff (int): Fock ladder cutoff.
        params (List[float] or None): Parameters of the gate.
    """

    def __init__(
        self, name: str, num_wires: int, cutoff: int, params: Optional[List[float]] = None
    ):
        super().__init__(name=name, num_wires=num_wires, dim=cutoff, params=params)

    def _validate_dimension(self, dim):
        if dim < 2:
            raise ValueError("The dimension of a Fock gate must be greater than one.")


@GateFactory.register(names=["Displacement", "displacement", "D", "d"])
class Displacement(FockGate):
    """Displacement represents a displacement gate. See `thewalrus.displacement
    <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.displacement.html>`__
    for more details.

    Args:
        r (float): Displacement magnitude.
        phi (float): Displacement angle.
        cutoff (int): Fock ladder cutoff.
    """

    def __init__(self, r: float, phi: float, cutoff: int = 2):
        super().__init__(name="Displacement", num_wires=1, cutoff=cutoff, params=[r, phi])

    @lru_cache()
    def _data(self):
        return displacement(*self.params, cutoff=self.dimension)


@GateFactory.register(names=["Squeezing", "squeezing"])
class Squeezing(FockGate):
    """Squeezing represents a squeezing gate. See `thewalrus.squeezing
    <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.squeezing.html>`__
    for more details.

    Args:
        r (float): Squeezing magnitude.
        theta (float): Squeezing angle.
        cutoff (int): Fock ladder cutoff.
    """

    def __init__(self, r: float, theta: float, cutoff: int = 2):
        super().__init__(name="Squeezing", num_wires=1, cutoff=cutoff, params=[r, theta])

    @lru_cache()
    def _data(self):
        return squeezing(*self.params, cutoff=self.dimension)


@GateFactory.register(names=["TwoModeSqueezing", "twomodesqueezing"])
class TwoModeSqueezing(FockGate):
    """TwoModeSqueezing represents a two-mode squeezing gate. See `thewalrus.two_mode_squeezing
    <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.two_mode_squeezing.html>`__
    for more details.

    Args:
        r (float): Squeezing magnitude.
        theta (float): Squeezing angle.
        cutoff (int): Fock ladder cutoff.
    """

    def __init__(self, r: float, theta: float, cutoff: int = 2):
        super().__init__(name="TwoModeSqueezing", num_wires=2, cutoff=cutoff, params=[r, theta])

    @lru_cache()
    def _data(self):
        return two_mode_squeezing(*self.params, cutoff=self.dimension)


@GateFactory.register(
    names=[
        "Beamsplitter",
        "beamsplitter",
        "BS",
        "bs",
    ]
)
class Beamsplitter(FockGate):
    """Beamsplitter represents a beamsplitter gate. See `thewalrus.beamsplitter
    <https://the-walrus.readthedocs.io/en/latest/code/api/thewalrus.fock_gradients.beamsplitter.html>`__
    for more details.

    Args:
        theta (float): Transmissivity angle of the beamsplitter. The transmissivity is
                       :math:`t=\\cos(\\theta)`.
        phi (float): Reflection phase of the beamsplitter.
        cutoff (int): Fock ladder cutoff.
    """

    def __init__(self, theta: float, phi: float, cutoff: int = 2):
        super().__init__(name="Beamsplitter", num_wires=2, cutoff=cutoff, params=[theta, phi])

    @lru_cache()
    def _data(self):
        return beamsplitter(*self.params, cutoff=self.dimension)


####################################################################################################
# Qubit gates
####################################################################################################


class QubitGate(Gate):
    """QubitGate represents a qubit gate.

    Args:
        name (str): Name of the gate.
        num_wires (int): Number of wires the gate is applied to.
        params (List[float] or None): Parameters of the gate.
    """

    def __init__(self, name: str, num_wires: int, params: Optional[List[float]] = None):
        super().__init__(name=name, num_wires=num_wires, dim=2, params=params)

    def _validate_dimension(self, dim):
        if dim != 2:
            raise ValueError("The dimension of a qubit gate must be exactly two.")


@GateFactory.register(names=["Hadamard", "hadamard", "H", "h"])
class Hadamard(QubitGate):
    """Hadamard represents a Hadamard gate."""

    def __init__(self):
        super().__init__(name="Hadamard", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]]
        return np.array(mat)


@GateFactory.register(names=["PauliX", "paulix", "X", "x", "NOT", "not"])
class PauliX(QubitGate):
    """PauliX represents a Pauli-X gate."""

    def __init__(self):
        super().__init__(name="PauliX", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[0, 1], [1, 0]]
        return np.array(mat)


@GateFactory.register(names=["PauliY", "pauliy", "Y", "y"])
class PauliY(QubitGate):
    """PauliY represents a Pauli-Y gate."""

    def __init__(self):
        super().__init__(name="PauliY", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[0, -1j], [1j, 0]]
        return np.array(mat)


@GateFactory.register(names=["PauliZ", "pauliz", "Z", "z"])
class PauliZ(QubitGate):
    """PauliZ represents a Pauli-Z gate."""

    def __init__(self):
        super().__init__(name="PauliZ", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[1, 0], [0, -1]]
        return np.array(mat)


@GateFactory.register(names=["S", "s"])
class S(QubitGate):
    """S represents a single-qubit phase gate."""

    def __init__(self):
        super().__init__(name="S", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[1, 0], [0, 1j]]
        return np.array(mat)


@GateFactory.register(names=["T", "t"])
class T(QubitGate):
    """T represents a single-qubit T gate."""

    def __init__(self):
        super().__init__(name="T", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[1, 0], [0, exp(0.25j * np.pi)]]
        return np.array(mat)


@GateFactory.register(names=["SX", "sx"])
class SX(QubitGate):
    """SX represents a single-qubit Square-Root X gate."""

    def __init__(self):
        super().__init__(name="SX", num_wires=1)

    @lru_cache()
    def _data(self):
        mat = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]
        return np.array(mat)


@GateFactory.register(names=["PhaseShift", "phaseshift"])
class PhaseShift(QubitGate):
    """PhaseShift represents a single-qubit local phase shift gate.

    Args:
        phi (float): Phase shift angle.
    """

    def __init__(self, phi: float):
        super().__init__(name="PhaseShift", num_wires=1, params=[phi])

    @lru_cache()
    def _data(self):
        phi = self.params[0]
        mat = [[1, 0], [0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["CPhaseShift", "cphaseshift"])
class CPhaseShift(QubitGate):
    """CPhaseShift represents a controlled phase shift gate.

    Args:
        phi (float): Phase shift angle.
    """

    def __init__(self, phi: float):
        super().__init__(name="CPhaseShift", num_wires=2, params=[phi])

    @lru_cache()
    def _data(self):
        phi = self.params[0]
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["CX", "cx", "CNOT", "cnot"])
class CX(QubitGate):
    """CX represents a controlled-X gate."""

    def __init__(self):
        super().__init__(name="CX", num_wires=2)

    @lru_cache()
    def _data(self):
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        return np.array(mat)


@GateFactory.register(names=["CY", "cy"])
class CY(QubitGate):
    """CY represents a controlled-Y gate."""

    def __init__(self):
        super().__init__(name="CY", num_wires=2)

    @lru_cache()
    def _data(self):
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]
        return np.array(mat)


@GateFactory.register(names=["CZ", "cz"])
class CZ(QubitGate):
    """CZ represents a controlled-Z gate."""

    def __init__(self):
        super().__init__(name="CZ", num_wires=2)

    @lru_cache()
    def _data(self):
        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        return np.array(mat)


@GateFactory.register(names=["SWAP", "swap"])
class SWAP(QubitGate):
    """SWAP represents a SWAP gate."""

    def __init__(self):
        super().__init__(name="SWAP", num_wires=2)

    @lru_cache()
    def _data(self):
        mat = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


@GateFactory.register(names=["ISWAP", "iswap"])
class ISWAP(QubitGate):
    """ISWAP represents a ISWAP gate."""

    def __init__(self):
        super().__init__(name="ISWAP", num_wires=2)

    @lru_cache()
    def _data(self):
        mat = [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]]
        return np.array(mat)


@GateFactory.register(names=["CSWAP", "cswap"])
class CSWAP(QubitGate):
    """CSWAP represents a CSWAP gate."""

    def __init__(self):
        super().__init__(name="CSWAP", num_wires=3)

    @lru_cache()
    def _data(self):
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


@GateFactory.register(names=["Toffoli", "toffoli"])
class Toffoli(QubitGate):
    """Toffoli represents a Toffoli gate."""

    def __init__(self):
        super().__init__(name="Toffoli", num_wires=3)

    @lru_cache()
    def _data(self):
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


@GateFactory.register(names=["RX", "rx"])
class RX(QubitGate):
    """RX represents a single-qubit X rotation gate.

    Args:
        theta (float): Rotation angle around the X-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="RX", num_wires=1, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]
        c = cos(theta / 2)
        js = 1j * sin(-theta / 2)

        mat = [[c, js], [js, c]]
        return np.array(mat)


@GateFactory.register(names=["RY", "ry"])
class RY(QubitGate):
    """RY represents a single-qubit Y rotation gate.

    Args:
        theta (float): Rotation angle around the Y-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="RY", num_wires=1, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]

        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [[c, -s], [s, c]]
        return np.array(mat)


@GateFactory.register(names=["RZ", "rz"])
class RZ(QubitGate):
    """RZ represents a single-qubit Z rotation gate.

    Args:
        theta (float): Rotation angle around the Z-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="RZ", num_wires=1, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]
        p = exp(-0.5j * theta)

        mat = [[p, 0], [0, np.conj(p)]]
        return np.array(mat)


@GateFactory.register(names=["Rot", "rot"])
class Rot(QubitGate):
    """Rot represents an arbitrary single-qubit rotation gate with three Euler
    angles. Each Pauli rotation gate can be recovered by fixing two of the three
    parameters:

    >>> assert RX(theta).tensor() == Rot(pi/2, -pi/2, theta).tensor()
    >>> assert RY(theta).tensor() == Rot(0, 0, theta).tensor()
    >>> assert RZ(theta).tensor() == Rot(theta, 0, 0).tensor()

    See `qml.Rot
    <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.Rot.html>`__
    for more details.

    Args:
        phi (float): First rotation angle.
        theta (float): Second rotation angle.
        omega (float): Third rotation angle.
    """

    def __init__(self, phi: float, theta: float, omega: float):
        super().__init__(name="Rot", num_wires=1, params=[phi, theta, omega])

    @lru_cache()
    def _data(self):
        phi, theta, omega = self.params
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [
            [exp(-0.5j * (phi + omega)) * c, -exp(0.5j * (phi - omega)) * s],
            [exp(-0.5j * (phi - omega)) * s, exp(0.5j * (phi + omega)) * c],
        ]
        return np.array(mat)


@GateFactory.register(names=["CRX", "crx"])
class CRX(QubitGate):
    """CRX represents a controlled-RX gate.

    Args:
        theta (float): Rotation angle around the X-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="CRX", num_wires=2, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]
        c = cos(theta / 2)
        js = 1j * sin(-theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, js], [0, 0, js, c]]
        return np.array(mat)


@GateFactory.register(names=["CRY", "cry"])
class CRY(QubitGate):
    """CRY represents a controlled-RY gate.

    Args:
        theta (float): Rotation angle around the Y-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="CRY", num_wires=2, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c, -s], [0, 0, s, c]]
        return np.array(mat)


@GateFactory.register(names=["CRZ", "crz"])
class CRZ(QubitGate):
    """CRZ represents a controlled-RZ gate.

    Args:
        theta (float): Rotation angle around the Z-axis.
    """

    def __init__(self, theta: float):
        super().__init__(name="CRZ", num_wires=2, params=[theta])

    @lru_cache()
    def _data(self):
        theta = self.params[0]
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp(-0.5j * theta), 0],
            [0, 0, 0, exp(0.5j * theta)],
        ]
        return np.array(mat)


@GateFactory.register(names=["CRot", "crot"])
class CRot(QubitGate):
    """CRot represents a controlled-rotation gate.

    Args:
        phi (float): First rotation angle.
        theta (float): Second rotation angle.
        omega (float): Third rotation angle.
    """

    def __init__(self, phi: float, theta: float, omega: float):
        super().__init__(name="CRot", num_wires=2, params=[phi, theta, omega])

    @lru_cache()
    def _data(self):
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


@GateFactory.register(names=["U1", "u1"])
class U1(QubitGate):
    """U1 represents a U1 gate.

    Args:
        phi (float): Rotation angle.
    """

    def __init__(self, phi: float):
        super().__init__(name="U1", num_wires=1, params=[phi])

    @lru_cache()
    def _data(self):
        phi = self.params[0]
        mat = [[1, 0], [0, exp(1j * phi)]]
        return np.array(mat)


@GateFactory.register(names=["U2", "u2"])
class U2(QubitGate):
    """U2 represents a U2 gate.

    Args:
        phi (float): First rotation angle.
        lam (float): Second rotation angle.
    """

    def __init__(self, phi: float, lam: float):
        super().__init__(name="U2", num_wires=1, params=[phi, lam])

    @lru_cache()
    def _data(self):
        phi, lam = self.params
        mat = [
            [INV_SQRT2, -INV_SQRT2 * exp(1j * lam)],
            [INV_SQRT2 * exp(1j * phi), INV_SQRT2 * exp(1j * (phi + lam))],
        ]
        return np.array(mat)


@GateFactory.register(names=["U3", "u3"])
class U3(QubitGate):
    """U3 represents a U3 gate.

    Args:
        theta (float): First rotation angle.
        phi (float): Second rotation angle.
        lam (float): Third rotation angle.
    """

    def __init__(self, theta: float, phi: float, lam: float):
        super().__init__(name="U3", num_wires=1, params=[theta, phi, lam])

    @lru_cache()
    def _data(self):
        theta, phi, lam = self.params
        c = cos(theta / 2)
        s = sin(theta / 2)

        mat = [
            [c, -s * exp(1j * lam)],
            [s * exp(1j * phi), c * exp(1j * (phi + lam))],
        ]
        return np.array(mat)
