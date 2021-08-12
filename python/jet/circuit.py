"""Module containing the ``Operation``, ``Wire``, and ``Circuit`` classes."""
from dataclasses import dataclass
from typing import Iterator, Sequence, Union

import numpy as np

from .factory import TensorNetwork, TensorNetworkType
from .gate import Adjoint, Gate
from .state import Qudit, State

__all__ = [
    "Circuit",
    "Operation",
    "Wire",
]


@dataclass(frozen=True)
class Operation:
    """Operation represents the application of a gate or state to a ``Circuit``.

    Args:
        part (jet.Gate or jet.State): Gate or state applied to the circuit.
        wire_ids (Sequence[int]): ID(s) of the wires connected to the part.
    """

    part: Union[Gate, State]
    wire_ids: Sequence[int]


@dataclass
class Wire:
    """Wire represents a collection of tensor indices that are directly or
    transitively associated with a qudit of a quantum circuit.

    Args:
        id_ (int): Position of the wire in the circuit.
        depth (int): Number of gates applied to this wire.
        closed (bool): Whether this wire has been terminated with a state.
    """

    id_: int
    depth: int = 0
    closed: bool = False

    @property
    def index(self) -> str:
        """Returns the current index label of this wire."""
        return f"{self.id_}-{self.depth}"


class Circuit:
    """Circuit represents a quantum circuit composed of wires, each of which
    is intitialized with a qudit of the specified dimension in the vacuum state.

    Args:
        num_wires (int): Number of wires in the circuit.
        dim (int): Dimension of each wire.
    """

    def __init__(self, num_wires: int, dim: int = 2):
        self._dim = dim

        self._ops = []
        self._wires = []

        for i in range(num_wires):
            wire = Wire(i)
            self._wires.append(wire)

            state = Qudit(dim=dim)
            state.indices = [wire.index]

            op = Operation(part=state, wire_ids=[i])
            self._ops.append(op)

    @property
    def dimension(self) -> int:
        """Returns the dimension of this circuit."""
        return self._dim

    @property
    def operations(self) -> Iterator[Operation]:
        """Returns the gates and states that comprise this circuit alongside the
        wires they are connected to.  The first ``self.num_wires`` operations
        describe the qudits that begin each wire; other operations appear in the
        order they were appended to the circuit.
        """
        return iter(self._ops)

    @property
    def wires(self) -> Iterator[Wire]:
        """Returns the wires of this circuit in increasing order of wire ID."""
        return iter(self._wires)

    def append_gate(self, gate: Gate, wire_ids: Sequence[int]) -> None:
        """Applies a gate along the specified wires.

        Args:
            gate (jet.Gate): Gate to be applied.
            wire_ids (Sequence[int]): IDs of the wires the gate is applied to.
        """
        self._validate_wire_ids(wire_ids)

        if len(wire_ids) != gate.num_wires:
            raise ValueError(
                f"Number of wire IDs ({len(wire_ids)}) must match the number of "
                f"wires connected to the gate ({gate.num_wires})."
            )

        input_indices = list(self.indices(wire_ids))

        for i in wire_ids:
            self._wires[i].depth += 1

        output_indices = list(self.indices(wire_ids))

        gate.indices = output_indices + input_indices
        self._ops.append(Operation(part=gate, wire_ids=wire_ids))

    def append_state(self, state: State, wire_ids: Sequence[int]) -> None:
        """Terminates the specified wires with a quantum state.

        Args:
            state (jet.State): state to be used for termination.
            wire_ids (Sequence[int]): IDs of the wires the state terminates.
        """
        self._validate_wire_ids(wire_ids)

        if len(wire_ids) != state.num_wires:
            raise ValueError(
                f"Number of wire IDs ({len(wire_ids)}) must match the number of "
                f"wires connected to the state ({state.num_wires})."
            )

        for i in wire_ids:
            self._wires[i].closed = True

        state.indices = list(self.indices(wire_ids))
        self._ops.append(Operation(part=state, wire_ids=wire_ids))

    def take_expected_value(self, observable: Iterator[Operation]) -> None:
        """Completes this circuit by taking the expected value of an observable.

        Args:
            observable (Iterator[Operation]): Sequence of gate and wire ID pairs
                representing the observable.

        Note:
            This function finalizes the circuit; no more gates or states can be
            appended after this function is called.
        """
        # Compute the bounds of the slice containing the gates to be inverted.
        beg_index = len(self._wires)
        end_index = len(self._ops)

        for op in observable:
            self.append_gate(gate=op.part, wire_ids=op.wire_ids)

        # The adjoints are appended in reverse order for index continuity.
        for op in reversed(self._ops[beg_index:end_index]):
            gate = Adjoint(gate=op.part)
            self.append_gate(gate=gate, wire_ids=op.wire_ids)

        for op in reversed(self._ops[:beg_index]):
            # No adjoint is required: the initial qudits have real amplitudes.
            state = Qudit(dim=self.dimension)
            self.append_state(state=state, wire_ids=op.wire_ids)

    def indices(self, wire_ids: Iterator[int]) -> Iterator[str]:
        """Returns the index labels associated with a sequence of wire IDs.

        Args:
            wire_ids (Iterator[int]): IDs of the wires to get the index labels for.

        Returns:
            Iterator[str]: Current index label of each wire.
        """
        return (self._wires[i].index for i in wire_ids)

    def tensor_network(self, dtype: np.dtype = np.complex128) -> TensorNetworkType:
        """Returns the tensor network representation of this circuit.

        Args:
            dtype (np.dtype): Data type of the tensor network.

        Returns:
            TensorNetworkType: Tensor network representation of this circuit.
        """
        tn = TensorNetwork(dtype=dtype)
        for op in self._ops:
            tensor = op.part.tensor(dtype=dtype)
            tn.add_tensor(tensor)
        return tn

    def _validate_wire_ids(self, wire_ids: Sequence[int]) -> None:
        """Reports whether a set of wire IDs are valid.

        Args:
            wire_ids (Sequence[int]): Wire IDs to validate.

        Raises:
            ValueError: If at least one of the wire IDs is invalid.
        """
        num_wires = len(self._wires)

        for wire_id in wire_ids:
            if not 0 <= wire_id < num_wires:
                raise ValueError(f"Wire ID {wire_id} falls outside the range [0, {num_wires}).")

            if wire_ids.count(wire_id) > 1:
                raise ValueError(f"Wire ID {wire_id} is specified more than once.")

            if self._wires[wire_id].closed:
                raise ValueError(f"Wire {wire_id} is closed.")
