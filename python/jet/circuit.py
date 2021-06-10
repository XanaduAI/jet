from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np

from .factory import TensorNetwork, TensorNetworkType
from .gate import Gate
from .state import Qudit, State

__all__ = [
    "Wire",
    "Circuit",
]


@dataclass
class Wire:
    """Wire represents a collection of tensor indices that are directly or
    transitively associated with a qudit of a quantum circuit.
    """

    # Position of the wire in the circuit.
    id_: int
    # Number of gates applied to this wire.
    depth: int = 0
    # Whether this wire has been terminated with a state.
    closed: int = False

    @property
    def index(self) -> str:
        """Returns the current index label of this wire."""
        return f"{self.id_}-{self.depth}"


class Circuit:
    def __init__(self, num_wires: int, dim: int = 2):
        """Constructs a quantum circuit. Each wire is initialized with a qudit
        of the specified dimension in the vacuum state.

        Args:
            num_wires (int): number of wires in the circuit.
            dim (int): dimension of each wire.
        """
        self._wires = [Wire(i) for i in range(num_wires)]
        self._parts = [Qudit(dim=dim) for _ in range(num_wires)]

        for (wire, part) in zip(self._wires, self._parts):
            part.indices = [wire.index]

    @property
    def parts(self) -> Tuple[Union[Gate, State]]:
        """Returns the gates and states that comprise this circuit.  The first
        ``self.num_wires`` parts are the qudits that begin each wire; other
        parts appear in the order they were appended to the circuit.
        """
        return tuple(self._parts)

    @property
    def wires(self) -> Tuple[Wire]:
        """Returns the wires of this circuit in increasing order of wire ID."""
        return tuple(self._wires)

    def _append_validator(append_fn: Callable) -> Callable:
        """Decorator which validates the arguments to an append function.

        Args:
            append_fn (Callable): append function to be validated.

        Returns:
            Function that validates the passed wire IDs before invoking the
            decorated append function with the given arguments.
        """

        @wraps(append_fn)
        def validator(self, part: Union[Gate, State], wire_ids: Sequence[int]) -> None:
            if not all(0 <= i < len(self.wires) for i in wire_ids):
                raise ValueError(f"Wire IDs must fall in the range [0, {len(self.wires)}).")

            elif any(self.wires[i].closed is True for i in wire_ids):
                raise ValueError(f"Wire IDs must correspond to open wires.")

            elif len(wire_ids) != len(set(wire_ids)):
                raise ValueError(f"Wire IDs must be unique.")

            elif len(wire_ids) != part.num_wires:
                raise ValueError(
                    f"Number of wire IDs ({len(wire_ids)}) must match the number of "
                    f"wires connected to the circuit component ({part.num_wires})."
                )

            append_fn(self, part, wire_ids)

        return validator

    @_append_validator
    def append_gate(self, gate: Gate, wire_ids: Sequence[int]) -> None:
        """Applies a gate along the specified wires.

        Args:
            gate (Gate): gate to be applied.
            wire_ids (Sequence[int]): IDs of the wires the gate is applied to.
        """
        input_indices = self.indices(wire_ids)

        for i in wire_ids:
            self.wires[i].depth += 1

        output_indices = self.indices(wire_ids)

        gate.indices = output_indices + input_indices
        self._parts.append(gate)

    @_append_validator
    def append_state(self, state: State, wire_ids: Sequence[int]) -> None:
        """Terminates the specified wires with a quantum state.

        Args:
            state (State): state to be used for termination.
            wire_ids (Sequence[int]): IDs of the wires the state terminates.
        """
        for i in wire_ids:
            self.wires[i].closed = True

        state.indices = self.indices(wire_ids)
        self._parts.append(state)

    def indices(self, wire_ids: Sequence[int]) -> List[str]:
        """Returns the index labels associated with a sequence of wire IDs.

        Args:
            wire_ids (Sequence[int]): IDs of the wires to get the index labels for.

        Returns:
            List of index labels.
        """
        return [self.wires[i].index for i in wire_ids]

    def tensor_network(self, dtype: type = np.complex128) -> TensorNetworkType:
        """Returns the tensor network representation of this circuit.

        Args:
            dtype (type): data type of the tensor network.
        """
        tn = TensorNetwork(dtype=dtype)
        for part in self._parts:
            tensor = part.tensor(dtype=dtype)
            tensor_id = tn.add_tensor(tensor)
            part.tensor_id = tensor_id
        return tn