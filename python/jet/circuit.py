from dataclasses import dataclass
from functools import wraps
from typing import Callable, Iterator, List, Sequence, Tuple, Union

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
        num_wires (int): number of wires in the circuit.
        dim (int): dimension of each wire.
    """

    def __init__(self, num_wires: int, dim: int = 2):
        self._dim = dim
        self._wires = [Wire(i) for i in range(num_wires)]
        self._parts = [Qudit(dim=dim) for _ in range(num_wires)]

        for (wire, part) in zip(self._wires, self._parts):
            part.indices = [wire.index]

    @property
    def dimension(self) -> int:
        """Returns the dimension of this circuit."""
        return self._dim

    @property
    def parts(self) -> Iterator[Union[Gate, State]]:
        """Returns the gates and states that comprise this circuit.  The first
        ``self.num_wires`` parts are the qudits that begin each wire; other
        parts appear in the order they were appended to the circuit.
        """
        return iter(self._parts)

    @property
    def wires(self) -> Iterator[Wire]:
        """Returns the wires of this circuit in increasing order of wire ID."""
        return iter(self._wires)

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
            if len(wire_ids) != part.num_wires:
                raise ValueError(
                    f"Number of wire IDs ({len(wire_ids)}) must match the number of "
                    f"wires connected to the circuit component ({part.num_wires})."
                )

            num_wires = len(self._wires)

            for wire_id in wire_ids:
                if not 0 <= wire_id < num_wires:
                    raise ValueError(f"Wire ID {wire_id} falls outside the range [0, {num_wires}).")

                elif wire_ids.count(wire_id) > 1:
                    raise ValueError(f"Wire ID {wire_id} is specified more than once.")

                elif self._wires[wire_id].closed:
                    raise ValueError(f"Wire {wire_id} is closed.")

            append_fn(self, part, wire_ids)

        return validator

    @_append_validator
    def append_gate(self, gate: Gate, wire_ids: Sequence[int]) -> None:
        """Applies a gate along the specified wires.

        Args:
            gate (jet.Gate): gate to be applied.
            wire_ids (Sequence[int]): IDs of the wires the gate is applied to.
        """
        input_indices = self.indices(wire_ids)

        for i in wire_ids:
            self._wires[i].depth += 1

        output_indices = self.indices(wire_ids)

        gate.indices = output_indices + input_indices
        self._parts.append(gate)

    @_append_validator
    def append_state(self, state: State, wire_ids: Sequence[int]) -> None:
        """Terminates the specified wires with a quantum state.

        Args:
            state (jet.State): state to be used for termination.
            wire_ids (Sequence[int]): IDs of the wires the state terminates.
        """
        for i in wire_ids:
            self._wires[i].closed = True

        state.indices = self.indices(wire_ids)
        self._parts.append(state)

    def indices(self, wire_ids: Sequence[int]) -> List[str]:
        """Returns the index labels associated with a sequence of wire IDs.

        Args:
            wire_ids (Sequence[int]): IDs of the wires to get the index labels for.

        Returns:
            List of index labels.
        """
        return [self._wires[i].index for i in wire_ids]

    def tensor_network(self, dtype: type = np.complex128) -> TensorNetworkType:
        """Returns the tensor network representation of this circuit.

        Args:
            dtype (type): data type of the tensor network.
        """
        tn = TensorNetwork(dtype=dtype)
        for part in self._parts:
            tensor = part.tensor(dtype=dtype)
            tn.add_tensor(tensor)
        return tn
