"""Contains the circuit class for building a tensor network"""

import jet
from .gates import Gate


class Circuit:
    """Circuit class"""

    def __init__(self) -> None:
        self.circuit = []

    def tensor_network(self) -> jet.TensorNetwork:
        """Builds and returns the tensor network corresponding to the circuit"""
        tn = jet.TensorNetwork()
        return tn

    def add_gate(self, gate: Gate) -> None:
        """Adds a gate to the circuit"""
