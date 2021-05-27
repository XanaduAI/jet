"""Tensor representations of quantum gates"""

import numpy as np

import jet

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Gate:
    """Gate class"""

    def __init__(self, label: str) -> None:
        self.label = label

    def tensor(self, adjoint: bool = False):
        """Tensor representation of gate"""
        raise NotImplementedError

    def _data(self) -> np.ndarray:
        """Matrix representation of the gate"""
        raise NotImplementedError


class Squeezing(Gate):
    """Squeezing gate"""

    def __init__(self, label: str, r: float, theta: float, cutoff: int) -> None:
        self.r = r
        self.theta = theta
        self.cutoff = cutoff

        super().__init__(label)

    def tensor(self, adjoint: bool = False) -> jet.Tensor:
        """Tensor representation of gate"""
        indices = list(ALPHABET[:self.cutoff])
        shape = [self.cutoff, self.cutoff]

        if adjoint:
            data = np.conj(self._data()).T.flatten()
        else:
            data = self._data().flatten()

        return jet.Tensor(indices, shape, data)

    def _data(self) -> np.ndarray:
        """Calculates the matrix elements of the squeezing gate using a recurrence relation.

        Args:
            r (float): squeezing magnitude
            theta (float): squeezing angle
            cutoff (int): Fock ladder cutoff
            dtype (data type): Specifies the data type used for the calculation

        Returns:
            array[complex]: matrix representing the squeezing gate.
        """
        S = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
        sqrt = np.sqrt(np.arange(self.cutoff, dtype=np.complex128))

        eitheta_tanhr = np.exp(1j * self.theta) * np.tanh(self.r)
        sechr = 1.0 / np.cosh(self.r)
        R = np.array([[-eitheta_tanhr, sechr], [sechr, np.conj(eitheta_tanhr)],])

        S[0, 0] = np.sqrt(sechr)
        for m in range(2, self.cutoff, 2):
            S[m, 0] = sqrt[m - 1] / sqrt[m] * R[0, 0] * S[m - 2, 0]

        for m in range(0, self.cutoff):
            for n in range(1, self.cutoff):
                if (m + n) % 2 == 0:
                    S[m, n] = sqrt[n - 1] / sqrt[n] * R[1, 1] * S[m, n - 2] + sqrt[m] / sqrt[n] * R[0, 1] * S[m - 1, n - 1]
        return S
