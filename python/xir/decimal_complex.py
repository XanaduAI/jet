"""Module containing the DecimalComplex class which stores complex numbers with
``decimal.Decimal`` precision."""
from __future__ import annotations

from decimal import Decimal
from numbers import Number
from typing import Union


class DecimalComplex:
    """Complex numbers represented by precision ``decimal.Decimal`` terms.

    Args:
        real (str, Decimal): the real part of the complex number. Defaults to
            ``0.0`` if not input.
        imag (str, Decimal): The imaginary part of the complex number. Defaults
            to ``0.0`` if not input.
    """

    def __init__(self, real: Union[str, Decimal] = "0.0", imag: Union[str, Decimal] = "0.0") -> None:
        self._real = Decimal(real)
        self._imag = Decimal(imag)

    def __add__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)

        real = self.real + c.real
        imag = self.imag + c.imag
        return DecimalComplex(real, imag)

    def __radd__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        return self.__add__(n)

    def __sub__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)

        real = self.real - c.real
        imag = self.imag - c.imag
        return DecimalComplex(real, imag)

    def __rsub__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)
        return c.__sub__(self)

    def __mul__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)

        real = self.real * c.real - self.imag * c.imag
        imag = self.imag * c.real + self.real * c.imag
        return DecimalComplex(real, imag)

    def __rmul__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        return self.__mul__(n)

    def __div__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)

        real = (self.real * c.real + self.imag * c.imag) / (c.real ** 2 + c.imag ** 2)
        imag = (self.imag * c.real - self.real * c.imag) / (c.real ** 2 + c.imag ** 2)
        return DecimalComplex(real, imag)

    def __rdiv__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)
        return c.__div__(self)

    def __truediv__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        return self.__div__(n)

    def __rtruediv__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        c = self._convert_type(n)
        return c.__truediv__(self)

    def __abs__(self) -> Decimal:
        return Decimal(self.real ** 2 + self.imag ** 2).sqrt()

    def __neg__(self) -> DecimalComplex:
        return DecimalComplex(-self.real, -self.imag)

    def __eq__(self, c: DecimalComplex) -> bool:
        return self.real == c.real and self.imag == c.imag

    def __ne__(self, c: DecimalComplex) -> bool:
        return not self.__eq__(c)

    def __str__(self) -> str:
        if self.imag >= 0:
            return f"{self.real}+{self.imag}j"
        else:
            return f"{self.real}{self.imag}j"

    def __repr__(self):
        return f"DecimalComplex('{self.__str__()}')"

    def __pow__(self, n: Union[Number, DecimalComplex]) -> DecimalComplex:
        res = complex(self) ** complex(n)
        return DecimalComplex(str(res.real), str(res.imag))

    def __gt__(self, n: Union[Number, DecimalComplex]):
        self._not_implemented(">")

    def __lt__(self, n: Union[Number, DecimalComplex]):
        self._not_implemented("<")

    def __ge__(self, n: Union[Number, DecimalComplex]):
        self._not_implemented(">=")

    def __le__(self, n: Union[Number, DecimalComplex]):
        self._not_implemented("<=")

    def __int__(self) -> int:
        raise TypeError("Cannot convert DecimalComplex to int")

    def __float__(self) -> float:
        raise TypeError("Cannot convert DecimalComplex to float")

    def __bool__(self) -> bool:
        return self.real != 0 or self.imag != 0

    def __complex__(self) -> complex:
        return float(self.real) + float(self.imag) * 1j

    @staticmethod
    def _convert_type(n: Union[Number, DecimalComplex]) -> DecimalComplex:
        """Converts number into valid ``DecimalComplex`` object."""
        if isinstance(n, DecimalComplex):
            return n
        elif hasattr(n, "real") and hasattr(n, "imag"):
            return DecimalComplex(str(n.real), str(n.imag))
        elif isinstance(n, Number):
            return DecimalComplex(Decimal(str(n)))

        raise TypeError("Must be a Number or have attributes real and imag.")

    def _not_implemented(self, op):
        raise TypeError(f"Cannot use {op} with complex numbers")

    @property
    def real(self):
        """Return the real part of the complex number."""
        return self._real

    @property
    def imag(self):
        """Return the imaginary part of the complex number."""
        return self._imag

    def conjugate(self):
        """Return the complex conjugate of its argument."""
        return DecimalComplex(self.real, -self.imag)
