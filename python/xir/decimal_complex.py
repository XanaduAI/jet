"""Module containing the DecimalComplex class which stores complex numbers with
``decimal.Decimal`` precision."""
from __future__ import annotations

import functools
from decimal import Decimal
from numbers import Complex
from typing import Callable, Union


def convert_input(func: Callable) -> Callable:
    """Converts inputs into valid ``DecimalComplex`` objects."""

    @functools.wraps(func)
    def convert_wrapper(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        if isinstance(c, DecimalComplex):
            return func(self, c)
        elif isinstance(c, Decimal):
            c = DecimalComplex(c)
            return func(self, c)
        elif isinstance(c, Complex):
            c = DecimalComplex(str(c.real), str(c.imag))
            return func(self, c)

        raise TypeError("Must be of type numbers.Complex.")

    return convert_wrapper


class DecimalComplex(Complex):
    """Complex numbers represented by precision ``decimal.Decimal`` terms.

    Args:
        real (str, Decimal): the real part of the complex number. Defaults to
            ``0.0`` if not input.
        imag (str, Decimal): The imaginary part of the complex number. Defaults
            to ``0.0`` if not input.
    """

    def __init__(
        self, real: Union[str, Decimal] = "0.0", imag: Union[str, Decimal] = "0.0"
    ) -> None:
        self._real = Decimal(real)
        self._imag = Decimal(imag)

    @convert_input
    def __add__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        real = self.real + c.real
        imag = self.imag + c.imag
        return DecimalComplex(real, imag)

    def __radd__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return self + c

    @convert_input
    def __sub__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        real = self.real - c.real
        imag = self.imag - c.imag
        return DecimalComplex(real, imag)

    def __rsub__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return -(self - c)

    @convert_input
    def __mul__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        real = self.real * c.real - self.imag * c.imag
        imag = self.imag * c.real + self.real * c.imag
        return DecimalComplex(real, imag)

    def __rmul__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return self * c

    @convert_input
    def __div__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        real = (self.real * c.real + self.imag * c.imag) / (c.real ** 2 + c.imag ** 2)
        imag = (self.imag * c.real - self.real * c.imag) / (c.real ** 2 + c.imag ** 2)
        return DecimalComplex(real, imag)

    @convert_input
    def __rdiv__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return c / self

    @convert_input
    def __truediv__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return self.__div__(c)

    @convert_input
    def __rtruediv__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        return c / self

    def __floordiv__(self, _: Union[Decimal, Complex]) -> DecimalComplex:
        raise TypeError("Cannot take floor of DecimalComplex")

    def __abs__(self) -> Decimal:
        return Decimal(self.real ** 2 + self.imag ** 2).sqrt()

    def __pos__(self) -> DecimalComplex:
        return self

    def __neg__(self) -> DecimalComplex:
        return DecimalComplex(-self.real, -self.imag)

    @convert_input
    def __eq__(self, c: DecimalComplex) -> bool:
        return self.real == c.real and self.imag == c.imag

    @convert_input
    def __ne__(self, c: DecimalComplex) -> bool:
        return not self.__eq__(c)

    def __str__(self) -> str:
        return f"{self.real}{self.imag:+}j"

    def __repr__(self):
        return f"DecimalComplex('{self.__str__()}')"

    def __pow__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        # TODO: calculate powers with precision using Decimal
        res = complex(self) ** complex(c)
        return DecimalComplex(str(res.real), str(res.imag))

    def __rpow__(self, c: Union[Decimal, Complex]) -> DecimalComplex:
        # TODO: calculate powers with precision using Decimal
        res = complex(c) ** complex(self)
        return DecimalComplex(str(res.real), str(res.imag))

    def __gt__(self, _: Union[Decimal, Complex]):
        self._not_implemented(">")

    def __lt__(self, _: Union[Decimal, Complex]):
        self._not_implemented("<")

    def __ge__(self, _: Union[Decimal, Complex]):
        self._not_implemented(">=")

    def __le__(self, _: Union[Decimal, Complex]):
        self._not_implemented("<=")

    def __bool__(self) -> bool:
        return self.real != 0 or self.imag != 0

    def __complex__(self) -> complex:
        return float(self.real) + float(self.imag) * 1j

    def _not_implemented(self, op):
        raise TypeError(f"Cannot use {op} with complex numbers")

    def __hash__(self):
        return hash((self.real, self.imag))

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
