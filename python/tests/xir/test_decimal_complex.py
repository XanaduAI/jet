"""Unit tests for the DecimalComplex class"""
from decimal import Decimal

import pytest

from xir import DecimalComplex
from xir.decimal_complex import convert_input


class TestDecimalComplex:
    """Tests for the DecimalComplex class"""

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (DecimalComplex("1", "2"), DecimalComplex("3", "4"), DecimalComplex("4", "6")),
            (DecimalComplex("3", "0.4"), DecimalComplex("8", "0.2"), DecimalComplex("11", "0.6")),
            (DecimalComplex("3", "0.4"), 3, DecimalComplex("6", "0.4")),
            (2, DecimalComplex("3", "0.4"), DecimalComplex("5", "0.4")),
            (DecimalComplex("3", "0.4"), 3j, DecimalComplex("3", "3.4")),
            (2j, DecimalComplex("3", "0.4"), DecimalComplex("3", "2.4")),
        ],
    )
    def test_addition(self, lhs, rhs, expected):
        """Test the addition operator"""
        result = lhs + rhs
        assert result == expected

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (DecimalComplex("1", "2"), DecimalComplex("3", "4"), DecimalComplex("-2", "-2")),
            (DecimalComplex("3", "0.4"), DecimalComplex("8", "0.2"), DecimalComplex("-5", "0.2")),
            (DecimalComplex("3", "0.4"), 3, DecimalComplex("0", "0.4")),
            (2, DecimalComplex("3", "0.4"), DecimalComplex("-1", "-0.4")),
            (DecimalComplex("3", "0.4"), 3j, DecimalComplex("3", "-2.6")),
            (2j, DecimalComplex("3", "0.4"), DecimalComplex("-3", "1.6")),
        ],
    )
    def test_subtraction(self, lhs, rhs, expected):
        """Test the subtraction operator"""
        result = lhs - rhs
        assert result == expected

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (DecimalComplex("1", "2"), DecimalComplex("3", "4"), DecimalComplex("-5", "10")),
            (
                DecimalComplex("3", "0.4"),
                DecimalComplex("8", "0.2"),
                DecimalComplex("23.92", "3.8"),
            ),
            (DecimalComplex("3", "0.4"), 3, DecimalComplex("9", "1.2")),
            (2, DecimalComplex("3", "0.4"), DecimalComplex("6", "0.8")),
            (DecimalComplex("3", "0.4"), 3j, DecimalComplex("-1.2", "9")),
            (2j, DecimalComplex("3", "0.4"), DecimalComplex("-0.8", "6")),
        ],
    )
    def test_multiplication(self, lhs, rhs, expected):
        """Test the multiplication operator"""
        result = lhs * rhs
        assert result == expected

    @pytest.mark.parametrize(
        "lhs, rhs, expected",
        [
            (DecimalComplex("1", "2"), DecimalComplex("3", "4"), DecimalComplex("0.44", "0.08")),
            (DecimalComplex("2", "4"), DecimalComplex("1", "2"), DecimalComplex("2", "0")),
            (
                DecimalComplex("2.5", "4.2"),
                DecimalComplex("1.3", "0.2"),
                DecimalComplex("2.3641618497", "2.8670520231"),
            ),
            (DecimalComplex("2", "4"), 2, DecimalComplex("1", "2")),
            (2, DecimalComplex("2", "4"), DecimalComplex("0.2", "-0.4")),
            (DecimalComplex("2", "4"), 3j, DecimalComplex("1.3333333333", "-0.6666666666")),
            (2.2j, DecimalComplex("2", "4"), DecimalComplex("0.44", "0.22")),
        ],
    )
    def test_division(self, lhs, rhs, expected):
        """Test the division operator"""
        result = lhs / rhs
        assert result.real == pytest.approx(expected.real)
        assert result.imag == pytest.approx(expected.imag)

    @pytest.mark.parametrize(
        "c, expected",
        [
            (DecimalComplex("3", "-4"), Decimal("5")),
            (DecimalComplex("6", "8"), Decimal("10")),
            (DecimalComplex("2.5", "4.2"), Decimal("4.8877397639")),
        ],
    )
    def test_absolute(self, c, expected):
        """Test the abs operation"""
        result = abs(c)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "c, expected",
        [
            (DecimalComplex("1", "2"), DecimalComplex("-1", "-2")),
            (DecimalComplex("3", "0.4"), DecimalComplex("-3", "-0.4")),
        ],
    )
    def test_negate(self, c, expected):
        """Test negating a DecimalComplex object"""
        result = -c
        assert result == expected

    @pytest.mark.parametrize(
        "c, po, expected",
        [
            (DecimalComplex("3", "2"), 2, DecimalComplex(5, 12)),
            (DecimalComplex("3", "2"), 1j, DecimalComplex(0.1579345325, 0.5325085840)),
            (DecimalComplex("3", "2"), 3 + 4j, DecimalComplex(3.6547543594, 2.5583022069)),
            (DecimalComplex("3", "2"), 0, DecimalComplex(1, 0)),
            (DecimalComplex("3", "2"), -1, DecimalComplex(0.2307692308, -0.1538461538)),
        ],
    )
    def test_power(self, c, po, expected):
        """Test the pow operation"""
        res = c ** po
        assert res.real == pytest.approx(expected.real)
        assert res.imag == pytest.approx(expected.imag)

    @pytest.mark.parametrize("term", [2, 3j, "2", Decimal("1"), -4.3])
    def test_greater_than(self, term):
        """Test the > operator"""
        c = DecimalComplex("2", "3")

        match = r"Cannot use > with complex numbers"
        with pytest.raises(TypeError, match=match):
            c > term

        with pytest.raises(TypeError, match=match):
            term < c

    @pytest.mark.parametrize("term", [2, 3j, "2", Decimal("1"), -4.3])
    def test_less_than(self, term):
        """Test the < operator"""
        c = DecimalComplex("2", "3")

        match = r"Cannot use < with complex numbers"
        with pytest.raises(TypeError, match=match):
            c < term

        with pytest.raises(TypeError, match=match):
            term > c

    @pytest.mark.parametrize("term", [2, 3j, "2", Decimal("1"), -4.3])
    def test_greater_equal_than(self, term):
        """Test the >= operator"""
        c = DecimalComplex("2", "3")

        match = r"Cannot use >= with complex numbers"
        with pytest.raises(TypeError, match=match):
            c >= term

        with pytest.raises(TypeError, match=match):
            term <= c

    @pytest.mark.parametrize("term", [2, 3j, "2", Decimal("1"), -4.3])
    def test_less_equal_than(self, term):
        """Test the <= operator"""
        c = DecimalComplex("2", "3")

        match = r"Cannot use <= with complex numbers"
        with pytest.raises(TypeError, match=match):
            c <= term

        with pytest.raises(TypeError, match=match):
            term >= c

    @pytest.mark.parametrize(
        "t, match",
        [
            (set, r"object is not iterable"),
            (list, r"object is not iterable"),
            (dict, r"object is not iterable"),
            (tuple, r"object is not iterable"),
            (frozenset, r"object is not iterable"),
            (float, r"argument must be a string or a number"),
            (range, r"object cannot be interpreted as an integer"),
            (bytes, r"cannot convert 'DecimalComplex' object to bytes"),
            (bytearray, r"cannot convert 'DecimalComplex' object to bytearray"),
            (int, r"argument must be a string, a bytes-like object or a number"),
            (memoryview, r"a bytes-like object is required"),
        ],
    )
    def test_invalid_typecasts(self, t, match):
        """Test casting a DecimalComplex to unsupported types"""
        c = DecimalComplex("2", "3")

        with pytest.raises(TypeError, match=match):
            t(c)

    @pytest.mark.parametrize(
        "c, expected",
        [
            (DecimalComplex("2", "3"), True),
            (DecimalComplex("5", "0"), True),
            (DecimalComplex("0", "5"), True),
            (DecimalComplex("2", "3"), True),
            (DecimalComplex("0", "0"), False),
            (DecimalComplex(0, 0), False),
        ],
    )
    def test_cast_to_bool(self, c, expected):
        """Test casting a DecimalComplex object to float"""
        assert bool(c) == expected

    def test_cast_to_complex(self):
        """Test casting a DecimalComplex object to complex"""
        c = DecimalComplex("2", "3")
        expected = 2 + 3j

        assert complex(c) == expected

    @pytest.mark.parametrize("n", [2, Decimal("1"), -4.3])
    def test_convert_type(self, n):
        """Test the ``_convert_type`` method with valid arguments"""
        res = convert_input(lambda _, c: c)(None, n)

        assert isinstance(res, DecimalComplex)
        assert res == DecimalComplex(str(n))

    @pytest.mark.parametrize("n", [0.2 + 7j, 3j, DecimalComplex("4", "2")])
    def test_convert_type_with_imag(self, n):
        """Test the ``_convert_type`` method with valid complex arguments"""
        res = convert_input(lambda _, c: c)(None, n)

        assert isinstance(res, DecimalComplex)
        assert res == DecimalComplex(str(n.real), str(n.imag))

    @pytest.mark.parametrize("n", ["2", "string"])
    def test_convert_real_type(self, n):
        """Test the ``_convert_type`` method with invalid arguments"""
        match = r"Must be of type numbers.Complex."
        with pytest.raises(TypeError, match=match):
            convert_input(lambda _, c: c)(None, n)

    @pytest.mark.parametrize("re, im", [("1", "2"), ("0.2", "0.4")])
    def test_real_and_imag(self, re, im):
        """Test the ``real`` and ``imag`` properties"""
        c = DecimalComplex(re, im)

        assert c.real == Decimal(re)
        assert c.imag == Decimal(im)

    @pytest.mark.parametrize(
        "re, im", [("1", "2"), ("0.2", "0.4"), ("0.2", "-0.4"), ("-0.2", "-0.4")]
    )
    def test_conjugate(self, re, im):
        """Test the ``conjugate`` function"""
        c = DecimalComplex(re, im)

        assert c.conjugate() == DecimalComplex(re, -Decimal(im))
