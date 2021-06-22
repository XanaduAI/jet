"""Unit tests for the DecimalComplex class"""
from decimal import Decimal

import pytest

from xir import DecimalComplex


class TestDecimalComplex:
    """Tests for the DecimalComplex class"""

    @pytest.mark.parametrize(
        "re_1, im_1, re_2, im_2", [("1", "2", "3", "4"), ("0.2", "0.4", "8", "0.16")]
    )
    def test_addition(self, re_1, im_1, re_2, im_2):
        """Test the addition operator"""
        re_sum = Decimal(re_1) + Decimal(re_2)
        im_sum = Decimal(im_1) + Decimal(im_2)

        c_1 = DecimalComplex(re_1, im_1)
        c_2 = DecimalComplex(re_2, im_2)

        expected = DecimalComplex(re_sum, im_sum)

        assert c_1 + c_2 == expected
        assert (c_1 + c_2).real == re_sum
        assert (c_1 + c_2).imag == im_sum

    @pytest.mark.parametrize(
        "re_1, im_1, re_2, im_2", [("1", "2", "3", "4"), ("0.2", "0.4", "8", "0.16")]
    )
    def test_subtraction(self, re_1, im_1, re_2, im_2):
        """Test the subtraction operator"""
        re_sub = Decimal(re_1) - Decimal(re_2)
        im_sub = Decimal(im_1) - Decimal(im_2)

        c_1 = DecimalComplex(re_1, im_1)
        c_2 = DecimalComplex(re_2, im_2)

        expected = DecimalComplex(re_sub, im_sub)

        assert c_1 - c_2 == expected
        assert (c_1 - c_2).real == re_sub
        assert (c_1 - c_2).imag == im_sub

    @pytest.mark.parametrize(
        "re_1, im_1, re_2, im_2", [("1", "2", "3", "4"), ("0.2", "0.4", "8", "0.16")]
    )
    def test_multiplication(self, re_1, im_1, re_2, im_2):
        """Test the multiplication operator"""
        re_prod = Decimal(re_1) * Decimal(re_2) - Decimal(im_1) * Decimal(im_2)
        im_prod = Decimal(re_1) * Decimal(im_2) + Decimal(re_2) * Decimal(im_1)

        c_1 = DecimalComplex(re_1, im_1)
        c_2 = DecimalComplex(re_2, im_2)

        expected = DecimalComplex(re_prod, im_prod)

        assert c_1 * c_2 == expected
        assert (c_1 * c_2).real == re_prod
        assert (c_1 * c_2).imag == im_prod

    @pytest.mark.parametrize(
        "re_1, im_1, re_2, im_2", [("1", "2", "3", "4"), ("0.2", "0.4", "8", "0.16")]
    )
    def test_division(self, re_1, im_1, re_2, im_2):
        """Test the division operator"""
        re_div = (Decimal(re_1) * Decimal(re_2) + Decimal(im_1) * Decimal(im_2)) / (
            Decimal(re_2) ** 2 + Decimal(im_2) ** 2
        )
        im_div = (Decimal(im_1) * Decimal(re_2) - Decimal(re_1) * Decimal(im_2)) / (
            Decimal(re_2) ** 2 + Decimal(im_2) ** 2
        )

        c_1 = DecimalComplex(re_1, im_1)
        c_2 = DecimalComplex(re_2, im_2)

        expected = DecimalComplex(re_div, im_div)

        assert c_1 / c_2 == expected
        assert (c_1 / c_2).real == re_div
        assert (c_1 / c_2).imag == im_div

    @pytest.mark.parametrize("re, im", [("1", "2"), ("0.2", "0.4")])
    def test_absolute(self, re, im):
        """Test the abs operation"""
        c = DecimalComplex(re, im)
        expected = (Decimal(re) ** 2 + Decimal(im) ** 2).sqrt()

        assert abs(c) == expected

    @pytest.mark.parametrize("re, im", [("1", "2"), ("0.2", "0.4")])
    def test_negate(self, re, im):
        """Test negating a DecimalComplex object"""
        re_neg = -Decimal(re)
        im_neg = -Decimal(im)

        c = DecimalComplex(re, im)
        expected = DecimalComplex(re_neg, im_neg)

        assert -c == expected
        assert (-c).real == re_neg
        assert (-c).imag == im_neg

    @pytest.mark.parametrize("ob", [2, 1j, 3 + 4j, 0, -1])
    def test_power(self, ob):
        """Test the pow operation"""
        c = DecimalComplex("3", "2")

        c_res = (3 + 2j) ** ob
        re_pow = str(c_res.real)
        im_pow = str(c_res.imag)

        assert c ** ob == DecimalComplex(re_pow, im_pow)

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

    def test_cast_to_int(self):
        """Test casting a DecimalComplex object to int"""
        c = DecimalComplex("2", "3")

        match = r"Cannot convert DecimalComplex to int"
        with pytest.raises(TypeError, match=match):
            int(c)

    def test_cast_to_float(self):
        """Test casting a DecimalComplex object to float"""
        c = DecimalComplex("2", "3")

        match = r"Cannot convert DecimalComplex to float"
        with pytest.raises(TypeError, match=match):
            float(c)

    @pytest.mark.parametrize("c, expected", [
        (DecimalComplex("2", "3"), True),
        (DecimalComplex("5", "0"), True),
        (DecimalComplex("0", "5"), True),
        (DecimalComplex("2", "3"), True),
        (DecimalComplex("0", "0"), False),
        (DecimalComplex(0, 0), False),
    ])
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
        res = DecimalComplex._convert_type(n)

        assert isinstance(res, DecimalComplex)
        assert res == DecimalComplex(str(n))

    @pytest.mark.parametrize("n", [0.2 + 7j, 3j, DecimalComplex("4", "2")])
    def test_convert_type_with_imag(self, n):
        """Test the ``_convert_type`` method with valid complex arguments"""
        res = DecimalComplex._convert_type(n)

        assert isinstance(res, DecimalComplex)
        assert res == DecimalComplex(str(n.real), str(n.imag))

    @pytest.mark.parametrize("n", ["2", "string"])
    def test_convert_real_type(self, n):
        """Test the ``_convert_type`` method with invalid arguments"""
        match = r"Must be a Number or have attributes real and imag."
        with pytest.raises(TypeError, match=match):
            _ = DecimalComplex._convert_type(n)

    @pytest.mark.parametrize("re, im", [("1", "2"), ("0.2", "0.4")])
    def test_real_and_imag(self, re, im):
        """Test the ``real`` and ``imag`` properties"""
        c = DecimalComplex(re, im)

        assert c.real == Decimal(re)
        assert c.imag == Decimal(im)

    @pytest.mark.parametrize("re, im", [("1", "2"), ("0.2", "0.4"), ("0.2", "-0.4"), ("-0.2", "-0.4")])
    def test_conjugate(self, re, im):
        """Test the ``conjugate`` function"""
        c = DecimalComplex(re, im)

        assert c.conjugate() == DecimalComplex(re, -Decimal(im))
