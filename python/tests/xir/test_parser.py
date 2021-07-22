"""Unit tests for the parser"""

import math
from decimal import Decimal

import pytest

from xir import DecimalComplex, parse_script


class TestParser:
    """Unit tests for the parser"""

    @pytest.mark.parametrize(
        "array, res",
        [
            ("[0, 1, 2]", [0, 1, 2]),
            ("[true, false]", [True, False]),
            ("[3+2, sin(3)]", [5, "sin(3)"]),
            ("[[0, 1], [[2], [3, 4]], [5, 6]]", [[0, 1], [[2], [3, 4]], [5, 6]]),
            ("[]", []),
        ],
    )
    def test_output_with_array(self, array, res):
        """Test outputs with arrays as parameter values"""
        circuit = f"an_output_statement(array: {array}) | [0, 1];"
        irprog = parse_script(circuit)

        assert irprog.statements[0].params["array"] == res

    @pytest.mark.parametrize(
        "key, val, expected",
        [
            ("cutoff", "5", 5),
            ("anything", "4.2", 4.2),
            ("a_number", "3 + 2.1", 5.1),
            ("a_number_with_pi", "pi / 2", math.pi / 2),
            ("a_string", "hello", "hello"),
            ("True", "False", "False"),
            ("true", "false", False),
        ],
    )
    def test_options(self, key, val, expected):
        """Test script-level options"""
        irprog = parse_script(f"options:\n    {key}: {val};\nend;", use_floats=True, eval_pi=True)

        assert key in irprog.options
        assert irprog.options[key] == expected

    @pytest.mark.parametrize(
        "key, val",
        [
            ("key", ["compound", "value"]),
            ("True", [1, 2, "False"]),
            ("key", [1, [2, [3, 4]]]),
        ],
    )
    def test_options_lists(self, key, val):
        """Test script-level options"""
        val_str = "[" + ", ".join(str(v) for v in val) + "]"
        irprog = parse_script(f"options:\n    {key}: {val_str};\nend;")

        assert key in irprog.options
        assert irprog.options[key] == val

    @pytest.mark.parametrize("script, adjoint", [
        ("adjoint ry(2.4) | [2];", True),
        ("ry(2.4) | [2];", False),
    ])
    def test_adjoint_modifier(self, script, adjoint):
        """Test that adjoint modifier for gate statments works correctly"""
        irprog = parse_script(script)

        assert irprog.statements[0].is_adjoint is adjoint

    @pytest.mark.parametrize("script, ctrl_wires", [
        ("ctrl[0, 1] ry(2.4) | [2];", (0, 1)),
        ("ry(2.4) | [2];", None),
    ])
    def test_ctrl_modifier(self, script, ctrl_wires):
        """Test that no ctrl modifier for gate statments works correctly"""
        irprog = parse_script(script)

        assert irprog.statements[0].ctrl_wires == ctrl_wires

    @pytest.mark.parametrize("use_floats", [True, False])
    @pytest.mark.parametrize("param", [3, 4.2, 2j])
    def test_use_floats(self, use_floats, param):
        """Test the ``use_floats`` kwarg to return float and complex types"""
        if use_floats:
            t = type(param)
        else:
            if isinstance(param, complex):
                t = DecimalComplex
            elif isinstance(param, float):
                t = Decimal
            elif isinstance(param, int):
                t = int

        circuit = f"a_gate({param}) | [0, 1];"
        irprog = parse_script(circuit, use_floats=use_floats)

        assert isinstance(irprog.statements[0].params[0], t)

    @pytest.mark.parametrize("eval_pi", [True, False])
    @pytest.mark.parametrize("param, expected", [("pi", math.pi), ("pi / 2", math.pi / 2)])
    def test_eval_pi(self, eval_pi, param, expected):
        """Test the ``eval_pi`` kwarg to evaluate mathematical expressions containing pi"""
        circuit = f"a_gate({param}) | [0, 1];"
        irprog = parse_script(circuit, eval_pi=eval_pi)
        if eval_pi:
            assert irprog.statements[0].params[0] == expected
        else:
            assert irprog.statements[0].params[0] == param.upper()
