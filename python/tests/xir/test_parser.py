"""Unit tests for the parser"""

import textwrap

import pytest

from xir import XIRTransformer, xir_parser
from xir.program import XIRProgram


def parse_script(circuit: str) -> XIRProgram:
    """Parse and transform a circuit XIR script and return an XIRProgram"""
    tree = xir_parser.parse(circuit)
    return XIRTransformer().transform(tree)


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
        "key, val",
        [
            ("cutoff", 5),
            # ("anything", 4.2),
            ("a_string", "hello"),
            ("True", "False"),
            # ("", "value"),
            # ("key", "")
        ],
    )
    def test_options(self, key, val):
        """Test script-level options"""
        irprog = parse_script(f"{key}: {val}")

        assert key in irprog.options
        assert irprog.options[key] == val

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
        print(val_str)
        irprog = parse_script(f"{key}: {val_str}")

        assert key in irprog.options
        assert irprog.options[key] == val
