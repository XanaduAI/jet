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
        circuit = f"an_output_statement(array: {array}) | [0, 1];"
        irprog = parse_script(circuit)

        assert irprog.statements[0].params["array"] == res
