from inspect import cleandoc
from math import sqrt

import pytest

import jet
import xir


@pytest.mark.parametrize(
    "script, want_result",
    [
        ("", []),
        (
            cleandoc(
                """
                use xstd;

                H | [0];
                """
            ),
            [],
        ),
    ],
)
def test_run_xir_program_with_no_output_statements(script, want_result):
    """Tests that running an XIR script with no output statementes gives the correct result."""
    tree = xir.xir_parser.parse(script)
    program = xir.XIRTransformer().transform(tree)

    have_result = jet.run_xir_program(program)
    assert have_result == pytest.approx(want_result)


@pytest.mark.parametrize(
    "script, want_result",
    [
        (
            cleandoc(
                """
                use xstd;

                X | [0];

                amplitude(state: 0) | [0];
                amplitude(state: 1) | [0];
                """
            ),
            [0, 1],
        ),
        (
            cleandoc(
                """
                use xstd;

                H | [0];
                CNOT | [0, 1];

                amplitude(state: 0) | [0, 1];
                amplitude(state: 1) | [0, 1];
                amplitude(state: 2) | [0, 1];
                amplitude(state: 3) | [0, 1];
                """
            ),
            [1 / sqrt(2), 0, 0, 1 / sqrt(2)],
        ),
        (
            cleandoc(
                """
                use xstd;

                TwoModeSqueezing(3, 1, 2) | [0, 1];

                amplitude(state: 0) | [0, 1];
                amplitude(state: 1) | [0, 1];
                amplitude(state: 2) | [0, 1];
                amplitude(state: 3) | [0, 1];
                """
            ),
            [0.0993279274194332, 0, 0, 0.053401711152745175 + 0.08316823745907517j],
        ),
    ],
)
def test_run_xir_program_with_amplitude_statements(script, want_result):
    """Tests that running an XIR script gives the correct result."""
    tree = xir.xir_parser.parse(script)
    program = xir.XIRTransformer().transform(tree)

    have_result = jet.run_xir_program(program)
    assert have_result == pytest.approx(want_result)
