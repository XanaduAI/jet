from math import sqrt

import pytest

import jet
import xir


def parse_xir_script(script: str) -> xir.XIRProgram:
    """Parses an XIR script into an XIR program."""
    tree = xir.xir_parser.parse(script)
    return xir.XIRTransformer().transform(tree)


@pytest.mark.parametrize(
    "program",
    [
        parse_xir_script(""),
        parse_xir_script("use xstd;"),
        parse_xir_script("use xstd; H | [0];"),
    ],
)
def test_run_xir_program_with_no_output_statements(program):
    """Tests that running an XIR program with no output statements returns an empty list."""
    assert jet.run_xir_program(program) == []


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            parse_xir_script(
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
            parse_xir_script(
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
            parse_xir_script(
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
def test_run_xir_program_with_amplitude_statements(program, want_result):
    """Tests that running an XIR program with amplitude statements gives the correct result."""
    assert jet.run_xir_program(program) == pytest.approx(want_result)


def test_run_xir_program_with_stateless_amplitude_statement():
    """Tests that a ValueError is raised when an XIR program contains an
    amplitude statement that is missing a "state" parameter.
    """
    program = parse_xir_script("use xstd; X | [0]; amplitude | [0];")

    with pytest.raises(
        ValueError, match=r"Statement 'amplitude \| \[0\]' is missing a 'state' parameter\."
    ):
        jet.run_xir_program(program)


def test_run_xir_program_with_unsupported_statement():
    """Tests that a ValueError is raised when an XIR program contains an
    unsupported statement.
    """
    program = parse_xir_script("use xstd; halt | [0];")

    with pytest.raises(ValueError, match=r"Statement 'halt \| \[0\]' is not supported\."):
        jet.run_xir_program(program)
