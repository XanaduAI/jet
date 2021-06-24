from inspect import cleandoc
from math import sqrt

import pytest

import jet
import xir


def parse_xir_script(script: str) -> xir.XIRProgram:
    """Parses an XIR script into an XIR program."""
    tree = xir.xir_parser.parse(script)
    return xir.XIRTransformer().transform(tree)


class TestGetXIRProgram:
    @pytest.mark.parametrize(
        "registry, want_xir_script",
        [
            pytest.param(
                {},
                "",
                id="Empty",
            ),
            pytest.param(
                {
                    "x": jet.PauliX,
                    "y": jet.PauliY,
                    "z": jet.PauliZ,
                },
                cleandoc(
                    """
                    gate x, 0, 1;
                    gate y, 0, 1;
                    gate z, 0, 1;
                    """
                ),
                id="Ordering",
            ),
            pytest.param(
                {
                    "u1": jet.U1,
                    "u2": jet.U2,
                    "u3": jet.U3,
                },
                cleandoc(
                    """
                    gate u1, 1, 1;
                    gate u2, 2, 1;
                    gate u3, 3, 1;
                    """
                ),
                id="Parameters",
            ),
            pytest.param(
                {
                    "swap": jet.SWAP,
                    "cswap": jet.CSWAP,
                },
                cleandoc(
                    """
                    gate cswap, 0, 3;
                    gate swap, 0, 2;
                    """
                ),
                id="Wires",
            ),
            pytest.param(
                {
                    "h": jet.Hadamard,
                    "hadamard": jet.Hadamard,
                },
                cleandoc(
                    """
                    gate h, 0, 1;
                    gate hadamard, 0, 1;
                    """
                ),
                id="Duplicate",
            ),
        ],
    )
    def test_fake_registry(self, monkeypatch, registry, want_xir_script):
        """Tests that the correct XIRProgram is returned for the fake gate registry."""
        monkeypatch.setattr("jet.GateFactory.registry", registry)
        have_xir_script = jet.get_xir_program().serialize()
        assert have_xir_script == want_xir_script

    def test_real_registry(self):
        """Tests that the correct XIRProgram is returned for the real gate registry."""
        assert jet.get_xir_program().serialize() == cleandoc(
            """
            gate beamsplitter, 3, 2;
            gate bs, 3, 2;
            gate cnot, 0, 2;
            gate cphaseshift, 1, 2;
            gate crot, 3, 2;
            gate crx, 1, 2;
            gate cry, 1, 2;
            gate crz, 1, 2;
            gate cswap, 0, 3;
            gate cx, 0, 2;
            gate cy, 0, 2;
            gate cz, 0, 2;
            gate d, 3, 1;
            gate displacement, 3, 1;
            gate h, 0, 1;
            gate hadamard, 0, 1;
            gate iswap, 0, 2;
            gate not, 0, 1;
            gate paulix, 0, 1;
            gate pauliy, 0, 1;
            gate pauliz, 0, 1;
            gate phaseshift, 1, 1;
            gate rot, 3, 1;
            gate rx, 1, 1;
            gate ry, 1, 1;
            gate rz, 1, 1;
            gate s, 0, 1;
            gate squeezing, 3, 1;
            gate swap, 0, 2;
            gate sx, 0, 1;
            gate t, 0, 1;
            gate toffoli, 0, 3;
            gate twomodesqueezing, 3, 2;
            gate u1, 1, 1;
            gate u2, 2, 1;
            gate u3, 3, 1;
            gate x, 0, 1;
            gate y, 0, 1;
            gate z, 0, 1;
            """
        )


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


@pytest.mark.parametrize(
    "program, match",
    [
        (
            parse_xir_script("X | [0]; amplitude | [0];"),
            r"Statement 'amplitude \| \[0\]' is missing a 'state' parameter\.",
        ),
        (
            parse_xir_script("X | [0]; amplitude(state: 2) | [0];"),
            r"Statement 'amplitude\(state: 2\) \| \[0\]' has a 'state' parameter which is too large\.",
        ),
        (
            parse_xir_script("CNOT | [0, 1]; amplitude(state: 0) | [0];"),
            r"Statement 'amplitude\(state: 0\) \| \[0\]' must be applied to \[0 \.\. 1\]\.",
        ),
        (
            parse_xir_script("CNOT | [0, 1]; amplitude(state: 0) | [1, 0];"),
            r"Statement 'amplitude\(state: 0\) \| \[1, 0\]' must be applied to \[0 \.\. 1\]\.",
        ),
    ],
)
def test_run_xir_program_with_invalid_amplitude_statement(program, match):
    """Tests that a ValueError is raised when an XIR program contains an
    invalid amplitude statement.
    """
    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


def test_run_xir_program_with_unsupported_statement():
    """Tests that a ValueError is raised when an XIR program contains an
    unsupported statement.
    """
    program = parse_xir_script("use xstd; halt | [0];")

    with pytest.raises(ValueError, match=r"Statement 'halt \| \[0\]' is not supported\."):
        jet.run_xir_program(program)
