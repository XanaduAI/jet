from inspect import cleandoc
from math import sqrt

import pytest

import jet
import xir


class TestGetXIRLibrary:
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
                    "X": jet.PauliX,
                    "Y": jet.PauliY,
                    "Z": jet.PauliZ,
                },
                cleandoc(
                    """
                    gate X[0]:
                        X | [0];
                    end;

                    gate Y[0]:
                        Y | [0];
                    end;

                    gate Z[0]:
                        Z | [0];
                    end;
                    """
                ),
                id="Ordering",
            ),
            pytest.param(
                {
                    "U1": jet.U1,
                    "U2": jet.U2,
                    "U3": jet.U3,
                },
                cleandoc(
                    """
                    gate U1(phi)[0]:
                        U1(phi) | [0];
                    end;

                    gate U2(phi, lam)[0]:
                        U2(phi, lam) | [0];
                    end;

                    gate U3(theta, phi, lam)[0]:
                        U3(theta, phi, lam) | [0];
                    end;
                    """
                ),
                id="Parameters",
            ),
            pytest.param(
                {
                    "SWAP": jet.SWAP,
                    "CSWAP": jet.CSWAP,
                },
                cleandoc(
                    """
                    gate CSWAP[0, 1, 2]:
                        CSWAP | [0, 1, 2];
                    end;

                    gate SWAP[0, 1]:
                        SWAP | [0, 1];
                    end;
                    """
                ),
                id="Wires",
            ),
            pytest.param(
                {
                    "H": jet.Hadamard,
                    "Hadamard": jet.Hadamard,
                },
                cleandoc(
                    """
                    gate H[0]:
                        H | [0];
                    end;

                    gate Hadamard[0]:
                        Hadamard | [0];
                    end;
                    """
                ),
                id="Duplicate",
            ),
        ],
    )
    def test_fake_registry(self, monkeypatch, registry, want_xir_script):
        """Tests that the correct XIRProgram is returned for the fake gate registry."""
        monkeypatch.setattr("jet.GateFactory.registry", registry)
        have_xir_script = jet.get_xir_library().serialize()
        assert have_xir_script == want_xir_script

    def test_real_registry(self):
        """Tests that the correct XIRProgram is returned for the real gate registry."""
        have_xir_program = jet.get_xir_library().serialize(minimize=True)
        want_xir_program = (
            "gate BS(theta, phi, cutoff)[0, 1]: BS(theta, phi, cutoff) | [0, 1]; end; "
            "gate Beamsplitter(theta, phi, cutoff)[0, 1]: Beamsplitter(theta, phi, cutoff) | [0, 1]; end; "
            "gate CNOT[0, 1]: CNOT | [0, 1]; end; "
            "gate CPhaseShift(phi)[0, 1]: CPhaseShift(phi) | [0, 1]; end; "
            "gate CRX(theta)[0, 1]: CRX(theta) | [0, 1]; end; "
            "gate CRY(theta)[0, 1]: CRY(theta) | [0, 1]; end; "
            "gate CRZ(theta)[0, 1]: CRZ(theta) | [0, 1]; end; "
            "gate CRot(phi, theta, omega)[0, 1]: CRot(phi, theta, omega) | [0, 1]; end; "
            "gate CSWAP[0, 1, 2]: CSWAP | [0, 1, 2]; end; "
            "gate CX[0, 1]: CX | [0, 1]; end; "
            "gate CY[0, 1]: CY | [0, 1]; end; "
            "gate CZ[0, 1]: CZ | [0, 1]; end; "
            "gate D(r, phi, cutoff)[0]: D(r, phi, cutoff) | [0]; end; "
            "gate Displacement(r, phi, cutoff)[0]: Displacement(r, phi, cutoff) | [0]; end; "
            "gate H[0]: H | [0]; end; "
            "gate Hadamard[0]: Hadamard | [0]; end; "
            "gate ISWAP[0, 1]: ISWAP | [0, 1]; end; "
            "gate NOT[0]: NOT | [0]; end; "
            "gate PauliX[0]: PauliX | [0]; end; "
            "gate PauliY[0]: PauliY | [0]; end; "
            "gate PauliZ[0]: PauliZ | [0]; end; "
            "gate PhaseShift(phi)[0]: PhaseShift(phi) | [0]; end; "
            "gate RX(theta)[0]: RX(theta) | [0]; end; "
            "gate RY(theta)[0]: RY(theta) | [0]; end; "
            "gate RZ(theta)[0]: RZ(theta) | [0]; end; "
            "gate Rot(phi, theta, omega)[0]: Rot(phi, theta, omega) | [0]; end; "
            "gate S[0]: S | [0]; end; "
            "gate SWAP[0, 1]: SWAP | [0, 1]; end; "
            "gate SX[0]: SX | [0]; end; "
            "gate Squeezing(r, theta, cutoff)[0]: Squeezing(r, theta, cutoff) | [0]; end; "
            "gate T[0]: T | [0]; end; "
            "gate Toffoli[0, 1, 2]: Toffoli | [0, 1, 2]; end; "
            "gate TwoModeSqueezing(r, theta, cutoff)[0, 1]: TwoModeSqueezing(r, theta, cutoff) | [0, 1]; end; "
            "gate U1(phi)[0]: U1(phi) | [0]; end; "
            "gate U2(phi, lam)[0]: U2(phi, lam) | [0]; end; "
            "gate U3(theta, phi, lam)[0]: U3(theta, phi, lam) | [0]; end; "
            "gate X[0]: X | [0]; end; "
            "gate Y[0]: Y | [0]; end; "
            "gate Z[0]: Z | [0]; end; "
            "gate beamsplitter(theta, phi, cutoff)[0, 1]: beamsplitter(theta, phi, cutoff) | [0, 1]; end; "
            "gate bs(theta, phi, cutoff)[0, 1]: bs(theta, phi, cutoff) | [0, 1]; end; "
            "gate cnot[0, 1]: cnot | [0, 1]; end; "
            "gate cphaseshift(phi)[0, 1]: cphaseshift(phi) | [0, 1]; end; "
            "gate crot(phi, theta, omega)[0, 1]: crot(phi, theta, omega) | [0, 1]; end; "
            "gate crx(theta)[0, 1]: crx(theta) | [0, 1]; end; "
            "gate cry(theta)[0, 1]: cry(theta) | [0, 1]; end; "
            "gate crz(theta)[0, 1]: crz(theta) | [0, 1]; end; "
            "gate cswap[0, 1, 2]: cswap | [0, 1, 2]; end; "
            "gate cx[0, 1]: cx | [0, 1]; end; "
            "gate cy[0, 1]: cy | [0, 1]; end; "
            "gate cz[0, 1]: cz | [0, 1]; end; "
            "gate d(r, phi, cutoff)[0]: d(r, phi, cutoff) | [0]; end; "
            "gate displacement(r, phi, cutoff)[0]: displacement(r, phi, cutoff) | [0]; end; "
            "gate h[0]: h | [0]; end; "
            "gate hadamard[0]: hadamard | [0]; end; "
            "gate iswap[0, 1]: iswap | [0, 1]; end; "
            "gate not[0]: not | [0]; end; "
            "gate paulix[0]: paulix | [0]; end; "
            "gate pauliy[0]: pauliy | [0]; end; "
            "gate pauliz[0]: pauliz | [0]; end; "
            "gate phaseshift(phi)[0]: phaseshift(phi) | [0]; end; "
            "gate rot(phi, theta, omega)[0]: rot(phi, theta, omega) | [0]; end; "
            "gate rx(theta)[0]: rx(theta) | [0]; end; "
            "gate ry(theta)[0]: ry(theta) | [0]; end; "
            "gate rz(theta)[0]: rz(theta) | [0]; end; "
            "gate s[0]: s | [0]; end; "
            "gate squeezing(r, theta, cutoff)[0]: squeezing(r, theta, cutoff) | [0]; end; "
            "gate swap[0, 1]: swap | [0, 1]; end; "
            "gate sx[0]: sx | [0]; end; "
            "gate t[0]: t | [0]; end; "
            "gate toffoli[0, 1, 2]: toffoli | [0, 1, 2]; end; "
            "gate twomodesqueezing(r, theta, cutoff)[0, 1]: twomodesqueezing(r, theta, cutoff) | [0, 1]; end; "
            "gate u1(phi)[0]: u1(phi) | [0]; end; "
            "gate u2(phi, lam)[0]: u2(phi, lam) | [0]; end; "
            "gate u3(theta, phi, lam)[0]: u3(theta, phi, lam) | [0]; end; "
            "gate x[0]: x | [0]; end; "
            "gate y[0]: y | [0]; end; "
            "gate z[0]: z | [0]; end;"
        )
        assert have_xir_program == want_xir_program


@pytest.mark.parametrize(
    "program",
    [
        xir.parse_script(""),
        xir.parse_script("use xstd;"),
        xir.parse_script("use xstd; H | [0];"),
    ],
)
def test_run_xir_program_with_no_output_statements(program):
    """Tests that running an XIR program with no output statements returns an empty list."""
    assert jet.run_xir_program(program) == []


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            xir.parse_script(
                """
                X | [0];

                amplitude(state: 0) | [0];
                amplitude(state: 1) | [0];
                """
            ),
            [0, 1],
        ),
        (
            xir.parse_script(
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
            xir.parse_script(
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
            xir.parse_script("X | [0]; amplitude | [0];"),
            r"Statement 'amplitude \| \[0\]' is missing a 'state' parameter\.",
        ),
        (
            xir.parse_script("X | [0]; amplitude(state: 2) | [0];"),
            r"Statement 'amplitude\(state: 2\) \| \[0\]' has a 'state' parameter which is too large\.",
        ),
        (
            xir.parse_script("CNOT | [0, 1]; amplitude(state: 0) | [0];"),
            r"Statement 'amplitude\(state: 0\) \| \[0\]' must be applied to \[0 \.\. 1\]\.",
        ),
        (
            xir.parse_script("CNOT | [0, 1]; amplitude(state: 0) | [1, 0];"),
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


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            xir.parse_script(
                """
                probabilities | [0];
                """
            ),
            [1, 0],
        ),
        (
            xir.parse_script(
                """
                probabilities | [0, 1, 2];
                """
            ),
            [1, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            xir.parse_script(
                """
                X | [0];

                probabilities | [0];
                """
            ),
            [0, 1],
        ),
        (
            xir.parse_script(
                """
                X | [1];

                probabilities | [0, 1];
                """
            ),
            [0, 1, 0, 0],
        ),
        (
            xir.parse_script(
                """
                H | [0];
                CNOT | [0, 1];

                probabilities | [0, 1];
                """
            ),
            [0.5, 0, 0, 0.5],
        ),
        (
            xir.parse_script(
                """
                H | [0];
                Y | [0];

                probabilities | [0];
                """
            ),
            [0.5, 0.5],
        ),
    ],
)
def test_run_xir_program_with_probabilities_statement(program, want_result):
    """Tests that running an XIR program with a probabilities statement gives the correct result."""
    assert jet.run_xir_program(program) == [pytest.approx(want_result)]


def test_run_xir_program_with_invalid_probabilities_statement():
    """Tests that a ValueError is raised when an XIR program contains an invalid
    probabilities statement.
    """
    program = xir.parse_script("CNOT | [0, 1]; probabilities | [0];")
    match = r"Statement 'probabilities \| \[0\]' must be applied to \[0 \.\. 1\]\."

    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            xir.parse_script(
                """
                gate H2:
                    H | [0];
                    H | [1];
                end;

                X | [0];

                amplitude(state: 0) | [0];
                amplitude(state: 1) | [0];
                """
            ),
            [0, 1],
        ),
        (
            xir.parse_script(
                """
                gate H2 [0, 1]:
                    H | [0];
                    H | [1];
                end;

                H2 | [0, 1];

                amplitude(state: 0) | [0, 1];
                amplitude(state: 1) | [0, 1];
                amplitude(state: 2) | [0, 1];
                amplitude(state: 3) | [0, 1];
                """
            ),
            [0.5, 0.5, 0.5, 0.5],
        ),
        (
            xir.parse_script(
                """
                gate Flip[coin]:
                    X | [coin];
                end;

                gate Stay[coin]:
                    Flip | [coin];
                    Flip | [coin];
                end;

                gate FlipStay[0, 1]:
                    Flip | [0];
                    Stay | [1];
                end;

                FlipStay | [0, 1];

                amplitude(state: 0) | [0, 1];
                amplitude(state: 1) | [0, 1];
                amplitude(state: 2) | [0, 1];
                amplitude(state: 3) | [0, 1];
                """
            ),
            [0, 0, 1, 0],
        ),
        (
            xir.parse_script(
                """
                gate RY3 (a, b, c) [0, 1, 2]:
                    RY(a) | [0];
                    RY(b) | [1];
                    RY(c) | [2];
                end;

                gate RX3 (a, b, c) [0, 1, 2]:
                    RY3(a: b, b: c, c: a) | [0, 1, 2];
                end;

                RX3(a: 0, b: 3.141592653589793, c: 3.141592653589793) | [0, 1, 2];

                amplitude(state: 0) | [0, 1, 2];
                amplitude(state: 1) | [0, 1, 2];
                amplitude(state: 2) | [0, 1, 2];
                amplitude(state: 3) | [0, 1, 2];
                amplitude(state: 4) | [0, 1, 2];
                amplitude(state: 5) | [0, 1, 2];
                amplitude(state: 6) | [0, 1, 2];
                amplitude(state: 7) | [0, 1, 2];
                """
            ),
            [0, 0, 0, 0, 0, 0, 1, 0],
        ),
    ],
)
def test_run_xir_program_with_valid_gate_definitions(program, want_result):
    """Tests that running an XIR program with valid gate definitions gives the correct result."""
    assert jet.run_xir_program(program) == pytest.approx(want_result)


@pytest.mark.parametrize(
    "program, match",
    [
        (
            xir.parse_script(
                """
                gate Circle[0]:
                    Circle | [0];
                end;

                Circle | [0];
                """
            ),
            r"Gate 'Circle' has a circular dependency\.",
        ),
        (
            xir.parse_script(
                """
                gate Day[0]:
                    Dawn | [0];
                end;

                gate Dawn[0]:
                    Dusk | [0];
                end;

                gate Dusk[0]:
                    Dawn | [0];
                end;

                Day | [0];
                """
            ),
            r"Gate 'Dawn' has a circular dependency\.",
        ),
        (
            xir.parse_script(
                """
                gate Incomplete[0]:
                    Missing | [0];
                end;

                Incomplete | [0];
                """
            ),
            r"Statement 'Missing \| \[0\]' applies a gate which has not been defined\.",
        ),
        (
            xir.parse_script(
                """
                gate Negate [0]:
                    X | [0];
                end;

                Negate(0) | [0];
                """
            ),
            r"Statement 'Negate\(0\) \| \[0\]' has the wrong number of parameters\.",
        ),
        (
            xir.parse_script(
                """
                gate Spin(theta) [0]:
                    Rot(theta, theta, theta) | [0];
                end;

                Spin(phi: pi) | [0];
                """
            ),
            r"Statement 'Spin\(phi: PI\) \| \[0\]' has an invalid set of parameters\.",
        ),
        (
            xir.parse_script(
                """
                gate Permute [0, 1]:
                    SWAP | [0, 1];
                end;

                Permute | [0];
                """
            ),
            r"Statement 'Permute \| \[0\]' has the wrong number of wires\.",
        ),
    ],
)
def test_run_xir_program_with_invalid_gate_definitions(program, match):
    """Tests that a ValueError is raised when an XIR program contains an
    invalid gate definition.
    """
    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


def test_run_xir_program_with_overridden_gate_definition():
    """Tests that a UserWarning is given when an XIR program contains a
    duplicate gate definition.
    """
    program = xir.parse_script("gate X: NOT | [0]; end;")

    with pytest.warns(UserWarning, match=r"Gate 'X' overrides the Jet gate with the same name\."):
        jet.run_xir_program(program)


def test_run_xir_program_with_unsupported_statement():
    """Tests that a ValueError is raised when an XIR program contains an
    unsupported statement.
    """
    program = xir.parse_script("use xstd; halt | [0];")

    with pytest.raises(ValueError, match=r"Statement 'halt \| \[0\]' is not supported\."):
        jet.run_xir_program(program)
