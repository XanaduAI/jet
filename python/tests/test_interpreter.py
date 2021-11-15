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
                    gate X[0];
                    gate Y[0];
                    gate Z[0];
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
                    gate U1(phi)[0];
                    gate U2(phi, lam)[0];
                    gate U3(theta, phi, lam)[0];
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
                    gate CSWAP[0, 1, 2];
                    gate SWAP[0, 1];
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
                    gate H[0];
                    gate Hadamard[0];
                    """
                ),
                id="Duplicate",
            ),
        ],
    )
    def test_fake_registry(self, monkeypatch, registry, want_xir_script):
        """Tests that the correct XIRProgram is returned for the fake gate registry."""
        monkeypatch.setattr("jet.GateFactory.registry", registry)
        have_xir_script = jet.get_xir_manifest().serialize()
        assert have_xir_script == want_xir_script

    def test_real_registry(self):
        """Tests that the correct XIRProgram is returned for the real gate registry."""
        have_xir_program = jet.get_xir_manifest().serialize(minimize=True)
        want_xir_program = (
            "gate BS(theta, phi)[0, 1]; "
            "gate Beamsplitter(theta, phi)[0, 1]; "
            "gate CNOT[0, 1]; "
            "gate CPhaseShift(phi)[0, 1]; "
            "gate CRX(theta)[0, 1]; "
            "gate CRY(theta)[0, 1]; "
            "gate CRZ(theta)[0, 1]; "
            "gate CRot(phi, theta, omega)[0, 1]; "
            "gate CSWAP[0, 1, 2]; "
            "gate CX[0, 1]; "
            "gate CY[0, 1]; "
            "gate CZ[0, 1]; "
            "gate D(r, phi)[0]; "
            "gate Displacement(r, phi)[0]; "
            "gate H[0]; "
            "gate Hadamard[0]; "
            "gate ISWAP[0, 1]; "
            "gate NOT[0]; "
            "gate PauliX[0]; "
            "gate PauliY[0]; "
            "gate PauliZ[0]; "
            "gate PhaseShift(phi)[0]; "
            "gate RX(theta)[0]; "
            "gate RY(theta)[0]; "
            "gate RZ(theta)[0]; "
            "gate Rot(phi, theta, omega)[0]; "
            "gate S[0]; "
            "gate SWAP[0, 1]; "
            "gate SX[0]; "
            "gate Squeezing(r, theta)[0]; "
            "gate T[0]; "
            "gate Toffoli[0, 1, 2]; "
            "gate TwoModeSqueezing(r, theta)[0, 1]; "
            "gate U1(phi)[0]; "
            "gate U2(phi, lam)[0]; "
            "gate U3(theta, phi, lam)[0]; "
            "gate X[0]; "
            "gate Y[0]; "
            "gate Z[0]; "
            "gate beamsplitter(theta, phi)[0, 1]; "
            "gate bs(theta, phi)[0, 1]; "
            "gate cnot[0, 1]; "
            "gate cphaseshift(phi)[0, 1]; "
            "gate crot(phi, theta, omega)[0, 1]; "
            "gate crx(theta)[0, 1]; "
            "gate cry(theta)[0, 1]; "
            "gate crz(theta)[0, 1]; "
            "gate cswap[0, 1, 2]; "
            "gate cx[0, 1]; "
            "gate cy[0, 1]; "
            "gate cz[0, 1]; "
            "gate d(r, phi)[0]; "
            "gate displacement(r, phi)[0]; "
            "gate h[0]; "
            "gate hadamard[0]; "
            "gate iswap[0, 1]; "
            "gate not[0]; "
            "gate paulix[0]; "
            "gate pauliy[0]; "
            "gate pauliz[0]; "
            "gate phaseshift(phi)[0]; "
            "gate rot(phi, theta, omega)[0]; "
            "gate rx(theta)[0]; "
            "gate ry(theta)[0]; "
            "gate rz(theta)[0]; "
            "gate s[0]; "
            "gate squeezing(r, theta)[0]; "
            "gate swap[0, 1]; "
            "gate sx[0]; "
            "gate t[0]; "
            "gate toffoli[0, 1, 2]; "
            "gate twomodesqueezing(r, theta)[0, 1]; "
            "gate u1(phi)[0]; "
            "gate u2(phi, lam)[0]; "
            "gate u3(theta, phi, lam)[0]; "
            "gate x[0]; "
            "gate y[0]; "
            "gate z[0];"
        )
        assert have_xir_program == want_xir_program


@pytest.mark.parametrize(
    "program",
    [
        xir.parse_script(""),
        xir.parse_script("use <xc/jet>;"),
        xir.parse_script("H | [0];"),
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
                use <xc/jet>;

                options:
                    dimension: 2;
                end;

                H | [0];
                S | [0];
                Displacement(3, 1) | [0];

                amplitude(state: [0]) | [0];
                amplitude(state: [1]) | [0];
                """
            ),
            [-0.011974639958 - 0.012732623852j, 0.012732623852 - 0.043012087532j],
        ),
        (
            xir.parse_script(
                """
                use <xc/jet>;

                options:
                    dimension: 3;
                end;

                Squeezing(1, 2) | [0];
                Squeezing(2, 1) | [1];

                amplitude(state: [0, 0]) | [0, 1];
                amplitude(state: [0, 1]) | [0, 1];
                amplitude(state: [0, 2]) | [0, 1];
                amplitude(state: [1, 0]) | [0, 1];
                amplitude(state: [1, 1]) | [0, 1];
                amplitude(state: [1, 2]) | [0, 1];
                amplitude(state: [2, 0]) | [0, 1];
                amplitude(state: [2, 1]) | [0, 1];
                amplitude(state: [2, 2]) | [0, 1];
                """
            ),
            [
                0.415035263978,
                0,
                -0.152860853701 - 0.238066674351j,
                0,
                0,
                0,
                0.093012260922 - 0.203235497887j,
                0,
                -0.150834249845 + 0.021500900893j,
            ],
        ),
    ],
)
def test_run_xir_program_with_options(program, want_result):
    """Tests that running an XIR program with script-level options gives the correct result."""
    assert jet.run_xir_program(program) == pytest.approx(want_result)


@pytest.mark.parametrize(
    "program, match",
    [
        (
            xir.parse_script("options: dimension: [2]; end;"),
            r"Option 'dimension' must be an integer\.",
        ),
        (
            xir.parse_script("options: dimension: 1; end;"),
            r"Option 'dimension' must be greater than one\.",
        ),
    ],
)
def test_run_xir_program_with_invalid_options(program, match):
    """Tests that a ValueError is raised when an XIR program contains an invalid option."""
    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


def test_run_xir_program_with_unsupported_options():
    """Tests that a UserWarning is given when an XIR program specifies at least
    one unsupported option.
    """
    program = xir.parse_script("options: dimension: 3; VSync: off; end;")
    with pytest.warns(UserWarning, match=r"Option 'VSync' is not supported and will be ignored\."):
        jet.run_xir_program(program)


def test_run_xir_program_with_incompatible_dimensions():
    """Tests that a ValueError is raised when an XIR program applies a qubit
    gate in the context of a CV circuit with a dimension greater than two.
    """
    program = xir.parse_script("options: dimension: 3; end; X | [0];")
    match = (
        r"Statement 'X \| \[0\]' applies a gate with a dimension \(2\) that "
        r"differs from the dimension of the circuit \(3\)\."
    )

    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            xir.parse_script(
                """
                use <xc/jet>;

                X | [0];

                amplitude(state: [0]) | [0];
                amplitude(state: [1]) | [0];
                """
            ),
            [0, 1],
        ),
        (
            xir.parse_script(
                """
                use <xc/jet>;

                H | [0];
                CNOT | [0, 1];

                amplitude(state: [0, 0]) | [0, 1];
                amplitude(state: [0, 1]) | [0, 1];
                amplitude(state: [1, 0]) | [0, 1];
                amplitude(state: [1, 1]) | [0, 1];
                """
            ),
            [1 / sqrt(2), 0, 0, 1 / sqrt(2)],
        ),
        (
            xir.parse_script(
                """
                use <xc/jet>;

                TwoModeSqueezing(3, 1) | [0, 1];

                amplitude(state: [0, 0]) | [0, 1];
                amplitude(state: [0, 1]) | [0, 1];
                amplitude(state: [1, 0]) | [0, 1];
                amplitude(state: [1, 1]) | [0, 1];
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
            xir.parse_script("X | [0]; amplitude(state) | [0];"),
            r"Statement 'amplitude\(state\) \| \[0\]' is missing a 'state' parameter\.",
        ),
        (
            xir.parse_script("X | [0]; amplitude(state: [0, -1]) | [0, 1];"),
            (
                r"Statement 'amplitude\(state: \[0, -1\]\) \| \[0, 1\]' has a 'state' "
                r"parameter with at least one entry that falls outside the range \[0, 2\)\."
            ),
        ),
        (
            xir.parse_script("X | [0]; amplitude(state: [0, 2]) | [0, 1];"),
            (
                r"Statement 'amplitude\(state: \[0, 2\]\) \| \[0, 1\]' has a 'state' "
                r"parameter with at least one entry that falls outside the range \[0, 2\)\."
            ),
        ),
        (
            xir.parse_script("X | [0]; amplitude(state: [0, 0]) | [0];"),
            (
                r"Statement 'amplitude\(state: \[0, 0\]\) \| \[0\]' has a 'state' "
                r"parameter with 2 \(!= 1\) entries\."
            ),
        ),
        (
            xir.parse_script("X | [0]; amplitude(state: [0]) | [0, 1];"),
            (
                r"Statement 'amplitude\(state: \[0\]\) \| \[0, 1\]' has a 'state' "
                r"parameter with 1 \(!= 2\) entries\."
            ),
        ),
        (
            xir.parse_script("CNOT | [0, 1]; amplitude(state: [0, 1]) | [0];"),
            r"Statement 'amplitude\(state: \[0, 1\]\) \| \[0\]' must be applied to \[0 \.\. 1\]\.",
        ),
        (
            xir.parse_script("CNOT | [0, 1]; amplitude(state: [0, 1]) | [1, 0];"),
            (
                r"Statement 'amplitude\(state: \[0, 1\]\) \| \[1, 0\]' "
                r"must be applied to \[0 \.\. 1\]\."
            ),
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

                amplitude(state: [0]) | [0];
                amplitude(state: [1]) | [0];
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

                amplitude(state: [0, 0]) | [0, 1];
                amplitude(state: [0, 1]) | [0, 1];
                amplitude(state: [1, 0]) | [0, 1];
                amplitude(state: [1, 1]) | [0, 1];
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

                amplitude(state: [0, 0]) | [0, 1];
                amplitude(state: [0, 1]) | [0, 1];
                amplitude(state: [1, 0]) | [0, 1];
                amplitude(state: [1, 1]) | [0, 1];
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

                amplitude(state: [0, 0, 0]) | [0, 1, 2];
                amplitude(state: [0, 0, 1]) | [0, 1, 2];
                amplitude(state: [0, 1, 0]) | [0, 1, 2];
                amplitude(state: [0, 1, 1]) | [0, 1, 2];
                amplitude(state: [1, 0, 0]) | [0, 1, 2];
                amplitude(state: [1, 0, 1]) | [0, 1, 2];
                amplitude(state: [1, 1, 0]) | [0, 1, 2];
                amplitude(state: [1, 1, 1]) | [0, 1, 2];
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


@pytest.mark.parametrize(
    "program, want_result",
    [
        (
            xir.parse_script(
                """
                obs Z [0]:
                    1, Z[0];
                end;

                expval(observable: Z) | [0];
                """
            ),
            [1],
        ),
        (
            xir.parse_script(
                """
                obs Z3 [wire]:
                    3, Z[wire];
                end;

                X | [0];

                expval(observable: Z3) | [0];
                """
            ),
            [-3],
        ),
        (
            xir.parse_script(
                """
                obs XY[0, 1]:
                    1, X[0];
                    1, Y[1];
                end;

                obs YX[0, 1]:
                    1, Y[0];
                    1, X[1];
                end;

                RY(pi/2) | [0];
                RX(pi/4) | [1];

                expval(observable: XY) | [0, 1];
                expval(observable: YX) | [0, 1];
                """,
                eval_pi=True,
            ),
            [-1 / sqrt(2), 0],
        ),
    ],
)
def test_run_xir_program_with_expval_statements(program, want_result):
    """Tests that running an XIR program with expected value statements gives
    the correct result.
    """
    assert jet.run_xir_program(program) == pytest.approx(want_result)


@pytest.mark.parametrize(
    "program, match",
    [
        (
            xir.parse_script("expval | [0];"),
            r"Statement 'expval \| \[0\]' is missing an 'observable' parameter\.",
        ),
        (
            xir.parse_script("expval(observable: dne) | [0];"),
            (
                r"Statement 'expval\(observable: dne\) \| \[0\]' has an "
                r"'observable' parameter which references an undefined observable\."
            ),
        ),
        (
            xir.parse_script("obs box[0]; expval(observable: box) | [0];"),
            (
                r"Statement 'expval\(observable: box\) \| \[0\]' has an "
                r"'observable' parameter which references an undefined observable\."
            ),
        ),
        (
            xir.parse_script(
                """
                obs up(scale)[0]:
                    scale, Z[0];
                end;

                expval(observable: up) | [0];
                """
            ),
            (
                r"Statement 'expval\(observable: up\) \| \[0\]' has an 'observable' "
                r"parameter which references a parameterized observable\."
            ),
        ),
        (
            xir.parse_script(
                """
                obs obs[0]:
                    1, Z[0];
                end;

                X | [0];
                X | [1];

                expval(observable: obs) | [0, 1];
                """
            ),
            (
                r"Statement 'expval\(observable: obs\) \| \[0, 1\]' has an 'observable' "
                r"parameter which applies the wrong number of wires\."
            ),
        ),
        (
            xir.parse_script(
                """
                obs natural[0]:
                    one, Z[0];
                end;

                expval(observable: natural) | [0];
                """
            ),
            (
                r"Observable statement 'one, Z\[0\]' has a prefactor \(one\) "
                r"which cannot be converted to a floating-point number\."
            ),
        ),
    ],
)
def test_run_xir_program_with_invalid_expval_statements(program, match):
    """Tests that a ValueError is raised when an XIR program contains an
    invalid expected value statement.
    """
    with pytest.raises(ValueError, match=match):
        jet.run_xir_program(program)


def test_run_xir_program_with_unsupported_statement():
    """Tests that a ValueError is raised when an XIR program contains an
    unsupported statement.
    """
    program = xir.parse_script("halt | [0];")

    with pytest.raises(ValueError, match=r"Statement 'halt \| \[0\]' is not supported\."):
        jet.run_xir_program(program)
