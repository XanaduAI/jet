# pylint: disable=redefined-outer-name
"""Unit tests for the program class"""

from decimal import Decimal
from typing import Any, Dict, Iterable, List, Set

import pytest
from xir.program import Declaration, OperatorStmt, Statement, XIRProgram


@pytest.fixture
def program():
    """Returns an empty XIR program."""
    return XIRProgram()


# pylint: disable=protected-access
def make_program(
    called_functions: Set[str],
    declarations: Dict[str, List[Declaration]],
    gates: Dict[str, Dict[str, Any]],
    includes: List[str],
    operators: Dict[str, Dict[str, Any]],
    options: Dict[str, Any],
    statements: Set[str],
    variables: List[str],
):
    """Returns an XIR program with the given attributes."""
    program = XIRProgram()
    program._called_functions = called_functions
    program._declarations = declarations
    program._gates = gates
    program._includes = includes
    program._operators = operators
    program._options = options
    program._statements = statements
    program._variables = variables
    return program


class TestSerialize:
    """Unit tests for the serialize method of the XIRProgram"""

    def test_empty_program(self, program):
        """Tests serializing an empty program."""
        assert program.serialize() == ""

    def test_includes(self, program):
        """Tests serializing a XIR program with includes."""
        program.add_include("xstd")
        program.add_include("randomlib")
        res = program.serialize()
        assert res == "use xstd;\nuse randomlib;"

    #####################
    # Test declarations
    #####################

    @pytest.mark.parametrize("params, wires, declaration_type, want_res", [
            (["a", "b"], (0,), "gate", "gate name(a, b)[0];"),
            ([], (0, 2, 1), "operator", "operator name[0, 2, 1];"),
            (["theta"], ("a", "b", "c"), "output", "output name(theta)[a, b, c];"),
        ],
    )
    def test_declarations(self, params, wires, declaration_type, want_res):
        """Test serializing gate, operation and output declarations"""
        decl = Declaration("name", params, wires, declaration_type=declaration_type)

        irprog = XIRProgram()
        irprog._declarations[declaration_type].append(decl)
        have_res = irprog.serialize()

        assert have_res  == want_res

    @pytest.mark.parametrize("params, want_res", [
            (["a", "b"], "function name(a, b);"),
            ([], "function name;"),
            (["theta"], "function name(theta);"),
        ],
    )
    def test_func_declaration(self, params, want_res):
        """Test serializing function declarations"""
        decl = Declaration("name", params, (), declaration_type="function")

        irprog = XIRProgram()
        irprog._declarations["func"].append(decl)
        have_res = irprog.serialize()
        assert have_res  == want_res

    ###################
    # Test statements
    ###################

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [[0, 3.14, -42]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_statements_params(self, program, name, params, wires):
        """Tests serializing an XIR program with general (gate) statements."""
        stmt = Statement(name, params, wires)

        program.add_statement(stmt)
        res = program.serialize()

        params_str = ", ".join(map(str, params))
        wires_str = ", ".join(map(str, wires))
        assert res == f"{name}({params_str}) | [{wires_str}];"

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_statements_no_params(self, program, name, wires):
        """Tests serializing an XIR program with general (gate) statements without parameters."""
        stmt = Statement(name, [], wires)

        program.add_statement(stmt)
        res = program.serialize()

        wires_str = ", ".join(map(str, wires))
        assert res == f"{name} | [{wires_str}];"

    @pytest.mark.parametrize("pref", [42, Decimal("3.14"), "2 * a + 1"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operator_stmt(self, program, pref, wires):
        """Tests serializing an XIR program with operator statements."""
        xyz = "XYZ"
        terms = [(xyz[i], w) for i, w in enumerate(wires)]
        terms_str = " @ ".join(f"{t[0]}[{t[1]}]" for t in terms)

        program.add_operator("H", ["a", "b"], (0, 1), [OperatorStmt(pref, terms)])

        res = program.serialize()
        assert res == f"operator H(a, b)[0, 1]:\n    {pref}, {terms_str};\nend;"

    #########################
    # Test gate definitions
    #########################

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_gates_params_and_wires(self, program, name, params, wires):
        """Tests serializing an XIR program with gates that have both parameters and wires."""
        stmts = [Statement("rz", [0.13], (0,)), Statement("cnot", [], (0, 1))]
        program.add_gate(name, params, wires, stmts)

        res = program.serialize()

        params_str = ", ".join(map(str, params))
        wires_str = ", ".join(map(str, wires))
        assert (
            res == f"gate {name}({params_str})[{wires_str}]:"
            "\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"
        )

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_gates_no_params(self, program, name, wires):
        """Tests serializing an XIR program with gates that have no parameters."""
        stmts = [Statement("rz", [0.13], (0,)), Statement("cnot", [], (0, 1))]
        program.add_gate(name, [], wires, stmts)

        res = program.serialize()

        wires_str = ", ".join(map(str, wires))
        assert res == f"gate {name}[{wires_str}]:\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    def test_gates_no_wires(self, program, name, params):
        """Tests serializing an XIR program with gates that have no wires."""
        stmts = [Statement("rz", [0.13], (0,)), Statement("cnot", [], (0, 1))]
        program.add_gate(name, params, (), stmts)

        res = program.serialize()

        params_str = ", ".join(map(str, params))
        assert res == f"gate {name}({params_str}):\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    @pytest.mark.parametrize("name", ["mygate", "a_beautiful_gate"])
    def test_gates_no_params_and_no_wires(self, program, name):
        """Tests serializing an XIR program with gates that have no parameters or wires."""
        stmts = [Statement("rz", [0.13], (0,)), Statement("cnot", [], (0, 1))]
        program.add_gate(name, [], (), stmts)

        res = program.serialize()
        assert res == f"gate {name}:\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    #############################
    # Test operator definitions
    #############################

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operators_params_and_wires(self, program, name, params, wires):
        """Tests serializing an XIR program with operators that have both parameters and wires."""
        stmts = [OperatorStmt(42, [("X", 0), ("Y", 1)])]
        program.add_operator(name, params, wires, stmts)

        res = program.serialize()

        params_str = ", ".join(map(str, params))
        wires_str = ", ".join(map(str, wires))
        assert res == f"operator {name}({params_str})[{wires_str}]:\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operators_no_params(self, program, name, wires):
        """Tests serializing an XIR program with operators that have no parameters."""
        stmts = [OperatorStmt(42, [("X", 0), ("Y", 1)])]
        program.add_operator(name, [], wires, stmts)

        res = program.serialize()

        wires_str = ", ".join(map(str, wires))
        assert res == f"operator {name}[{wires_str}]:\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    def test_operators_no_wires(self, program, name, params):
        """Tests serializing an XIR program with operators that have no declared wires."""
        stmts = [OperatorStmt(42, [("X", 0), ("Y", 1)])]
        program.add_operator(name, params, (), stmts)

        res = program.serialize()

        params_str = ", ".join(map(str, params))
        assert res == f"operator {name}({params_str}):\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["my_op", "op2"])
    def test_operators_no_params_and_no_wires(self, program, name):
        """Tests serializing an XIR program with operators that have no parameters or wires."""
        stmts = [OperatorStmt(42, [("X", 0), ("Y", 1)])]
        program.add_operator(name, [], (), stmts)

        res = program.serialize()
        assert res == f"operator {name}:\n    42, X[0] @ Y[1];\nend;"


class TestXIRProgram:
    """Unit tests for the XIRProgram class"""

    def test_init(self):
        """Tests that an (empty) XIR program can be constructed."""
        program = XIRProgram(version="1.2.3", use_floats=False)

        assert program.version == "1.2.3"
        assert program.use_floats is False

        assert list(program.wires) == []

        assert set(program.called_functions) == set()
        assert dict(program.declarations) == {"gate": [], "func": [], "output": [], "operator": []}
        assert dict(program.gates) == {}
        assert list(program.includes) == []
        assert dict(program.operators) == {}
        assert list(program.options) == []
        assert list(program.statements) == []
        assert set(program.variables) == set()

    def test_repr(self):
        """Test that the string representation of an XIR program has the correct format."""
        program = XIRProgram(version="1.2.3")
        assert repr(program) == "<XIRProgram: version=1.2.3>"

    def test_add_called_function(self, program):
        """Tests that called functions can be added to an XIR program."""
        program.add_called_function("cos")
        assert set(program.called_functions) == {"cos"}

        program.add_called_function("sin")
        assert set(program.called_functions) == {"cos", "sin"}

        program.add_called_function("cos")
        assert set(program.called_functions) == {"cos", "sin"}

    def test_add_declaration(self, program):
        """Tests that declarations can be added to an XIR program."""
        tan = FuncDeclaration("tan", 1)
        program.add_declaration("func", tan)
        assert program.declarations == {"func": [tan], "gate": [], "operator": [], "output": []}

        u2 = GateDeclaration("U2", 2, 1)
        program.add_declaration("gate", u2)
        assert program.declarations == {"func": [tan], "gate": [u2], "operator": [], "output": []}

        z3 = GateDeclaration("Z3", 0, 3)
        program.add_declaration("operator", z3)
        assert program.declarations == {"func": [tan], "gate": [u2], "operator": [z3], "output": []}

    def test_add_declaration_with_same_key(self, program):
        """Tests that multiple declarations with the same key can be added to an XIR program."""
        amplitude = OutputDeclaration("amplitude")
        program.add_declaration("output", amplitude)

        probabilities = OutputDeclaration("probabilities")
        program.add_declaration("output", probabilities)

        assert program.declarations["output"] == [amplitude, probabilities]

    def test_add_declaration_with_wrong_subclass(self, program):
        """Tests that the concrete type of a declaration does not affect the
        key(s) that can be associated with it in an XIR program.
        """
        decl = OutputDeclaration("gradient")
        program.add_declaration("func", decl)
        assert program.declarations == {"func": [decl], "gate": [], "operator": [], "output": []}

    def test_add_declaration_with_wrong_key(self, program):
        """Tests that an exception is raised when a declaration with an unknown
        key is added to an XIR program.
        """
        decl = Declaration("Variable")
        with pytest.raises(KeyError, match=r"Key 'var' is not a supported declaration"):
            program.add_declaration("var", decl)

    def test_add_declaration_with_same_name(self, program):
        """Tests that a warning is issued when two declarations with the same
        name are added to an XIR program.
        """
        atan1 = FuncDeclaration("atan", 1)
        program.add_declaration("func", atan1)

        with pytest.warns(UserWarning, match=r"Func 'atan' has already been declared"):
            atan2 = FuncDeclaration("atan", 2)
            program.add_declaration("func", atan2)

        assert program.declarations["func"] == [atan1, atan2]

    def test_add_gate(self, program):
        """Tests that gates can be added to an XIR program."""
        crx = {
            "params": ["theta"],
            "wires": (0, 1),
            "statements": [
                Statement(name="X", params=[], wires=[0]),
                Statement(name="X", params=[], wires=[0]),
                Statement(name="CRX", params=["theta"], wires=[0, 1]),
            ],
        }
        program.add_gate("CRX", **crx)
        assert program.gates == {"CRX": crx}

        u3 = {
            "params": ["theta", "phi", "lam"],
            "wires": [1],
            "statements": [Statement(name="U3", params=["theta", "phi", "lam"], wires=[1])],
        }
        program.add_gate("U3", **u3)
        assert program.gates == {"CRX": crx, "U3": u3}

    def test_add_gate_with_same_name(self, program):
        """Tests that a warning is issued when two gates with the same name are
        added to an XIR program.
        """
        phi = {"params": ["phi"], "wires": [0, 1], "statements": []}
        psi = {"params": ["psi"], "wires": [0, 1], "statements": []}

        program.add_gate("CRX", **phi)
        assert program.gates == {"CRX": phi}

        with pytest.warns(Warning, match=r"Gate 'CRX' already defined"):
            program.add_gate("CRX", **psi)

        assert program.gates == {"CRX": psi}

    def test_add_include(self, program):
        """Tests that includes can be added to an XIR program."""
        program.add_include("complex")
        assert list(program.includes) == ["complex"]

        program.add_include("algorithm")
        assert list(program.includes) == ["complex", "algorithm"]

    def test_add_include_with_same_name(self, program):
        """Tests that a warning is issued when two identical includes are added
        to an XIR program.
        """
        program.add_include("memory")

        with pytest.warns(Warning, match=r"Module 'memory' is already included"):
            program.add_include("memory")

        assert list(program.includes) == ["memory"]

    def test_add_operator(self, program):
        """Tests that operators can be added to an XIR program."""
        x = {"params": [], "wires": [0], "statements": [OperatorStmt(pref=1, terms=[("X", 0)])]}
        program.add_operator("X", **x)
        assert program.operators == {"X": x}

        y = {"params": [], "wires": [1], "statements": [OperatorStmt(pref=2, terms=[("Y", 0)])]}
        program.add_operator("Y", **y)
        assert program.operators == {"X": x, "Y": y}

    def test_add_operator_with_same_name(self, program):
        """Tests that a warning is issued when two operators with the same name
        are added to an XIR program.
        """
        degrees = {"params": ["degrees"], "wires": (0, 1), "statements": []}
        radians = {"params": ["radians"], "wires": (0, 1), "statements": []}

        program.add_operator("Rotate", **degrees)
        assert program.operators == {"Rotate": degrees}

        with pytest.warns(Warning, match=r"Operator 'Rotate' already defined"):
            program.add_operator("Rotate", **radians)

        assert program.operators == {"Rotate": radians}

    def test_add_option(self, program):
        """Tests that options can be added to an XIR program."""
        program.add_option("cutoff", 3)
        assert program.options == {"cutoff": 3}

        program.add_option("speed", "fast")
        assert program.options == {"cutoff": 3, "speed": "fast"}

    def test_add_option_with_same_key(self, program):
        """Tests that a warning is issued when two options with the same key
        are added to an XIR program.
        """
        program.add_option("precision", "float")
        assert program.options == {"precision": "float"}

        with pytest.warns(Warning, match=r"Option 'precision' already set"):
            program.add_option("precision", "double")

        assert program.options == {"precision": "double"}

    def test_add_statement(self, program):
        """Tests that statements can be added to an XIR program."""
        program.add_statement(Statement("X", {}, [0]))
        assert [stmt.name for stmt in program.statements] == ["X"]

        program.add_statement(Statement("Y", {}, [0]))
        assert [stmt.name for stmt in program.statements] == ["X", "Y"]

        program.add_statement(Statement("X", {}, [0]))
        assert [stmt.name for stmt in program.statements] == ["X", "Y", "X"]

    def test_add_variable(self, program):
        """Tests that variables can be added to an XIR program."""
        program.add_variable("theta")
        assert set(program.variables) == {"theta"}

        program.add_variable("phi")
        assert set(program.variables) == {"theta", "phi"}

        program.add_variable("theta")
        assert set(program.variables) == {"theta", "phi"}

    def test_merge_zero_programs(self):
        """Test that a ValueError is raised when zero XIR programs are merged."""
        with pytest.raises(ValueError, match=r"Merging requires at least one XIR program"):
            XIRProgram.merge()

    def test_merge_programs_with_different_versions(self):
        """Test that a ValueError is raised when two XIR programs with different
        versions are merged.
        """
        p1 = XIRProgram(version="0.0.1")
        p2 = XIRProgram(version="0.0.2")

        match = r"XIR programs with different versions cannot be merged"

        with pytest.raises(ValueError, match=match):
            XIRProgram.merge(p1, p2)

    def test_merge_programs_with_different_float_settings(self):
        """Test that a warning is issued when two XIR programs with different
        float settings are merged.
        """
        p1 = XIRProgram(use_floats=True)
        p2 = XIRProgram(use_floats=False)

        match = r"XIR programs with different float settings are being merged"

        with pytest.warns(UserWarning, match=match):
            assert XIRProgram.merge(p1, p2).use_floats is True

    @pytest.mark.parametrize(
        ["programs", "want_result"],
        [
            pytest.param(
                [
                    make_program(
                        called_functions={"tanh"},
                        declarations={"func": [], "gate": [], "operator": [], "output": []},
                        gates={"U2": {"params": ["phi", "lam"], "wires": [0], "statements": []}},
                        includes=["xstd"],
                        operators={"Z": {"params": [], "wires": [1], "statements": []}},
                        options={"cutoff": 3},
                        statements=[Statement("U1", {"phi": 1}, [0])],
                        variables={"angle"},
                    ),
                ],
                make_program(
                    called_functions={"tanh"},
                    declarations={"func": [], "gate": [], "operator": [], "output": []},
                    gates={"U2": {"params": ["phi", "lam"], "wires": [0], "statements": []}},
                    includes=["xstd"],
                    operators={"Z": {"params": [], "wires": [1], "statements": []}},
                    options={"cutoff": 3},
                    statements=[Statement("U1", {"phi": 1}, [0])],
                    variables={"angle"},
                ),
                id="One XIR program",
            ),
            pytest.param(
                [
                    make_program(
                        called_functions={"cos"},
                        declarations={
                            "func": [FuncDeclaration("cos", 1)],
                            "gate": [GateDeclaration("H", 0, 1)],
                            "operator": [],
                            "output": [],
                        },
                        gates={"H": {"params": [], "wires": [0], "statements": []}},
                        includes=[],
                        operators={"X": {"params": [], "wires": [0], "statements": []}},
                        options={"cutoff": 2},
                        statements=[Statement("S", [], [0])],
                        variables={"theta"},
                    ),
                    make_program(
                        called_functions={"sin"},
                        declarations={
                            "func": [FuncDeclaration("sin", 1)],
                            "gate": [],
                            "operator": [OperatorDeclaration("Y", 0, 1)],
                            "output": [],
                        },
                        gates={"D": {"params": ["r", "phi"], "wires": [1], "statements": []}},
                        includes=["xstd"],
                        operators={"Y": {"params": [], "wires": [1], "statements": []}},
                        options={"cutoff": 4},
                        statements=[Statement("T", [], [0])],
                        variables=set(),
                    ),
                ],
                make_program(
                    called_functions={"cos", "sin"},
                    declarations={
                        "func": [FuncDeclaration("cos", 1), FuncDeclaration("sin", 1)],
                        "gate": [GateDeclaration("H", 0, 1)],
                        "operator": [OperatorDeclaration("Y", 0, 1)],
                        "output": [],
                    },
                    gates={
                        "H": {"params": [], "wires": [0], "statements": []},
                        "D": {"params": ["r", "phi"], "wires": [1], "statements": []},
                    },
                    includes=["xstd"],
                    operators={
                        "X": {"params": [], "wires": [0], "statements": []},
                        "Y": {"params": [], "wires": [1], "statements": []},
                    },
                    options={"cutoff": 4},
                    statements=[Statement("S", [], [0]), Statement("T", [], [0])],
                    variables={"theta"},
                ),
                id="Two XIR programs",
            ),
        ],
    )
    def test_merge_programs(self, programs, want_result):
        """Test that one or more XIR programs can be merged."""
        have_result = XIRProgram.merge(*programs)

        assert have_result.called_functions == want_result.called_functions
        assert have_result.variables == want_result.variables
        assert have_result.includes == want_result.includes
        assert have_result.options == want_result.options

        def serialize(mapping: Dict[str, Iterable[Any]]) -> Dict[str, List[str]]:
            """Partially serializes a dictionary with sequence values by casting
            each item of each sequence into a string.
            """
            return {k: list(map(str, v)) for k, v in mapping.items()}

        have_declarations = serialize(have_result.declarations)
        want_declarations = serialize(want_result.declarations)
        assert have_declarations == want_declarations

        have_gates = {k: serialize(v) for k, v in have_result.gates.items()}
        want_gates = {k: serialize(v) for k, v in want_result.gates.items()}
        assert have_gates == want_gates

        have_operators = {k: serialize(v) for k, v in have_result.operators.items()}
        want_operators = {k: serialize(v) for k, v in want_result.operators.items()}
        assert have_operators == want_operators

        have_statements = list(map(str, have_result.statements))
        want_statements = list(map(str, want_result.statements))
        assert have_statements == want_statements

    # pylint: disable=protected-access
    @pytest.mark.parametrize("version", ["4.2.0", "0.3.0"])
    def test_validate_version(self, version):
        """Test that a correct version passes validation."""
        XIRProgram._validate_version(version)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("version", [42, 0.2, True, object()])
    def test_validate_version_with_wrong_type(self, version):
        """Test that an exception is raised when a version has the wrong type."""
        with pytest.raises(TypeError, match=r"Version '[^']*' must be a string"):
            XIRProgram._validate_version(version)

    # pylint: disable=protected-access
    @pytest.mark.parametrize("version", ["", "abc", "4.2", "1.2.3-alpha", "0.1.2.3"])
    def test_validate_version_with_wrong_format(self, version):
        """Test that an exception is raised when a version has the wrong format."""
        with pytest.raises(ValueError, match=r"Version '[^']*' must be a semantic version"):
            XIRProgram._validate_version(version)
