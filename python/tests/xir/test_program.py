"""Unit tests for the program class"""

from decimal import Decimal

import pytest

from xir.program import (
    FuncDeclaration,
    GateDeclaration,
    OperatorDeclaration,
    OperatorStmt,
    OutputDeclaration,
    Statement,
    XIRProgram,
)


class TestSerialize:
    """Unit tests for the serialize method of the XIRProgram"""

    def test_empty_program(self):
        """Test serializing an empty program"""
        irprog = XIRProgram()
        res = irprog.serialize()
        assert res == ""

    def test_includes(self):
        """Test serializing the included libraries statement"""
        irprog = XIRProgram()
        irprog._includes.extend(["xstd", "randomlib"])
        res = irprog.serialize()
        assert res == "use xstd;\nuse randomlib;"

    #####################
    # Test declarations
    #####################

    @pytest.mark.parametrize("name", ["rx", "CNOT", "a_gate"])
    @pytest.mark.parametrize("num_params", [0, 1, 42])
    @pytest.mark.parametrize("num_wires", [1, 42])
    def test_gate_declaration(self, name, num_params, num_wires):
        """Test serializing gate declarations"""
        decl = GateDeclaration(name, num_params=num_params, num_wires=num_wires)

        irprog = XIRProgram()
        irprog._declarations["gate"].append(decl)
        res = irprog.serialize()
        assert res == f"gate {name}, {num_params}, {num_wires};"

    @pytest.mark.parametrize("name", ["sin", "COS", "arc_tan"])
    @pytest.mark.parametrize("num_params", [0, 1, 42])
    def test_func_declaration(self, name, num_params):
        """Test serializing function declarations"""
        decl = FuncDeclaration(name, num_params=num_params)

        irprog = XIRProgram()
        irprog._declarations["func"].append(decl)
        res = irprog.serialize()
        assert res == f"func {name}, {num_params};"

    @pytest.mark.parametrize("name", ["op", "OPERATOR", "op_42"])
    @pytest.mark.parametrize("num_params", [0, 1, 42])
    @pytest.mark.parametrize("num_wires", [1, 42])
    def test_operator_declaration(self, name, num_params, num_wires):
        """Test serializing operator declarations"""
        decl = OperatorDeclaration(name, num_params=num_params, num_wires=num_wires)

        irprog = XIRProgram()
        irprog._declarations["operator"].append(decl)
        res = irprog.serialize()
        assert res == f"operator {name}, {num_params}, {num_wires};"

    @pytest.mark.parametrize("name", ["sample", "amplitude"])
    def test_output_declaration(self, name):
        """Test serializing output declarations"""
        decl = OutputDeclaration(name)

        irprog = XIRProgram()
        irprog._declarations["output"].append(decl)
        res = irprog.serialize()
        assert res == f"output {name};"

    ###################
    # Test statements
    ###################

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [[0, 3.14, -42]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_statements_params(self, name, params, wires):
        """Test serializing general (gate) statements"""
        stmt = Statement(name, params, wires)

        irprog = XIRProgram()
        irprog._statements.append(stmt)
        res = irprog.serialize()

        params_str = ", ".join(str(p) for p in params)
        wires_str = ", ".join(str(w) for w in wires)
        assert res == f"{name}({params_str}) | [{wires_str}];"

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_statements_no_params(self, name, wires):
        """Test serializing general (gate) statements without parameters"""
        stmt = Statement(name, [], wires)

        irprog = XIRProgram()
        irprog._statements.append(stmt)
        res = irprog.serialize()

        wires_str = ", ".join(str(w) for w in wires)
        assert res == f"{name} | [{wires_str}];"

    @pytest.mark.parametrize("pref", [42, Decimal("3.14"), "2 * a + 1"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operator_stmt(self, pref, wires):
        """Test serializing operator statements"""
        irprog = XIRProgram()

        xyz = "XYZ"
        terms = [(xyz[i], w) for i, w in enumerate(wires)]
        terms_str = " @ ".join(f"{t[0]}[{t[1]}]" for t in terms)

        irprog._operators["H"] = {
            "params": ["a", "b"],
            "wires": (0, 1),
            "statements": [
                OperatorStmt(pref, terms),
            ],
        }

        res = irprog.serialize()
        assert res == f"operator H(a, b)[0, 1]:\n    {pref}, {terms_str};\nend;"

    #########################
    # Test gate definitions
    #########################

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_gates_params_and_wires(self, name, params, wires):
        """Test serializing gates with parameters and wires declared"""
        irprog = XIRProgram()
        irprog._gates[name] = {
            "params": params,
            "wires": wires,
            "statements": [
                Statement("rz", [0.13], (0,)),
                Statement("cnot", [], (0, 1)),
            ],
        }
        res = irprog.serialize()

        params_str = ", ".join(str(p) for p in params)
        wires_str = ", ".join(str(w) for w in wires)
        assert (
            res == f"gate {name}({params_str})[{wires_str}]:"
            "\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"
        )

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_gates_no_params(self, name, wires):
        """Test serializing gates with no parameters"""
        irprog = XIRProgram()
        irprog._gates[name] = {
            "params": [],
            "wires": wires,
            "statements": [
                Statement("rz", [0.13], (0,)),
                Statement("cnot", [], (0, 1)),
            ],
        }
        res = irprog.serialize()

        wires_str = ", ".join(str(w) for w in wires)
        assert res == f"gate {name}[{wires_str}]:\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    @pytest.mark.parametrize("name", ["ry", "toffoli"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    def test_gates_no_wires(self, name, params):
        """Test serializing gates without declared wires"""
        irprog = XIRProgram()
        irprog._gates[name] = {
            "params": params,
            "wires": (),
            "statements": [
                Statement("rz", [0.13], (0,)),
                Statement("cnot", [], (0, 1)),
            ],
        }
        res = irprog.serialize()

        params_str = ", ".join(str(p) for p in params)
        assert res == f"gate {name}({params_str}):\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    @pytest.mark.parametrize("name", ["mygate", "a_beautiful_gate"])
    def test_gates_no_params_and_no_wires(self, name):
        """Test serializing gates with no parameters and without declared wires"""
        irprog = XIRProgram()

        irprog._gates[name] = {
            "params": [],
            "wires": (),
            "statements": [
                Statement("rz", [0.13], (0,)),
                Statement("cnot", [], (0, 1)),
            ],
        }
        res = irprog.serialize()
        assert res == f"gate {name}:\n    rz(0.13) | [0];\n    cnot | [0, 1];\nend;"

    #############################
    # Test operator definitions
    #############################

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operators_params_and_wires(self, name, params, wires):
        """Test serializing operators with parameters and wires declared"""
        irprog = XIRProgram()
        irprog._operators[name] = {
            "params": params,
            "wires": wires,
            "statements": [
                OperatorStmt(42, [("X", 0), ("Y", 1)]),
            ],
        }
        res = irprog.serialize()

        params_str = ", ".join(str(p) for p in params)
        wires_str = ", ".join(str(w) for w in wires)
        assert res == f"operator {name}({params_str})[{wires_str}]:\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("wires", [(0, 1), (0,), (0, 2, 42)])
    def test_operators_no_params(self, name, wires):
        """Test serializing operators with no parameters"""
        xyz = "XYZ"

        irprog = XIRProgram()
        irprog._operators[name] = {
            "params": [],
            "wires": wires,
            "statements": [
                OperatorStmt(42, [("X", 0), ("Y", 1)]),
            ],
        }
        res = irprog.serialize()

        wires_str = ", ".join(str(w) for w in wires)
        assert res == f"operator {name}[{wires_str}]:\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["H", "my_op"])
    @pytest.mark.parametrize("params", [["a", "b"]])
    def test_operators_no_wires(self, name, params):
        """Test serializing operators without declared wires"""
        irprog = XIRProgram()
        irprog._operators[name] = {
            "params": params,
            "wires": (),
            "statements": [
                OperatorStmt(42, [("X", 0), ("Y", 1)]),
            ],
        }
        res = irprog.serialize()

        params_str = ", ".join(str(p) for p in params)
        assert res == f"operator {name}({params_str}):\n    42, X[0] @ Y[1];\nend;"

    @pytest.mark.parametrize("name", ["my_op", "op2"])
    def test_operators_no_params_and_no_wires(self, name):
        """Test serializing operators with no parameters and without declared wires"""
        irprog = XIRProgram()

        irprog._operators[name] = {
            "params": [],
            "wires": (),
            "statements": [
                OperatorStmt(42, [("X", 0), ("Y", 1)]),
            ],
        }
        res = irprog.serialize()
        assert res == f"operator {name}:\n    42, X[0] @ Y[1];\nend;"


class TestXIRProgram:
    """Unit tests for the XIRProgram class"""

    def test_init(self):
        """Tests that an (empty) XIR program can be constructed."""
        program = XIRProgram(version="1.2.3", use_floats=False)

        assert program.version == "1.2.3"
        assert program.use_floats is False

        assert list(program.wires) == []

        assert list(program.called_functions) == []
        assert dict(program.declarations) == {
            "gate": [],
            "func": [],
            "output": [],
            "operator": [],
        }
        assert dict(program.gates) == {}
        assert list(program.includes) == []
        assert dict(program.operators) == {}
        assert list(program.options) == []
        assert list(program.statements) == []
        assert list(program.variables) == []

    def test_repr(self):
        """Test that __repr__() returns a string with the correct format."""
        program = XIRProgram(version="1.2.3")
        assert repr(program) == f"<XIRProgram: version=1.2.3>"

    def test_add_called_function(self):
        """Tests that the called functions can be added to an XIR program."""
        program = XIRProgram()
        assert set(program.called_functions) == set()

        program.add_called_function("cos")
        assert set(program.called_functions) == {"cos"}

        program.add_called_function("sin")
        assert set(program.called_functions) == {"cos", "sin"}

        program.add_called_function("cos")
        assert set(program.called_functions) == {"cos", "sin"}

    def test_add_statement(self):
        """Tests that statements can be added to an XIR program."""
        program = XIRProgram()
        assert list(program.statements) == []

        program.add_statement(Statement("X", {}, [0]))
        assert [stmt.name for stmt in program.statements] == ["X"]

        program.add_statement(Statement("X", {}, [1]))
        assert [stmt.name for stmt in program.statements] == ["X", "Y"]

        program.add_statement(Statement("X", {}, [0]))
        assert [stmt.name for stmt in program.statements] == ["X", "Y", "X"]

    def test_add_variable(self):
        """Tests that variables can be added to an XIR program."""
        program = XIRProgram()
        assert set(program.variables) == set()

        program.add_variable("theta")
        assert set(program.variables) == {"theta"}

        program.add_variable("phi")
        assert set(program.variables) == {"theta", "phi"}

        program.add_variable("theta")
        assert set(program.variables) == {"theta", "phi"}

    def test_add_gate(self):
        """Test that the add_gate function works"""
        irprog = XIRProgram()
        statements = [
            Statement("rx", ["x"], (0,)),
            Statement("ry", ["y"], (0,)),
            Statement("rx", ["x"], (0,)),
        ]
        params = ["x", "y", "z"]
        wires = (0,)
        irprog.add_gate("rot", params, wires, statements)

        assert irprog.gates == {"rot": {"params": params, "wires": wires, "statements": statements}}

        # check that gate is replaced, with a warning, if added again
        params = ["a", "b", "c"]
        wires = (1,)
        with pytest.warns(Warning, match=r"Gate '[^']*' already defined"):
            irprog.add_gate("rot", params, wires, statements)

        assert irprog.gates == {"rot": {"params": params, "wires": wires, "statements": statements}}

        # check that a second gate can be added and the former is kept
        params_2 = []
        wires_2 = (0, 1)
        statements_2 = ["cnot", [], (0, 1)]
        irprog.add_gate("cnot", params_2, wires_2, statements_2)

        assert irprog.gates == {
            "rot": {"params": params, "wires": wires, "statements": statements},
            "cnot": {"params": params_2, "wires": wires_2, "statements": statements_2},
        }

    def test_add_operator(self):
        """Test that the add_operator function works"""
        irprog = XIRProgram()
        statements = [
            OperatorStmt(13, [("X", 0), ("Y", 1)]),
            OperatorStmt(-2, [("ABC", 1), ("D", 0)]),
        ]
        params = []
        wires = (0, 1)
        irprog.add_operator("H", params, wires, statements)

        assert irprog.operators == {
            "H": {"params": params, "wires": wires, "statements": statements}
        }

        # check that operator is replaced, with a warning, if added again
        statements = [
            OperatorStmt(2, [("X", 0)]),
        ]
        wires = (0,)
        with pytest.warns(Warning, match=r"Operator '[^']*' already defined"):
            irprog.add_operator("H", params, wires, statements)

        assert irprog.operators == {
            "H": {"params": params, "wires": wires, "statements": statements}
        }

        # check that a second operator can be added and the former is kept
        params_2 = ["a"]
        wires_2 = (2,)
        statements_2 = [OperatorStmt("2 * a", [("X", 2)])]
        irprog.add_operator("my_op", params_2, wires_2, statements_2)

        assert irprog.operators == {
            "H": {"params": params, "wires": wires, "statements": statements},
            "my_op": {"params": params_2, "wires": wires_2, "statements": statements_2},
        }

    @pytest.mark.parametrize("version", ["4.2.0", "0.3.0"])
    def test_validate_version(self, version):
        """Test that a valid version passes validation."""
        XIRProgram._validate_version(version)

    @pytest.mark.parametrize("version", [42, 0.2, True, object()])
    def test_validate_version_with_wrong_type(self, version):
        """Test that an Exception is raised when a version has the wrong type."""
        with pytest.raises(TypeError, match=r"Version '[^']*' must be a string"):
            XIRProgram._validate_version(version)

    @pytest.mark.parametrize("version", ["", "abc", "4.2", "1.2.3-alpha", "0.1.2.3"])
    def test_validate_version_with_wrong_format(self, version):
        """Test that an Exception is raised when a version has the wrong format."""
        with pytest.raises(ValueError, match=r"Version '[^']*' must be a semantic version"):
            XIRProgram._validate_version(version)
