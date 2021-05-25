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
        irprog._include.extend(["xstd", "randomlib"])
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


class TestIRProgram:
    """Unit tests for the XIRProgram class"""

    def test_empty_initialize(self):
        """Test initializing an empty program"""
        irprog = XIRProgram()

        assert irprog.version == "0.1.0"
        assert irprog.statements == []
        assert irprog.include == []
        assert irprog.statements == []
        assert irprog.declarations == {"gate": [], "func": [], "output": [], "operator": []}

        assert irprog.gates == dict()
        assert irprog.operators == dict()
        assert irprog.variables == set()
        assert irprog._called_ops == set()

    def test_repr(self):
        irprog = XIRProgram()
        assert irprog.__repr__() == f"<XIRProgram: version=0.1.0>"

    @pytest.mark.parametrize(
        "version",
        [
            "4.2.0",
            "0.3.0",
        ],
    )
    def test_version(self, version):
        """Test that the correct version is passed"""
        irprog = XIRProgram(version=version)
        assert irprog.version == version
        assert irprog.__repr__() == f"<XIRProgram: version={version}>"

    @pytest.mark.parametrize("version", ["4.2", "0.1.2.3", "abc", 42, 0.2])
    def test_invalid_version(self, version):
        """Test that error is raised when passing invalid version numbers"""
        with pytest.raises((ValueError, TypeError), match="Invalid version number"):
            XIRProgram(version=version)

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
        with pytest.warns(Warning, match="Gate already defined"):
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
        with pytest.warns(Warning, match="Operator already defined"):
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
