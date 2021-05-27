"""Unit tests for the interfaces module"""

from decimal import Decimal
from typing import List, Tuple

import pytest
import strawberryfields as sf
from strawberryfields import ops

from xir.interfaces.strawberryfields_io import to_program, to_xir
from xir.program import GateDeclaration, Statement, XIRProgram


def create_xir_prog(
    data: List[Tuple],
    external_libs: List[str] = None,
    include_decl: bool = True,
    version: str = None,
) -> XIRProgram:
    """Create an XIRProgram object used for testing"""
    # if no version number is passed, use the default one (by not specifying it)
    if version is None:
        irprog = XIRProgram()
    else:
        irprog = XIRProgram(version=version)

    # add the statements to the program
    stmts = [Statement(n, p, w) for n, p, w in data]
    irprog._statements.extend(stmts)

    # if declaration should be included, add them to the program
    if include_decl:
        declarations = [GateDeclaration(n, len(p), len(w)) for n, p, w in data]
        irprog._declarations["gate"].extend(declarations)

    # if any external libraries/files are included, add them to the program
    if external_libs is not None:
        irprog._include.extend(external_libs)

    return irprog


def create_sf_prog(num_of_wires: int, ops: List[Tuple]):
    """Create a Strawberry Fields program"""
    prog = sf.Program(num_of_wires)

    with prog.context as q:
        for gate, params, wires in ops:
            regrefs = [q[w] for w in wires]
            if len(params) == 0:
                gate | regrefs
            else:
                gate(*params) | regrefs

    return prog


class TestXIRToStrawberryFields:
    """Unit tests for the XIR to Strawberry Fields conversion"""

    def test_empty_irprogram(self):
        """Test that converting an empty XIR program raises an error"""
        irprog = create_xir_prog(data=[])
        with pytest.raises(ValueError, match="XIR program is empty and cannot be transformed"):
            to_program(irprog)

    def test_gate_not_defined(self):
        """Test unknown gate raises error"""
        circuit_data = [
            ("not_a_real_gate", [Decimal("0.42")], (1, 2, 3)),
        ]
        irprog = create_xir_prog(data=circuit_data)

        with pytest.raises(NameError, match="operation 'not_a_real_gate' not defined"):
            to_program(irprog)

    def test_gates_no_args(self):
        """Test that gates without arguments work"""
        circuit_data = [
            ("Vac", [], (0,)),
        ]
        irprog = create_xir_prog(data=circuit_data)

        sfprog = to_program(irprog)

        assert len(sfprog) == 1
        assert sfprog.circuit[0].op.__class__.__name__ == "Vacuum"
        assert sfprog.circuit[0].reg[0].ind == 0

    def test_gates_with_args(self):
        """Test that gates with arguments work"""

        circuit_data = [
            ("Sgate", [Decimal("0.1"), Decimal("0.2")], (0,)),
            ("Sgate", [Decimal("0.3")], (1,)),
        ]
        irprog = create_xir_prog(data=circuit_data)
        sfprog = to_program(irprog)

        assert len(sfprog) == 2
        assert sfprog.circuit[0].op.__class__.__name__ == "Sgate"
        assert sfprog.circuit[0].op.p[0] == 0.1
        assert sfprog.circuit[0].op.p[1] == 0.2
        assert sfprog.circuit[0].reg[0].ind == 0

        assert sfprog.circuit[1].op.__class__.__name__ == "Sgate"
        assert sfprog.circuit[1].op.p[0] == 0.3
        assert sfprog.circuit[1].op.p[1] == 0.0  # default value
        assert sfprog.circuit[1].reg[0].ind == 1


class TestStrawberryFieldsToXIR:
    """Unit tests for the XIR to Strawberry Fields conversion"""

    def test_empty_sfprogram(self):
        """Test that converting from an empty SF program works"""
        sfprog = create_sf_prog(num_of_wires=2, ops=[])
        irprog = to_xir(sfprog)

        assert irprog.version == "0.1.0"
        assert irprog.statements == []
        assert irprog.include == []
        assert irprog.statements == []
        assert irprog.declarations == {
            "gate": [],
            "func": [],
            "output": [],
            "operator": [],
        }

        assert irprog.gates == dict()
        assert irprog.operators == dict()
        assert irprog.variables == set()
        assert irprog._called_ops == set()

    @pytest.mark.parametrize("add_decl", [True, False])
    def test_gates_no_args(self, add_decl):
        """Test unknown gate raises error"""
        circuit_data = [
            (ops.Vac, [], (0,)),
        ]

        sfprog = create_sf_prog(num_of_wires=2, ops=circuit_data)
        irprog = to_xir(sfprog, add_decl=add_decl)

        assert irprog.statements[0].name == "Vacuum"
        assert irprog.statements[0].params == []
        assert irprog.statements[0].wires == (0,)

        if add_decl:
            assert len(irprog.declarations["gate"]) == 1
            assert irprog.declarations["gate"][0].name == "Vacuum"
            assert irprog.declarations["gate"][0].num_params == 0
            assert irprog.declarations["gate"][0].num_wires == 1
        else:
            assert irprog.declarations["gate"] == []

    @pytest.mark.parametrize("add_decl", [True, False])
    def test_gates_with_args(self, add_decl):
        """Test that gates with arguments work"""

        circuit_data = [
            (ops.Sgate, [0.1, 0.2], (0,)),
            (ops.Sgate, [Decimal("0.3")], (1,)),
        ]

        sfprog = create_sf_prog(num_of_wires=2, ops=circuit_data)
        irprog = to_xir(sfprog, add_decl=add_decl)

        assert irprog.statements[0].name == "Sgate"
        assert irprog.statements[0].params == [0.1, 0.2]
        assert irprog.statements[0].wires == (0,)

        assert irprog.statements[1].name == "Sgate"
        assert irprog.statements[1].params == [Decimal("0.3"), 0.0]
        assert irprog.statements[1].wires == (1,)

        if add_decl:
            assert len(irprog.declarations["gate"]) == 1
            assert irprog.declarations["gate"][0].name == "Sgate"
            assert irprog.declarations["gate"][0].num_params == 2
            assert irprog.declarations["gate"][0].num_wires == 1
        else:
            assert irprog.declarations["gate"] == []
