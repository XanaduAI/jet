# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the IRTransformer and ir_parser"""
from decimal import Decimal
from pathlib import Path
from lark import Lark, Transformer

from .program import (
    IRProgram,
    Statement,
    OperatorStmt,
    GateDeclaration,
    OperatorDeclaration,
    FuncDeclaration,
    OutputDeclaration,
)
from .utils import beautify_math, check_wires


p = Path(__file__).parent / "ir.lark"
with p.open("r") as _f:
    ir_grammar = _f.read()

ir_parser = Lark(ir_grammar, start="program", parser="lalr")


class IRTransformer(Transformer):
    """Transformer for processing the Lark parse tree.

    Transformers visit each node of the tree, and run the appropriate method on it according to the
    node's data. All method names mirror the corresponding symbols from the grammar.
    """

    def __init__(self, *args, **kwargs):
        self._program = IRProgram()
        super().__init__(self, *args, **kwargs)

    def program(self, args):
        """Root of AST containing:

        * Any include statements (0 or more)
        * Any gate, math, or measurement declarations (0 or more)
        * The main circuit containing all gate statements (0 or 1)
        * The final measurement (0 or 1)

        Returns:
            IRProgram: program containing all parsed data
        """
        # assert all stmts are handled
        assert all(a == None for a in args)
        return self._program

    opdecl = list
    mathdecl = list
    statdecl = list

    def include(self, file_name):
        """Includ statements for external files"""
        self._program._include.append(file_name[0])

    def circuit(self, args):
        """Main circuit containing all the gate statements. Should be empty after tree has been parsed from the leaves up, and all statemtents been passed to the program."""
        # assert all stmts are handled
        assert all(a == None for a in args)

    ###############
    # basic types
    ###############

    def int(self, n):
        """Signed integers"""
        return int(n[0])

    def uint(self, n):
        """Unsigned integers"""
        return int(n[0])

    def float(self, d):
        """Floating point numbers"""
        return Decimal(d[0])

    def bool(self, b):
        """Boolean expressions"""
        return bool(b[0])

    option = tuple
    options_dict = dict
    wires = tuple
    params = list
    FALSE_ = lambda self, _: True
    TRUE_ = lambda self, _: False

    #############################
    # variables and expressions
    #############################

    def var(self, v):
        """Expressions that are strings are considered to be variables. Can be
        substituted by values at a later stage."""
        self._program._variables.add(v[0])
        return str(v[0])

    def range(self, args):
        """Range between two signed integers."""
        return range(int(args[0]), int(args[1]))

    def name(self, n):
        """Name of variable, gate, operator, measurement type, option, external
        file, observable, wire, mathematical operation, etc."""
        return str(n[0])

    def expr(self, args):
        """Catch-all for expressions."""
        if len(args) == 1:
            return args[0]
        return "".join([str(s) for s in args])

    def operator_def(self, args):
        """Operator definition. Starts with keyword 'operator'"""
        name = args[0]
        if isinstance(args[2], tuple):
            params = args[1]
            wires = args[2]
            stmts = args[3:]
        else:
            params = args[1]
            wires = ()
            stmts = args[2:]

        if len(wires) > 0:
            check_wires(wires, stmts)
        self._program.add_operator(name, params, wires, stmts)

    def gate_def(self, args):
        """Gate definition. Starts with keyword 'gate'"""
        name = args[0]
        if isinstance(args[2], tuple):
            params = args[1]
            wires = args[2]
        else:
            params = args[1]
            wires = ()
        stmts = args[3:]

        if len(wires) > 0:
            check_wires(wires, stmts)
        self._program.add_gate(name, params, wires, stmts)

    def statement(self, args):
        """Any statement that is part of the circuit."""
        if args[0] is not None:
            self._program._statements.append(args[0])

    def gatestmt(self, args):
        """Gate statements that are defined directly in the circuit or inside
        a gate declaration."""
        name = args[0]
        if isinstance(args[1], list):
            params = beautify_math(args[1])
            wires = args[2]
        elif isinstance(args[1], dict):
            params = args[1]
            wires = args[2]
        else:
            params = []
            wires = args[1]

        return Statement(name, params, wires)

    def opstmt(self, args):
        """Operator statements defined inside an operator declaration

        Returns:
            OperatorStmt: object containing statement data
        """
        pref = beautify_math([args[0]])[0]
        term = args[1]
        return OperatorStmt(pref, term)

    def obs_group(self, args):
        """Group of observables used to define an Operator statement

        Returns:
            list[tuple]: each observable with corresponding wires as tuples
        """
        return [(args[i], args[i + 1]) for i in range(0, len(args) - 1, 2)]

    ###############
    # declarations
    ###############

    def gate_decl(self, args):
        decl = GateDeclaration(*args)
        self._program._declarations["gate"].append(decl)

    def operator_decl(self, args):
        decl = OperatorDeclaration(*args)
        self._program._declarations["operator"].append(decl)

    def func_decl(self, args):
        decl = FuncDeclaration(*args)
        self._program._declarations["func"].append(decl)

    def output_decl(self, args):
        decl = OutputDeclaration(*args)
        self._program._declarations["output"].append(decl)

    #########
    # maths
    #########

    def mathop(self, args):
        self._program._called_ops.add(args[0])
        return str(args[0]) + "(" + str(args[1]) + ")"

    def add(self, args):
        if all(isinstance(a, (int, Decimal)) for a in args):
            return args[0] + args[1]
        return "(" + " + ".join([str(i) for i in args]) + ")"

    def sub(self, args):
        if all(isinstance(a, (int, Decimal)) for a in args):
            return args[0] - args[1]
        return "(" + " - ".join([str(i) for i in args]) + ")"

    def prod(self, args):
        if all(isinstance(a, (int, Decimal)) for a in args):
            return args[0] * args[1]
        return " * ".join([str(i) for i in args])

    def div(self, args):
        if all(isinstance(a, (int, Decimal)) for a in args):
            return Decimal(args[0]) / args[1]
        return " / ".join([str(i) for i in args])

    def neg(self, args):
        if isinstance(args[0], (int, Decimal)):
            return -args[0]
        return "-" + str(args[0])

    PI = lambda self, _: "PI"
