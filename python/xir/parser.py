"""This module contains the XIRTransformer and xir_parser"""
import math
from decimal import Decimal
from pathlib import Path
from typing import Union

from lark import Lark, Transformer

from .decimal_complex import DecimalComplex
from .program import (
    XIRProgram,
    Declaration,
    Statement,
    OperatorStmt,
)
from .utils import check_wires, simplify_math

p = Path(__file__).parent / "ir.lark"
with p.open("r") as _f:
    ir_grammar = _f.read()

xir_parser = Lark(ir_grammar, start="program", parser="lalr")

# pylint: disable=missing-function-docstring
class XIRTransformer(Transformer):
    """Transformer for processing the Lark parse tree.

    Transformers visit each node of the tree, and run the appropriate method on it according to the
    node's data. All method names mirror the corresponding symbols from the grammar.

    Keyword args:
        eval_pi (bool): Whether pi should be evaluated and stored as a float
            instead of symbolically as a string. Defaults to ``False``.
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(self, *args, **kwargs):
        self._eval_pi = kwargs.pop("eval_pi", False)
        self._use_floats = kwargs.pop("use_floats", True)

        self._program = XIRProgram()
        super().__init__(self, *args, **kwargs)

    @property
    def eval_pi(self) -> bool:  # pylint: disable=used-before-assignment
        """Reports whether pi is evaluated and stored as a float."""
        return self._eval_pi

    @property
    def use_floats(self) -> bool:
        """Reports whether floats and complex types are used."""
        return self._use_floats

    def program(self, args):
        """Root of AST containing:

        * Any include statements (0 or more)
        * Any gate, math, or measurement declarations (0 or more)
        * The main circuit containing all gate statements (0 or 1)
        * The final measurement (0 or 1)

        Returns:
            XIRProgram: program containing all parsed data
        """
        # assert all stmts are handled
        assert all(a is None for a in args)
        return self._program

    opdecl = list
    mathdecl = list
    statdecl = list

    def include(self, file_name):
        """Includ statements for external files"""
        self._program.add_include(file_name[0])

    def circuit(self, args):
        """Main circuit containing all the gate statements. Should be empty after
        the tree has been parsed from the leaves up and all statetents have been
        passed to the program.
        """
        # assert all stmts are handled
        assert all(a is None for a in args)

    def script_options(self, args):
        """Script level options."""
        for name, value in args:
            self._program.add_option(name, value)

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

    def imag(self, c):
        """Imaginary numbers"""
        return DecimalComplex("0.0", c[0])

    def bool(self, b):
        """Boolean expressions"""
        return bool(b[0])

    def wires(self, w):
        """Tuple with wires and identifier"""
        return "wires", tuple(w)

    def params(self, p):
        """Tuple with params and identifier"""
        # p will be a list with one element, which is either a list (params_list)
        # or a dict (params_dict)
        return "params", p[0]

    params_list = list
    params_dict = dict

    option = tuple
    array = list

    ADJOINT = str
    CTRL = str

    def FALSE_(self, _):
        return False

    def TRUE_(self, _):
        return True

    #############################
    # variables and expressions
    #############################

    def var(self, v):
        """Expressions that are strings are considered to be variables. Can be
        substituted by values at a later stage."""
        self._program.add_variable(v[0])
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
        name = args.pop(0)
        wires = ()
        params = []
        stmts = []

        for i, arg in enumerate(args):
            if is_param(arg):
                params = arg[1]
            elif is_wire(arg):
                wires = arg[1]

            if isinstance(arg, OperatorStmt):
                stmts = args[i:]
                break

        if len(wires) > 0:
            check_wires(wires, stmts)
        self._program.add_operator(name, params, wires, stmts)

    def gate_def(self, args):
        """Gate definition. Starts with keyword 'gate'"""
        name = args.pop(0)
        wires = ()
        params = []
        stmts = []

        for i, arg in enumerate(args):
            if is_param(arg):
                params = arg[1]
            elif is_wire(arg):
                wires = arg[1]

            if isinstance(arg, Statement):
                stmts = args[i:]
                break

        if len(wires) > 0:
            check_wires(wires, stmts)
        self._program.add_gate(name, params, wires, stmts)

    def statement(self, args):
        """Any statement that is part of the circuit."""
        if args[0] is not None:
            self._program.add_statement(args[0])

    def gatestmt(self, args):
        """Gate statements that are defined directly in the circuit or inside
        a gate definition."""
        adjoint = False
        ctrl_wires = set()

        while args[0] in ("adjoint", "ctrl"):
            a = args.pop(0)
            if a == "adjoint":
                adjoint = not adjoint
            elif a == "ctrl":
                ctrl_wires.update(args.pop(0)[1])

        name = args.pop(0)
        if is_param(args[0]):
            if isinstance(args[0][1], list):
                params = list(map(simplify_math, args[0][1]))
                wires = args[1][1]
            else:  # if dict
                params = {k: simplify_math(v) for k, v in args[0][1].items()}
                wires = args[1][1]
        else:
            params = []
            wires = args[0][1]

        stmt_options = {
            "ctrl_wires": tuple(sorted(ctrl_wires, key=hash)),
            "adjoint": adjoint,
            "use_floats": self.use_floats,
        }
        return Statement(name, params, wires, **stmt_options)

    def opstmt(self, args):
        """Operator statements defined inside an operator definition.

        Returns:
            OperatorStmt: object containing statement data
        """
        pref = simplify_math(args[0])
        term = args[1]
        return OperatorStmt(pref, term, use_floats=self.use_floats)

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
        if len(args) == 3:
            name, params, wires = args[0], args[1][1], args[2][1]
        else:
            name, wires = args[0], args[1][1]
            params = []

        decl = Declaration(name, params, wires, declaration_type="gate")
        self._program._declarations["gate"].append(decl)

    def operator_decl(self, args):
        if len(args) == 3:
            name, params, wires = args[0], args[1][1], args[2][1]
        else:
            name, wires = args[0], args[1][1]
            params = []
        decl = Declaration(name, params, wires, declaration_type="operator")
        self._program._declarations["operator"].append(decl)

    def func_decl(self, args):
        if len(args) == 2:
            name, params = args[0], args[1][1]
        else:
            name = args[0]
            params = []
        decl = Declaration(name, params, (), declaration_type="function")
        self._program._declarations["func"].append(decl)

    def output_decl(self, args):
        if len(args) == 3:
            name, params, wires = args[0], args[1][1], args[2][1]
        else:
            name, wires = args[0], args[1][1]
            params = []
        decl = Declaration(name, params, wires, declaration_type="output")
        self._program._declarations["output"].append(decl)

    #########
    # maths
    #########

    def mathop(self, args):
        self._program.add_called_function(args[0])
        return str(args[0]) + "(" + str(args[1]) + ")"

    def add(self, args):
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] + args[1]
        return "(" + " + ".join([str(i) for i in args]) + ")"

    def sub(self, args):
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] - args[1]
        return "(" + " - ".join([str(i) for i in args]) + ")"

    def prod(self, args):
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] * args[1]
        return " * ".join([str(i) for i in args])

    def div(self, args):
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            # if numerator and denominator are ints, then cast numerator to
            # Decimal so that no floats are being returned
            if all(isinstance(a, int) for a in args):
                return Decimal(args[0]) / args[1]
            return args[0] / args[1]
        return " / ".join([str(i) for i in args])

    def neg(self, args):
        if isinstance(args[0], (int, Decimal, DecimalComplex)):
            return -args[0]
        return "-" + str(args[0])

    def PI(self, _):
        return "PI" if not self._eval_pi else Decimal(str(math.pi))


def is_wire(arg):
    """Returns whether the passed argument is a tuple of wires."""
    return isinstance(arg, tuple) and arg[0] == "wires"


def is_param(arg):
    """Returns whether the passed argument is a list of params."""
    return isinstance(arg, tuple) and arg[0] == "params"
