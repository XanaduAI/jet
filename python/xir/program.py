import re
import warnings
from decimal import Decimal
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from .decimal_complex import DecimalComplex
from .utils import strip

"""This module contains the XIRProgram class and classes for the Xanadu IR"""

Wire = Union[int, str]
Param = Union[complex, str, Decimal, DecimalComplex, bool, List["Param"]]
Params = Union[List[Param], Dict[str, Param]]


def get_floats(params: Params) -> Params:
    """Converts `decimal.Decimal` and `DecimalComplex` objects to ``float`` and
    ``complex`` respectively"""
    if isinstance(params, List):
        params_with_floats = []
        for p in params:
            if isinstance(p, DecimalComplex):
                params_with_floats.append(complex(p))
            elif isinstance(p, Decimal):
                params_with_floats.append(float(p))
            else:
                params_with_floats.append(p)

    elif isinstance(params, Dict):
        params_with_floats = dict()
        for k, v in params.items():
            if isinstance(v, DecimalComplex):
                params_with_floats[k] = complex(v)
            elif isinstance(v, Decimal):
                params_with_floats[k] = float(v)
            else:
                params_with_floats[k] = v

    return params_with_floats


class Statement:
    """A general statement consisting of a name, optional parameters and wires

    This is used for gate statements (e.g. ``rx(0.13) | [0]``) or output statements
    (e.g. ``sample(shots: 1000) | [0, 1]``).

    Args:
        name (str): name of the statement
        params (list, Dict): parameters for the statement (can be empty)
        wires (tuple): the wires on which the statement is applied

    Keyword args:
        adjoint (bool): whether the statement is an adjoint gate
        ctrl_wires (tuple): the control wires of a controlled gate statement
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(
            self,
            name: str,
            params: Params,
            wires: Sequence[Wire],
            **kwargs
        ):
        self._name = name
        self._params = params
        self._wires = wires

        self._is_adjoint = kwargs.get("adjoint", False)
        self._ctrl_wires = kwargs.get("ctrl_wires", tuple())

        self._use_floats = kwargs.get("use_floats", True)

    def __str__(self):
        """Serialized string representation of a Statement"""
        if isinstance(self.params, dict):
            params = [f"{k}: {v}" for k, v in self.params.items()]
        else:
            params = [str(p) for p in self.params]
        params_str = ", ".join(params)
        if params_str != "":
            params_str = "(" + params_str + ")"

        wires = ", ".join([str(w) for w in self.wires])

        modifier_str = ""
        if len(self.ctrl_wires) != 0:
            ctrl_wires = ", ".join([str(w) for w in self.ctrl_wires])
            modifier_str = f"ctrl[{ctrl_wires}] "
        if self.is_adjoint:
            modifier_str += "adjoint "

        return f"{modifier_str}{self.name}{params_str} | [{wires}]"

    @property
    def name(self) -> str:
        """Returns the name of the gate statement"""
        return self._name

    @property
    def params(self) -> Params:
        """Returns the parameters of the gate statement"""
        if self.use_floats:
            return get_floats(self._params)
        return self._params

    @property
    def wires(self) -> Sequence[Wire]:
        """Returns the wires that the gate is applied to"""
        return self._wires

    @property
    def is_adjoint(self) -> bool:
        """Returns whether the statement applies an adjoint gate"""
        return self._is_adjoint

    @property
    def ctrl_wires(self) -> Sequence[Wire]:
        """Returns the control wires of a controlled gate statement.
        If no control wires are specified, an empty tuple is returned.
        """
        return self._ctrl_wires

    @property
    def use_floats(self) -> bool:
        """Returns whether floats and complex types are returned instead of
        ``Decimal`` and ``DecimalComplex`` objects, respectively.
        """
        return self._use_floats


class OperatorStmt:
    """Operator statements to be used in operator definitions

    Args:
        pref (Decimal, int, str): prefactor to the operator terms
        terms (list): list of operators and the wire(s) they are applied to
    """

    def __init__(self, pref: Union[Decimal, int, str], terms: List, use_floats: bool = True):
        self._pref = pref
        self._terms = terms

        self._use_floats = use_floats

    def __str__(self):
        terms = [f"{t[0]}[{t[1]}]" for t in self.terms]
        terms_as_string = " @ ".join(terms)
        pref = str(self.pref)

        return f"{pref}, {terms_as_string}"

    @property
    def pref(self) -> Union[Decimal, float, int, str]:
        if isinstance(self._pref, Decimal) and self.use_floats:
            return float(self._pref)
        return self._pref

    @property
    def terms(self) -> List:
        return self._terms

    @property
    def use_floats(self) -> bool:
        return self._use_floats

    @property
    def wires(self) -> Tuple:
        return tuple({t[1] for t in self.terms})


def _serialize_declaration(name, params, wires, declaration_type):
    """TODO"""
    if len(params):
        params = "(" + ", ".join(map(str, params)) + ")"
    else:
        params = ""
    if wires != ():
        wires = "[" + ", ".join([str(w) for w in wires]) + "]"
    else:
        wires = ""

    return f"{declaration_type} {name}{params}{wires}"


class Declaration:
    """General declaration for declaring operators, gates, functions and outputs

    Args:
        name (str): name of the declaration
    """

    def __init__(self, name: str, params: Sequence[str], wires: Sequence[Wire], declaration_type: str):
        self.name = name
        self.params = list(map(str, params))
        if len(set(self.params)) != len(self.params):
            raise ValueError("All parameters in a declaration must be unique.")
        self.wires = wires
        if declaration_type not in ("gate", "output", "operator", "function"):
            raise TypeError(f"Declaration type '{declaration_type}' is invalid.")
        self.declaration_type = declaration_type

    def __str__(self) -> str:
        return _serialize_declaration(self.name, self.params, self.wires, self.declaration_type)

    def __repr__(self) -> str:
        return f"<{self.declaration_type.capitalize()} declaration:name={self.name}>"


class XIRProgram:
    """Main XIR program containing all parsed information

    Args:
        version (str): Version number of the program. Must follow SemVer style (MAJOR.MINOR.PATCH).
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(self, version: str = "0.1.0", use_floats: bool = True):
        if not isinstance(version, str):
            raise TypeError(f"Invalid version number input. Must be a string.")

        valid_match = re.match(r"^\d+\.\d+\.\d+$", version)
        if valid_match is None or valid_match.string != version:
            raise ValueError(
                f"Invalid version number {version} input. Must be SemVer style (MAJOR.MINOR.PATCH)."
            )
        self._version = version
        self._use_floats = use_floats

        self._include = []
        self._options = dict()
        self._statements = []

        self._declarations = {"gate": [], "func": [], "output": [], "operator": []}

        self._gates = dict()
        self._operators = dict()
        self._variables = set()

        self._called_ops = set()

    def __repr__(self) -> str:
        return f"<XIRProgram: version={self._version}>"

    @property
    def version(self) -> str:
        """Version number of the program

        Returns:
            str: program version number
        """
        return self._version

    @property
    def use_floats(self) -> bool:
        return self._use_floats

    @property
    def wires(self) -> Set[int]:
        """Get the wires of an XIR program"""
        wires = []
        for stmt in self.statements:
            wires.extend(stmt.wires)

        return set(wires)

    @property
    def include(self) -> List[str]:
        """Included XIR libraries/files used in the program

        Returns:
            list[str]: included libraries/files
        """
        return self._include

    @property
    def options(self) -> Dict[str, Any]:
        """Script-level options declared in the program

        Returns:
            Dict: declared scipt-level options
        """
        if self.use_floats:
            options_with_floats = get_floats(self._options)
            if isinstance(options_with_floats, Dict):
                return options_with_floats
        return self._options

    @property
    def statements(self) -> List[Statement]:
        """Statements in the program

        Returns:
            list[Statement]: a list of all statements
        """
        return self._statements

    @property
    def declarations(self) -> Dict[str, List]:
        """Declarations in the program

        Returns:
            dict[str, list]: a dictionary of all declarations sorted into the following keys:
            'gate', 'func', 'output' and 'operator'
        """
        return self._declarations

    @property
    def gates(self) -> Dict[str, Dict[str, Sequence]]:
        """All user-defined gates in the program

        Returns:
            dict[str, dict]: a dictionary of all user-defined gates, each gate consisting of a name
            as well as a dictionary containing parameters, wires and statements
        """
        return self._gates

    @property
    def operators(self) -> Dict[str, Dict[str, Sequence]]:
        """All user-defined operators in the program

        Returns:
            dict[str, dict]: a dictionary of all user-defined operators, each operator consisting of
            a name as well as a dictionary containing parameters, wires and statements
        """
        return self._operators

    @property
    def variables(self) -> Set[str]:
        """Free parameter variables used when defining gates and operators

        Returns:
            set[str]: all variables as strings contained in a set
        """
        return self._variables

    @property
    def called_ops(self) -> Set[str]:
        """Functions that are called at any point inside the script

        Returns:
            set[str]: all functions as strings contained in a set
        """
        return self._called_ops

    def add_gate(self, name: str, params: List[str], wires: Tuple, statements: List[Statement]):
        """Adds a gate to the program

        Args:
            name (str): name of the gate
            params (str): parameters used in the gate
            wires (str): wires that the gate is applied to
            statements (list[Statement]): statements that the gate applies
        """
        if name in self._gates:
            warnings.warn("Gate already defined. Replacing old definition with new definiton.")
        self._gates[name] = {"params": params, "wires": wires, "statements": statements}

    def add_operator(
        self, name: str, params: List[str], wires: Tuple, statements: List[OperatorStmt]
    ):
        """Adds an operator to the program

        Args:
            name (str): name of the operator
            params (str): parameters used in the operator
            wires (str): wires that the operator is applied to
            statements (list[OperatorStmt]): statements that the operator applies
        """
        if name in self._operators:
            warnings.warn("Operator already defined. Replacing old definition with new definiton.")
        self._operators[name] = {
            "params": params,
            "wires": wires,
            "statements": statements,
        }

    def serialize(self, minimize: bool = False) -> str:
        """Serialize an XIRProgram returning an XIR script

        Args:
            minimize (bool): whether to strip whitespace and newlines from file

        Returns:
            str: the serialized IR script
        """
        res = []
        res.extend([f"use {use};" for use in self._include])
        if len(self._include) != 0:
            res.append("")

        if len(self.options) > 0:
            res.append("options:")
            res.extend([f"    {k}: {v};" for k, v in self.options.items()])
            res.append("end;\n")

        res.extend([f"{dec};" for v in self._declarations.values() for dec in v ])
        if any(len(dec) != 0 for dec in self._declarations.values()):
            res.append("")

        for name, gate in self._gates.items():
            decl_str = _serialize_declaration(name, gate["params"], gate["wires"], "gate")

            res.append(decl_str + ":")

            res.extend([f"    {stmt};" for stmt in gate["statements"]])
            res.append("end;\n")

        for name, op in self._operators.items():
            decl_str = _serialize_declaration(name, op["params"], op["wires"], "operator")

            res.append(decl_str + ":")

            res.extend([f"    {stmt};" for stmt in op["statements"]])
            res.append("end;\n")

        res.extend([f"{str(stmt)};" for stmt in self._statements])

        res_script = "\n".join(res).strip()
        if minimize:
            return strip(res_script)
        return res_script
