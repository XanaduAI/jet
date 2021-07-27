"""This module contains the XIRProgram class and classes for the Xanadu IR"""

import re
import warnings
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple, Union

from .decimal_complex import DecimalComplex
from .utils import strip

Wire = Union[int, str]


def get_floats(params: Union[List, Dict]) -> Union[List, Dict]:
    """Converts `decimal.Decimal` and `DecimalComplex` objects to ``float`` and
    ``complex`` respectively"""
    params_with_floats = params.copy()

    if isinstance(params, List):
        for i, p in enumerate(params_with_floats):
            if isinstance(p, DecimalComplex):
                params_with_floats[i] = complex(p)
            elif isinstance(p, Decimal):
                params_with_floats[i] = float(p)

    elif isinstance(params, Dict):
        for k, v in params_with_floats.items():
            if isinstance(v, DecimalComplex):
                params_with_floats[k] = complex(v)
            elif isinstance(v, Decimal):
                params_with_floats[k] = float(v)

    return params_with_floats


class Statement:
    """A general statement consisting of a name, optional parameters and wires

    This is used for gate statements (e.g. ``rx(0.13) | [0]``) or output statements
    (e.g. ``sample(shots: 1000) | [0, 1]``).

    Args:
        name (str): name of the statement
        params (list, Dict): parameters for the statement (can be empty)
        wires (tuple): the wires on which the statement is applied
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(self, name: str, params: Union[List, Dict], wires: Tuple, use_floats: bool = True):
        self._name = name
        self._params = params
        self._wires = wires

        self._use_floats = use_floats

    def __str__(self):
        if isinstance(self.params, dict):
            params = [f"{k}: {v}" for k, v in self.params.items()]
        else:
            params = [str(p) for p in self.params]
        params_str = ", ".join(params)

        wires = ", ".join([str(w) for w in self.wires])

        if params_str == "":
            return f"{self.name} | [{wires}]"
        return f"{self.name}({params_str}) | [{wires}]"

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Union[List, Dict]:
        if self.use_floats:
            return get_floats(self._params)
        return self._params

    @property
    def wires(self) -> Tuple:
        return self._wires

    @property
    def use_floats(self) -> bool:
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


class Declaration:
    """General declaration for declaring operators, gates, functions and outputs

    Args:
        name (str): name of the declaration
    """

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.name}"


class OperatorDeclaration(Declaration):
    """Quantum operator declarations

    Args:
        name (str): name of the operator
        num_params (int): number of parameters that the operator uses
        num_wires (int): number of wires that the operator is applied to
    """

    def __init__(self, name: str, num_params: int, num_wires: int):
        self.num_params = num_params
        self.num_wires = num_wires

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}, {self.num_wires}"


class GateDeclaration(Declaration):
    """Quantum gate declarations

    Args:
        name (str): name of the gate
        num_params (int): number of parameters that the gate uses
        num_wires (int): number of wires that the gate is applied to
    """

    def __init__(self, name: str, num_params: int, num_wires: int):
        self.num_params = num_params
        self.num_wires = num_wires

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}, {self.num_wires}"


class FuncDeclaration(Declaration):
    """Function declarations

    Args:
        name (str): name of the function
        num_params (int): number of parameters that the function uses
    """

    def __init__(self, name: str, num_params: int):
        self.num_params = num_params

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}"


class OutputDeclaration(Declaration):
    """Output declarations

    Args:
        name (str): name of the output declaration
    """


class XIRProgram:
    """Structured representation of an XIR program.

    Args:
        version (str): Version number of the program. Must follow SemVer style (MAJOR.MINOR.PATCH).
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(self, version: str = "0.1.0", use_floats: bool = True):
        XIRProgram._validate_version(version)

        self._version = version
        self._use_floats = use_floats

        self._includes = []
        self._options = {}
        self._statements = []

        self._declarations = {key: [] for key in ("gate", "func", "output", "operator")}

        self._gates = {}
        self._operators = {}

        self._variables = set()
        self._called_functions = set()

    def __repr__(self) -> str:
        """Returns a string representation of the XIR program."""
        return f"<XIRProgram: version={self._version}>"

    @property
    def version(self) -> str:
        """Returns the version number of the XIR program.

        Returns:
            str: version number
        """
        return self._version

    @property
    def use_floats(self) -> bool:
        """Returns whether floats and complex types are used by the XIR program.

        Returns:
            bool: whether floats and complex types are used
        """
        return self._use_floats

    @property
    def wires(self) -> Iterator[Wire]:
        """Returns the wires of the XIR program.

        Returns:
            Iterator[Wire]: iterator over the wires
        """
        wires = []
        for stmt in self.statements:
            wires.extend(stmt.wires)

        return iter(set(wires))

    @property
    def called_functions(self) -> Iterator[str]:
        """Return the functions that are called in the XIR program.

        Returns:
            Iterator[str]: functions as strings
        """
        return iter(self._called_functions)

    @property
    def declarations(self) -> Mapping[str, List[Declaration]]:
        """Returns the declarations in the XIR program.

        Returns:
            Mapping[str, List[Declaration]]: dictionary of declarations sorted
                into the following keys: 'gate', 'func', 'output' and 'operator'.
        """
        return self._declarations

    @property
    def gates(self) -> Mapping[str, Mapping[str, Sequence]]:
        """Returns the gates in the XIR program.

        Returns:
            Mapping[str, Mapping[str, Sequence]]: dictionary of gates, each gate
                consisting of a name and a dictionary with the following keys:
                'parameters', 'wires' and 'statements'
        """
        return self._gates

    @property
    def includes(self) -> Iterator[str]:
        """Returns the included XIR modules used by the XIR program.

        Returns:
            Iterator[str]: iterator over the included XIR modules
        """
        return iter(self._includes)

    @property
    def operators(self) -> Mapping[str, Mapping[str, Sequence]]:
        """Returns the operators in the XIR program.

        Returns:
            Mapping[str, Mapping[str, Sequence]]: dictionary of operators, each
                operator consisting of a name and a dictionary with the following
                keys: 'parameters', 'wires' and 'statements'
        """
        return self._operators

    @property
    def options(self) -> Mapping[str, Any]:
        """Returns the script-level options declared in the XIR program.

        Returns:
            Mapping[str, Any]: declared scipt-level options
        """
        return get_floats(self._options) if self.use_floats else self._options

    @property
    def statements(self) -> Iterator[Statement]:
        """Returns the statements in the XIR program.

        Returns:
            Iterator[Statement]: iterator over the statements
        """
        return iter(self._statements)

    @property
    def variables(self) -> Iterator[str]:
        """Returns the free parameter variables used when defining gates and
        operators in the XIR program.

        Returns:
            Iterator[str]: free parameter variables as strings
        """
        return iter(self._variables)

    def add_called_function(self, name: str) -> None:
        """Adds the name of a called function to the XIR program.

        Args:
            name (str): name of the function
        """
        self._called_functions.add(name)

    def add_declaration(self, key: str, decl: Declaration) -> None:
        """Adds a declaration to the XIR program.

        Args:
            key (str): key of the declaration ('func', 'gate', 'operator', or 'output')
            decl (Declaration): the declaration
        """
        if key not in self._declarations:
            raise KeyError(f"Key '{key}' is not a supported declaration.")

        if decl.name in (decl.name for decl in self._declarations[key]):
            warnings.warn(f"{key.title()} '{decl.name}' has already been declared.")

        self._declarations[key].append(decl)

    def add_gate(
        self,
        name: str,
        params: Sequence[str],
        wires: Sequence[Wire],
        statements: Sequence[Statement],
    ) -> None:
        """Adds a gate to the XIR program.

        Args:
            name (str): name of the gate
            params (Sequence[str]): parameters used in the gate
            wires (Sequence[Wire]): wires that the gate is applied to
            statements (Sequence[Statement]): statements that the gate applies
        """
        if name in self._gates:
            warnings.warn(
                f"Gate '{name}' already defined. Replacing old definition with new definition."
            )

        self._gates[name] = {"params": params, "wires": wires, "statements": statements}

    def add_include(self, include: str) -> None:
        """Adds an included XIR module to the XIR program.

        Args:
            include (str): name of the XIR module
        """
        if include in self._includes:
            warnings.warn(f"Module '{include}' is already included. Skipping include.")
            return None

        self._includes.append(include)

    def add_operator(
        self,
        name: str,
        params: Sequence[str],
        wires: Sequence[Wire],
        statements: Sequence[OperatorStmt],
    ) -> None:
        """Adds an operator to the XIR program.

        Args:
            name (str): name of the operator
            params (Sequence[str]): parameters used in the operator
            wires (Sequence[Wire]): wires that the operator is applied to
            statements (Sequence[OperatorStmt]): statements that the operator applies
        """
        if name in self._operators:
            warnings.warn(
                f"Operator '{name}' already defined. Replacing old definition with new definition."
            )

        self._operators[name] = {"params": params, "wires": wires, "statements": statements}

    def add_option(self, name: str, value: Any) -> None:
        """Adds an option to the XIR program.

        Args:
            name (str): name of the option
            value (Any): value of the option
        """
        if name in self._options:
            warnings.warn(f"Option '{name}' already set. Replacing old value with new value.")

        self._options[name] = value

    def add_statement(self, statement: Statement) -> None:
        """Adds a statement to the XIR program.

        Args:
            statement (Statement): the statement
        """
        self._statements.append(statement)

    def add_variable(self, name: str) -> None:
        """Adds the name of a free parameter variable to the XIR program.

        Args:
            name (str): name of the variable
        """
        self._variables.add(name)

    def serialize(self, minimize: bool = False) -> str:
        """Serialize an ``XIRProgram`` to an XIR script.

        Args:
            minimize (bool): whether to strip whitespace and newlines from file

        Returns:
            str: the serialized IR script
        """
        res = []
        res.extend([f"use {use};" for use in self._includes])
        if len(self._includes) != 0:
            res.append("")

        if len(self.options) > 0:
            res.append("options:")
            res.extend([f"    {k}: {v};" for k, v in self.options.items()])
            res.append("end;\n")

        res.extend([f"gate {dec};" for dec in self._declarations["gate"]])
        res.extend([f"func {dec};" for dec in self._declarations["func"]])
        res.extend([f"output {dec};" for dec in self._declarations["output"]])
        res.extend([f"operator {dec};" for dec in self._declarations["operator"]])
        if any(len(dec) != 0 for dec in self._declarations.values()):
            res.append("")

        for name, gate in self._gates.items():
            if gate["params"] != []:
                params = "(" + ", ".join([str(p) for p in gate["params"]]) + ")"
            else:
                params = ""
            if gate["wires"] != ():
                wires = "[" + ", ".join([str(w) for w in gate["wires"]]) + "]"
            else:
                wires = ""

            res.extend([f"gate {name}{params}{wires}:"])

            res.extend([f"    {stmt};" for stmt in gate["statements"]])
            res.append("end;\n")

        for name, op in self._operators.items():
            if op["params"] != []:
                params = "(" + ", ".join([str(p) for p in op["params"]]) + ")"
            else:
                params = ""
            if op["wires"] != ():
                wires = "[" + ", ".join([str(w) for w in op["wires"]]) + "]"
            else:
                wires = ""

            res.extend([f"operator {name}{params}{wires}:"])

            res.extend([f"    {stmt};" for stmt in op["statements"]])
            res.append("end;\n")

        res.extend([f"{str(stmt)};" for stmt in self._statements])

        res_script = "\n".join(res).strip()
        if minimize:
            return strip(res_script)
        return res_script

    @staticmethod
    def _validate_version(version: str) -> None:
        """Validates the given version number.

        Raises:
            TypeError: If the version number is not a string.
            ValueError: If the version number is not a semantic version.
        """
        if not isinstance(version, str):
            raise TypeError(f"Version '{version}' must be a string.")

        valid_match = re.fullmatch(r"\d+\.\d+\.\d+", version)
        if valid_match is None or valid_match.string != version:
            raise ValueError(f"Version '{version}' must be a semantic version (MAJOR.MINOR.PATCH).")
