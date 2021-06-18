import re
import warnings
from decimal import Decimal
from typing import Union, List, Dict, Set, Tuple, Sequence

from .utils import strip
from .decimal_complex import DecimalComplex

"""This module contains the XIRProgram class and classes for the Xanadu IR"""


def get_floats(params: Union[List, Dict]) -> Union[List, Dict]:
    params_with_floats = params.copy()

    if isinstance(params, List):
        for i, p in enumerate(params_with_floats):
            if isinstance(p, DecimalComplex):
                params_with_floats[i] = complex(p)
            if isinstance(p, Decimal):
                params_with_floats[i] = float(p)

    elif isinstance(params, Dict):
        for k, v in params_with_floats.items():
            if isinstance(v, DecimalComplex):
                params_with_floats[k] = complex(v)
            if isinstance(v, Decimal):
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
    """Main XIR program containing all parsed information

    Args:
        version (str): Version number of the program. Must follow SemVer style (MAJOR.MINOR.PATCH).
    """

    def __init__(self, version: str = "0.1.0"):
        if not isinstance(version, str):
            raise TypeError(f"Invalid version number input. Must be a string.")

        valid_match = re.match(r"^\d+\.\d+\.\d+$", version)
        if valid_match is None or valid_match.string != version:
            raise ValueError(
                f"Invalid version number {version} input. Must be SemVer style (MAJOR.MINOR.PATCH)."
            )
        self._version = version

        self._include = []
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
    def include(self) -> List[str]:
        """Included XIR libraries/files used in the program

        Returns:
            list[str]: included libraries/files
        """
        return self._include

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

    def add_gate(
        self, name: str, params: List[str], wires: Tuple, statements: List[Statement]
    ):
        """Adds a gate to the program

        Args:
            name (str): name of the gate
            params (str): parameters used in the gate
            wires (str): wires that the gate is applied to
            statements (list[Statement]): statements that the gate applies
        """
        if name in self._gates:
            warnings.warn(
                "Gate already defined. Replacing old definition with new definiton."
            )
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
            warnings.warn(
                "Operator already defined. Replacing old definition with new definiton."
            )
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
