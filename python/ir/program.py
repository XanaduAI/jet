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

import re
import warnings
from decimal import Decimal
from typing import Union, List, Dict, Set, Tuple, Sequence

from .utils import strip

"""This module contains the IRProgram class and classes for the Xanadu IR"""


class Statement:
    """TODO"""

    def __init__(self, name: str, params: Union[List, Dict], wires: Tuple):
        self.name = name
        self.params = params
        self.wires = wires

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


class OperatorStmt:
    """TODO"""

    def __init__(self, pref: Union[Decimal, int, str], terms: List):
        self.pref = pref
        self.terms = terms

    def __str__(self):
        terms = [f"{t[0]}[{t[1]}]" for t in self.terms]
        terms_as_string = " @ ".join(terms)
        pref = str(self.pref)

        return f"{pref}, {terms_as_string}"

class Declaration:
    """TODO"""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.name}"


class OperatorDeclaration(Declaration):
    """Quantum operator declarations"""

    def __init__(self, name: str, num_params: int, num_wires: int):
        self.num_params = num_params
        self.num_wires = num_wires

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}, {self.num_wires}"


class GateDeclaration(Declaration):
    """Quantum gate declarations"""

    def __init__(self, name: str, num_params: int, num_wires: int):
        self.num_params = num_params
        self.num_wires = num_wires

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}, {self.num_wires}"


class FuncDeclaration(Declaration):
    """Function declarations"""

    def __init__(self, name: str, num_params: int):
        self.num_params = num_params

        super().__init__(name)

    def __str__(self):
        return f"{self.name}, {self.num_params}"


class OutputDeclaration(Declaration):
    """Output declarations"""


class IRProgram:
    """TODO"""

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

        self._declarations = {
            "gate": [],
            "func": [],
            "output": [],
            "operator": []
        }

        self._gates = dict()
        self._operators = dict()
        self._variables = set()

        self._called_ops = set()

    def __repr__(self) -> str:
        """TODO"""
        return f"<IRProgram: version={self._version}>"

    @property
    def version(self) -> str:
        """TODO"""
        return self._version

    @property
    def include(self) -> List[str]:
        """TODO"""
        return self._include

    @property
    def statements(self) -> List[Statement]:
        """TODO"""
        return self._statements

    @property
    def declarations(self) -> Dict[str, List]:
        """TODO"""
        return self._declarations

    @property
    def gates(self) -> Dict[str, Dict[str, Sequence]]:
        """TODO"""
        return self._gates

    @property
    def operators(self) -> Dict[str, Dict[str, Sequence]]:
        """TODO"""
        return self._operators

    @property
    def variables(self) -> Set[str]:
        """TODO"""
        return self._variables

    @property
    def called_ops(self) -> Set[str]:
        """TODO"""
        return self._called_ops

    def add_gate(self, name: str, params: List[str], wires: Tuple, statements: List[Statement]):
        """TODO"""
        if name in self._gates:
            warnings.warn("Gate already defined. Replacing old definition with new definiton.")
        self._gates[name] = {
            "params": params,
            "wires": wires,
            "statements": statements
        }

    def add_operator(self, name: str, params: List[str], wires: Tuple, statements: List[OperatorStmt]):
        """TODO"""
        if name in self._operators:
            warnings.warn("Operator already defined. Replacing old definition with new definiton.")
        self._operators[name] = {
            "params": params,
            "wires": wires,
            "statements": statements
        }

    def serialize(self, minimize: bool = False) -> str:
        """Serialize an IRProgram returning an XIR script

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
