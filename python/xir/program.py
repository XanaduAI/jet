"""This module contains the XIRProgram class and classes for the Xanadu IR"""

import re
import warnings
from decimal import Decimal
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .decimal_complex import DecimalComplex
from .utils import strip

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

    def __init__(self, name: str, params: Params, wires: Sequence[Wire], **kwargs):
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

    def __str__(self) -> str:
        terms = [f"{t[0]}[{t[1]}]" for t in self.terms]
        terms_as_string = " @ ".join(terms)
        pref = str(self.pref)

        return f"{pref}, {terms_as_string}"

    @property
    def pref(self) -> Union[Decimal, float, int, str]:
        """Returns the prefactor of this operator statement."""
        if isinstance(self._pref, Decimal) and self.use_floats:
            return float(self._pref)
        return self._pref

    @property
    def terms(self) -> List:
        """Returns the terms in this operator statement."""
        return self._terms

    @property
    def use_floats(self) -> bool:
        """Returns the float setting of this operator statement."""
        return self._use_floats

    @property
    def wires(self) -> Tuple:
        """Returns the wires this operator statement is applied to."""
        return tuple({t[1] for t in self.terms})


def _serialize_declaration(
    name: str, params: Sequence[Param], wires: Sequence[Wire], declaration_type
) -> str:
    """Constructs and returns a declaration as a serialized string."""
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
        declaration_type (str): The type of declaration. Can be either "gate", "operator", "output"
            or "function".
        params (Sequence[str]): parameters used by the declared object
        wires (Sequence[Wire]): wires that the declared object is applied to
    """

    def __init__(
        self,
        name: str,
        declaration_type: str,
        params: Optional[Sequence[str]] = None,
        wires: Optional[Sequence[Wire]] = None,
    ) -> None:
        if declaration_type not in ("gate", "output", "operator", "function"):
            raise TypeError(f"Declaration type '{declaration_type}' is invalid.")

        self.name = name
        self.declaration_type = declaration_type
        self.params = list(params or [])
        self.wires = tuple(wires or ())

        if not all(isinstance(p, str) for p in self.params):
            raise TypeError("Declaration '{name}' has parameters which are not strings.")
        if len(set(self.params)) != len(self.params):
            raise ValueError("Declaration '{name}' has duplicate parameters.")

    def __str__(self) -> str:
        return _serialize_declaration(self.name, self.params, self.wires, self.declaration_type)

    def __repr__(self) -> str:
        return f"<{self.declaration_type.capitalize()} declaration:name={self.name}>"


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
    def wires(self) -> Collection[Wire]:
        """Returns the wires of the XIR program.

        Returns:
            Collection[Wire]: collection of wires
        """
        wires = []
        for stmt in self.statements:
            wires.extend(stmt.wires)

        return set(wires)

    @property
    def called_functions(self) -> Collection[str]:
        """Returns the functions that are called in the XIR program.

        Returns:
            Collection[str]: collection of function names
        """
        return self._called_functions

    @property
    def declarations(self) -> Mapping[str, Sequence[Declaration]]:
        """Returns the declarations in the XIR program.

        Returns:
            Mapping[str, Sequence[Declaration]]: dictionary of declarations
            with the following keys: 'gate', 'func', 'output', and 'operator'
        """
        return self._declarations

    @property
    def gates(self) -> Mapping[str, Mapping[str, Sequence]]:
        """Returns the gates in the XIR program.

        Returns:
            Mapping[str, Mapping[str, Sequence]]: dictionary of gates, each one
            consisting of a name and a dictionary with the following keys:
            'parameters', 'wires', and 'statements'
        """
        return self._gates

    @property
    def includes(self) -> Sequence[str]:
        """Returns the included XIR modules used by the XIR program.

        Returns:
            Sequence[str]: sequence of included XIR modules
        """
        return self._includes

    @property
    def operators(self) -> Mapping[str, Mapping[str, Sequence]]:
        """Returns the operators in the XIR program.

        Returns:
            Mapping[str, Mapping[str, Sequence]]: dictionary of operators, each
            one consisting of a name and a dictionary with the following keys:
            'parameters', 'wires', and 'statements'
        """
        return self._operators

    @property
    def options(self) -> Mapping[str, Any]:
        """Returns the script-level options declared in the XIR program.

        Returns:
            Mapping[str, Any]: declared scipt-level options
        """
        if self.use_floats:
            options_with_floats = get_floats(self._options)
            # The following condition should always be True. For more context,
            # see https://github.com/XanaduAI/jet/pull/52/files#r681872696.
            if isinstance(options_with_floats, Dict):
                return options_with_floats
        return self._options

    @property
    def statements(self) -> Sequence[Statement]:
        """Returns the statements in the XIR program.

        Returns:
            Sequence[Statement]: sequence of statements
        """
        return self._statements

    @property
    def variables(self) -> Collection[str]:
        """Returns the free parameter variables used when defining gates and
        operators in the XIR program.

        Returns:
            Collection[str]: collection of free parameter variable names
        """
        return self._variables

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
            return

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

    def clear_includes(self) -> None:
        """Clears the includes of an XIR program."""
        self._includes = []

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

        res.extend([f"{decl};" for v in self._declarations.values() for decl in v])
        if any(len(decl) != 0 for decl in self._declarations.values()):
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

    @staticmethod
    def merge(*programs: "XIRProgram") -> "XIRProgram":
        """Merges one or more XIR programs into a new XIR program.

        The merged XIR program is formed by concatenating the given XIR programs
        in the order they are passed to the function. Warnings may be issued for
        duplicate declarations, gates, includes, operators, and options.

        Args:
            programs (XIRProgram): XIR programs to merge

        Returns:
            XIRProgram: the merged XIR program

        Raises:
            ValueError: if no XIR programs are provided or if at least two XIR
                programs have different versions or float settings
        """
        if len(programs) == 0:
            raise ValueError("Merging requires at least one XIR program.")

        version = programs[0].version
        if any(program.version != version for program in programs):
            raise ValueError("XIR programs with different versions cannot be merged.")

        use_floats = any(program.use_floats for program in programs)
        if any(program.use_floats != use_floats for program in programs):
            warnings.warn("XIR programs with different float settings are being merged.")

        result = XIRProgram(version=version, use_floats=use_floats)

        for program in programs:
            for called_function in program.called_functions:
                result.add_called_function(called_function)

            for key, decls in program.declarations.items():
                for decl in decls:
                    result.add_declaration(key, decl)

            for name, gate in program.gates.items():
                result.add_gate(name, **gate)

            for include in program.includes:
                result.add_include(include)

            for name, operator in program.operators.items():
                result.add_operator(name, **operator)

            for name, value in program.options.items():
                result.add_option(name, value)

            for statement in program.statements:
                result.add_statement(statement)

            for variable in program.variables:
                result.add_variable(variable)

        return result

    @staticmethod
    def resolve(library: Mapping[str, "XIRProgram"], name: str) -> "XIRProgram":
        """Resolves the includes of an XIR program using an XIR library.

        Args:
            library (Mapping[str, XIRProgram]): mapping from names to XIR programs
            name (str): name of the root XIR program to resolve

        Returns:
            XIRProgram: XIR program obtained by recursively merging each XIR
            program starting from the root XIR program

        **Example**

        Consider an XIR program ``main`` which includes another XIR program ``util``:

        .. code-block:: python

            import xir

            # Create two simple XIR programs.
            main_program = xir.parse_script("use util; H2 | [0, 1];")
            util_program = xir.parse_script("gate H2, 0, 2; gate H2: H | [0]; H | [1]; end;")

        The ``main`` XIR program can be resolved using

        .. code-block:: python

            # Define a library containing the XIR programs in the resolution scope.
            library = {"main": main_program, "util": util_program}

            # Resolve the imports from the main XIR program.
            resolved_main_program = xir.XIRProgram.resolve(library, "main")

            # Display the result.
            print(resolved_main_program.serialize())

        The output is

        .. code-block:: none

            gate H2:
                H | [0];
                H | [1];
            end;

            H2 | [0, 1];
        """
        resolved_names = XIRProgram._resolve_include_order(library, name, stack=set(), cache={})
        resolved_programs = map(library.get, resolved_names)

        resolved_program = XIRProgram.merge(*resolved_programs)
        resolved_program.clear_includes()
        return resolved_program

    @staticmethod
    def _resolve_include_order(
        library: Mapping[str, "XIRProgram"],
        name: str,
        stack: MutableSet[str],
        cache: MutableMapping[str, Sequence[str]],
    ) -> Sequence[str]:
        """
        Resolves the include order of an XIR program using C3 superclass linearization.

        See https://en.wikipedia.org/wiki/C3_linearization for a summary of the
        properties that hold following the linearization.

        Args:
            library (Mapping[str, XIRProgram]): mapping from names to XIR programs
            name (str): name of the root XIR program to resolve
            stack (MutableSet[str]): names in the current call stack; prevents
                infinite recursion in the case of circular dependencies
            cache (MutableMapping[str, Sequence[str]]): mapping from names to
                linearizations; improves performance when an XIR program is
                included by multiple XIR programs

        Returns:
            Sequence[str]: names representing a C3 linearization of the given XIR program.

        Raises:
            KeyError: if ``library`` does not have an entry for ``name``, or the
                XIR program associated with ``name`` (transitively) includes
                another XIR program which does not have an entry in ``library``
            ValueError: if ``name`` is already in ``stack``, or the XIR program
                associated with ``name`` (transitively) includes an XIR program
                with a circular dependency
        """
        if name not in library:
            raise KeyError(f"XIR program '{name}' cannot be found.")

        if name in stack:
            raise ValueError(f"XIR program '{name}' has a circular dependency.")

        # The stack is only empty when processing the root XIR program.
        if stack and library[name].statements:
            raise ValueError(f"XIR program '{name}' contains a statement.")

        if name in cache:
            return cache[name]

        # Step 1: Generate the C3 linearizations of the included XIR programs.
        stack.add(name)

        linearizations = []
        for include in library[name].includes:
            # Call the list constructor to avoid modifying cached linearizations.
            linearization = list(XIRProgram._resolve_include_order(library, include, stack, cache))
            linearizations.append(linearization)

        stack.discard(name)

        # Step 2: Iteratively select the next XIR program to be included.
        includes = []

        while linearizations:
            tails = {include for include in includes[1:] for includes in linearizations}
            heads = (includes[0] for includes in linearizations)

            selection = next(head for head in heads if head not in tails)
            includes.append(selection)

            # Remove the included XIR program from the linearizations and delete
            # any empty linearizations that are generated as a result.
            for linearization in linearizations:
                if selection in linearization:
                    linearization.remove(selection)
            linearizations = [linearization for linearization in linearizations if linearization]

        # Unlike MRO, the current XIR program has lower precendence than its includes.
        includes.append(name)

        cache[name] = includes

        return includes

    @staticmethod
    def _validate_version(version: str) -> None:
        """Validates the given version number.

        Raises:
            TypeError: if the version number is not a string
            ValueError: if the version number is not a semantic version
        """
        if not isinstance(version, str):
            raise TypeError(f"Version '{version}' must be a string.")

        valid_match = re.fullmatch(r"\d+\.\d+\.\d+", version)
        if valid_match is None:
            raise ValueError(f"Version '{version}' must be a semantic version (MAJOR.MINOR.PATCH).")
