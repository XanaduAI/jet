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

from decimal import Decimal
import strawberryfields as sf
from strawberryfields import ops
from xir.program import IRProgram, Statement, GateDeclaration, OutputDeclaration


def find_number_of_modes(xir):
    """Helper function to find the number of modes in an XIR program"""
    wires = set()
    for stmt in xir.statements:
        wires.add(stmt.wires)

    return len(wires)


def to_program(xir, **kwargs):
    """Convert an IR Program to a Strawberry Fields Program.

    Args:
        xir (IRProgram): the input XIR program object

    Kwargs:
        name (str): name of the resulting Strawberry Fields program
        target (str): Name of the target hardware device. Default is "xir".
        shots (int): number of times the program measurement evaluation is repeated
        cutoff_dim (int): the Fock basis truncation size

    Returns:
        Program: corresponding Strawberry Fields program
    """
    num_of_modes = find_number_of_modes(xir)
    name = kwargs.get("name", "xir")
    if num_of_modes == 0:
        raise ValueError(
            "The XIR program is empty and cannot be transformed into a Strawberry Fields program"
        )
    prog = sf.Program(num_of_modes, name=name)

    # append the quantum operations
    with prog.context as q:
        for op in xir.statements:
            # check if operation name is in the list of
            # defined StrawberryFields operations.
            # This is used by checking against the ops.py __all__
            # module attribute, which contains the names
            # of all defined quantum operations
            if op.name in ops.__all__:
                # get the quantum operation from the sf.ops module
                gate = getattr(ops, op.name)
            else:
                raise NameError(f"Quantum operation {op.name!r} not defined!")

            # create the list of regrefs
            regrefs = [q[i] for i in op.wires]

            if op.params != []:
                # convert symbolic expressions to symbolic expressions containing the corresponding
                # MeasuredParameter and FreeParameter instances.
                if isinstance(op.params, dict):
                    vals = sf.parameters.par_convert(op.params.values(), prog)
                    params = dict(zip(op.params.keys(), vals))
                    gate(**params) | regrefs  # pylint:disable=expression-not-assigned
                else:
                    for i, p in enumerate(op.params):
                        if isinstance(p, Decimal):
                            op.params[i] = float(p)
                    params = sf.parameters.par_convert(op.params, prog)
                    gate(*params) | regrefs  # pylint:disable=expression-not-assigned
            else:
                gate | regrefs  # pylint:disable=expression-not-assigned,pointless-statement

    prog._target = kwargs.get("target")

    if kwargs.get("shots") is not None:
        prog.run_options["shots"] = kwargs.get("shots")

    if kwargs.get("cutoff_dim") is not None:
        prog.backend_options["cutoff_dim"] = kwargs.get("cutoff_dim")

    return prog


def to_xir(prog, **kwargs):
    """Convert a Strawberry Fields Program to an IR Program.

    Args:
        prog (Program): the Strawberry Fields program

    Kwargs:
        add_decl (bool): Whether gate and output declarations should be added to
            the IR program. Default is False.
        version (str): Version number for the program. Default is 0.1.0.

    Returns:
        IRProgram
    """
    version = kwargs.get("version", "0.1.0")
    xir = IRProgram(version=version)

    # fill in the quantum circuit
    for cmd in prog.circuit:

        name = cmd.op.__class__.__name__
        wires = tuple(i.ind for i in cmd.reg)

        if "Measure" in name:
            if kwargs.get("add_decl", False):
                output_decl = OutputDeclaration(name)
                xir._declarations["output"].append(output_decl)

            params = dict()
            # special case to take into account 'select' keyword argument
            if cmd.op.select is not None:
                params["select"] = cmd.op.select

            if cmd.op.p:
                # argument is quadrature phase
                params["phi"] = cmd.op.p[0]

            if name == "MeasureFock":
                # special case to take into account 'dark_counts' keyword argument
                if cmd.op.dark_counts is not None:
                    params["dark_counts"] = cmd.op.dark_counts
        else:
            if kwargs.get("add_decl", False):
                if name not in [gdecl.name for gdecl in xir._declarations["gate"]]:
                    gate_decl = GateDeclaration(name, len(cmd.op.p), len(wires))
                    xir._declarations["gate"].append(gate_decl)

            params = []
            for a in cmd.op.p:
                if sf.parameters.par_is_symbolic(a):
                    # SymPy object, convert to string
                    a = str(a)
                params.append(a)

        op = Statement(name, params, wires)
        xir._statements.append(op)

    return xir