import pennylane as qml
from xir.program import GateDeclaration, OutputDeclaration, Statement, XIRProgram, OperatorStmt

def to_xir(circuit, *args, **kwargs) -> XIRProgram:
    """Convert a PennyLane QNode to an XIR Program.

    Args:
        circuit (pennylane.QNode): the PennyLane QNode

    Kwargs:
        add_decl (bool): Whether gate and output declarations should be added to
            the IR program. Default is ``False``.
        version (str): Version number for the program. Default is ``0.1.0``.

    Returns:
        XIRProgram
    """
    if circuit.qtape is None:
        if args is not None or kwargs is not None:
            circuit.construct(args, kwargs)
        else:
            raise ValueError("Circuit must have been executed or parameters must be passed.")

    version = kwargs.get("version", "0.1.0")
    xir = XIRProgram(version=version)

    for op in circuit.qtape.operations:
        if kwargs.get("add_decl", False):
            if op.name.lower() not in [gdecl.name for gdecl in xir._declarations["gate"]]:
                gate_decl = GateDeclaration(op.name.lower(), len(op.parameters), len(op.wires))
                xir._declarations["gate"].append(gate_decl)

        stmt = Statement(op.name.lower(), op.parameters, tuple(op.wires))
        xir.statements.append(stmt)

    for i, ob in enumerate(circuit.qtape.observables):

        if kwargs.get("add_decl", False):
            output_decl = OutputDeclaration(ob.return_type.name.lower())
            xir._declarations["output"].append(output_decl)

        operator_stmts = []
        if isinstance(ob.name, list):
            operator_stmts.append(OperatorStmt(1, [(n.lower(), w) for n, w in zip(ob.name, ob.wires)]))
        else:
            operator_stmts.append(OperatorStmt(1, [(ob.name.lower(), ob.wires[0])]))

        ob_name = "op_" + str(i + 1)
        xir.add_operator(ob_name, [], ob.wires, operator_stmts)

        output_stmt = Statement(ob.return_type.name.lower(), {"operator": ob_name}, tuple(op.wires))
        xir.statements.append(output_stmt)

    return xir
