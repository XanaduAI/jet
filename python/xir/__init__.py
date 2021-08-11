from .decimal_complex import DecimalComplex  # noqa: F401
from .parser import XIRTransformer, xir_parser  # noqa: F401
from .program import GateDeclaration, OperatorStmt, Statement, XIRProgram  # noqa: F401


def parse_script(circuit: str, **kwargs) -> XIRProgram:
    """Parse and transform a circuit XIR script and return an XIRProgram."""
    tree = xir_parser.parse(circuit)
    return XIRTransformer(**kwargs).transform(tree)
