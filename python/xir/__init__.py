from .parser import XIRTransformer, xir_parser
from .program import XIRProgram


def parse_script(circuit: str, **kwargs) -> XIRProgram:
    """Parse and transform a circuit XIR script and return an XIRProgram."""
    tree = xir_parser.parse(circuit)
    return XIRTransformer(**kwargs).transform(tree)
