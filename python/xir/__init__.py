from .parser import XIRTransformer, xir_parser
from .program import XIRProgram

def parse_script(circuit: str, eval_pi=False, use_floats=True) -> XIRProgram:
    """Parse and transform a circuit XIR script and return an XIRProgram"""
    tree = xir_parser.parse(circuit)
    return XIRTransformer(eval_pi=eval_pi, use_floats=use_floats).transform(tree)
