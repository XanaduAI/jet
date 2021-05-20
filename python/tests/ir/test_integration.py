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

"""Integration tests for the IR"""

from ..program import IRProgram
import pytest

from .. import ir_parser, IRTransformer
from ..utils import is_equal


def parse_script(circuit: str) -> IRProgram:
    """Parse and transform a circuit XIR script and return an IRProgram"""
    tree = ir_parser.parse(circuit)
    return IRTransformer().transform(tree)


photonics_script = """
gate Sgate, 2, 1;
gate BSgate, 2, 2;
gate Rgate, 1, 1;
output MeasureHomodyne;

Sgate(0.7, 0) | [1];
BSgate(0.1, 0.0) | [0, 1];
Rgate(0.2) | [1];

MeasureHomodyne(phi: 3) | [0];
"""

photonics_script_no_decl = """
use xstd;

Sgate(0.7, 0) | [1];
BSgate(0.1, 0.0) | [0, 1];
Rgate(0.2) | [1];

MeasureHomodyne(phi: 3) | [0];
"""

qubit_script = """
// this file works with the current parser
use xstd;

gate h(a)[0, 1]:
    rz(-2.3932854391951004) | [0];
    rz(a) | [1];
    // rz(pi / sin(3 * 4 / 2 - 2)) | [a, 2];
end;

operator o(a):
    0.7 * sin(a), X[0] @ Z[1];
    -1.6, X[0];
    2.45, Y[0] @ X[1];
end;

g_one(pi) | [0, 1];
g_two | [2];
g_three(1, 3.3) | [2];

// The circuit and statistics
ry(1.23) | [0];
rot(0.1, 0.2, 0.3) | [1];
h(0.2) | [0, 1, 2];

sample(observable: o(0.2), shots: 1000) | [0, 1];
"""


class TestParser:
    """Integration tests for parsing, and serializing, XIR scripts"""

    @pytest.mark.parametrize("circuit", [qubit_script, photonics_script, photonics_script_no_decl])
    def test_parse_and_serialize(self, circuit):
        """Test parsing and serializing an XIR script.

        Tests parsing, serializing as well as the ``is_equal`` utils function.
        """
        irprog = parse_script(circuit)
        res = irprog.serialize()
        assert is_equal(res, circuit)
