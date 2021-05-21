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

"""Unit test for the utils module"""

import pytest
from xir.utils import is_equal

script_1 = """
gate Sgate, 2, 1;
gate BSgate, 2, 2;
gate Rgate, 1, 1;
output MeasureHomodyne;

Sgate(0.7, 0) | [1];
BSgate(0.1, 0.0) | [0, 1];
Rgate(0.2) | [1];

MeasureHomodyne(phi: 3) | [0];
"""

script_1_no_decl = """
use xstd;

Sgate(0.7, 0) | [1];
BSgate(0.1, 0.0) | [0, 1];
Rgate(0.2) | [1];

MeasureHomodyne(phi: 3) | [0];
"""

script_2 = """
gate Sgate, 2, 1;
gate Rgate, 1, 1;
output MeasureFock;

Sgate(0.7, 0) | [1];
Rgate(0.2) | [1];

MeasureFock | [0];
"""


@pytest.mark.parametrize("check_decl", [True, False])
def test_is_equal_same_script(check_decl):
    """Tests the equality of the same script"""
    assert is_equal(script_1, script_1, check_decl=check_decl)


def test_is_equal_with_decl():
    """Tests the ``check_decl`` bool parameter"""
    assert not is_equal(script_1_no_decl, script_1, check_decl=True)
    assert is_equal(script_1_no_decl, script_1, check_decl=False)


@pytest.mark.parametrize("check_decl", [True, False])
def test_is_equal_different_scripts(check_decl):
    """Tests that ``is_equal`` run with two different scripts returns false"""
    assert not is_equal(script_1, script_2, check_decl=check_decl)
