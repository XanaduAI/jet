"""Unit tests for the utils module"""

import pytest

from xir.utils import is_equal

script_1 = """
gate Sgate(a, b)[0];
gate BSgate(theta, phi)[0, 1];
gate Rgate(p0)[a];
output MeasureHomodyne[0];

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
gate Sgate(a, b)[0];
gate Rgate(p0)[a];
output MeasureHomodyne[0];

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
