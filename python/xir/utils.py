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

"""This module contains the utility functions used when parsing"""

import re
from typing import List, Tuple


def beautify_math(numstr: List[str]) -> List[str]:
    """Simplifies specified substrings and removes unnecessary parantheses

    Used for e.g. statement parameter inputs to beautify mathematical expressions.
    For example, ``((a+2))+-(b*(3+c))`` becomes ``a+2-b*(3+c)``.

    Args:
        numstr (list[str]): string(s) containing mathematical expressions

    Returns:
        list[str]: the input expression(s) beautified
    """
    # list of replacement strings; can be expanded with more if needed
    repstr = [
        ("--", "+"),
        ("+-", "-"),
        ("-+", "+"),
        ("++", "+"),
    ]

    for i, n in enumerate(numstr):
        if not isinstance(n, str):
            continue
        for r in repstr:
            numstr[i] = numstr[i].replace(*r)
        if n[0] == "+":
            numstr[i] = numstr[i][1:]

    numstr = remove_paranth(numstr)
    return numstr


def remove_paranth(numstr: List[str]) -> List[str]:
    """Removes matching parantheses where unnecessary

    For example, ``((a+b)+c)`` becomes ``a+b+c``.

    Args:
        numstr (list[str]): string(s) containing mathematical expressions

    Returns:
        list[str]: the input expressions beautified
    """
    for i, n in enumerate(numstr):
        if not isinstance(n, str):
            continue

        pre = []
        dels = [0, len(n) + 1]

        # expand string with "+" so that outer parantheses can be removed,
        # then store all indices where unnecessary mathing parantheses are found
        n = "+" + n + "+"
        for j, char in enumerate(n):
            if char == "(":
                pre.append((n[j - 1], j))
            if char == ")":
                p = pre.pop()
                if p[0] in ("+", "-", "(") and n[j + 1] in ("+", "-", ")"):
                    dels.extend([p[1], j])

        # delete all unnecessary mathing parantheses found int the previous step
        new_str = ""
        dels = sorted(dels)
        for j in range(len(dels) - 1):
            new_str += n[dels[j] + 1 : dels[j + 1]]
        numstr[i] = new_str

    return numstr


def check_wires(wires: Tuple, stmts: List):
    """Check that declared wires are the same as the wires used in statements

    Args:
        wires (tuple): declared wires for either a gate or operator declaration
        stmts (list[Statement]): statements used in gate or operator declaration

    Raises:
        ValueError: if wires differ from the wires used in the statements
    """
    wires_flat = [i for s in stmts for i in s.wires]
    if set(wires) != set(wires_flat):
        raise ValueError(
            f"Wrong wires supplied. Expected {set(wires)}, got {set(wires_flat)}"
        )


def strip(script: str) -> str:
    """Removes comments, newlines and unnecessary whitespace from script

    Args:
        script (str): the text to be stripped

    Returns:
        str: the stripped text
    """
    # search for any comments ("//") preceded by one or more spaces (" "), followed
    # by one or more characters ("."), ending with 0 or 1 newline ("\n?")
    expr = r"( *//.*\n?|\s+)"
    return re.sub(expr, " ", script)


# TODO: fix so that the order of declarations, definitions, and statements does not matter
def is_equal(circuit_1: str, circuit_2: str, check_decl: bool = True):
    """Check that two circuit scripts are equal.

    Args:
        circuit_1 (str): first script to be compared
        circuit_2 (str): second script to be compared to the first

    Returns:
        bool: whether circuit_1 and circuit_2 are semantically equivalent
    """
    clist_1 = strip(circuit_1).split(";")
    clist_2 = strip(circuit_2).split(";")

    i, j = 0, 0
    while i < len(clist_1) and j < len(clist_2):
        i, j = i + 1, j + 1
        if not check_decl:
            if is_decl(clist_1[i - 1]):
                j -= 1
                continue
            elif is_decl(clist_2[j - 1]):
                i -= 1
                continue

        if clist_1[i - 1].strip().lower() != clist_2[j - 1].strip().lower():
            return False
    return True


def is_decl(line: str) -> bool:
    """Whether an XIR line is a declaration or an include statement

    Args:
        line (str): a single line from an XIR script

    Returns:
        bool: whether the line is a declaration or an include statement
    """
    if not set(line.split()).isdisjoint({"gate", "output", "operator", "use"}):
        return True
    return False
