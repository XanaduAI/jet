# Xanadu IR - Grammar

An XIR script consists of three parts, which are all optional:

* Included external files that are needed to interpret and validate parts of the script; e.g.,
`use xstd`, for including `xstd.xir`.

* Declarations of gates, mathematical functions and/or measurement and post processing statistics.
  Can be used to define which gates will be used, e.g. `cnot 0, 1`, followed by the number of
  parameters and number of wires which the gate is applicable to.

* The main circuit containing gate statements, gate and observable definitions, and measurements.

The order in which these parts appear does not matter.

## Including external files and libraries

Zero or more `use` statements, declaring which other scripts that should be included with the main
script. They may contain other XIR data such as gate or operator declarations, mathematical function
declarations or measurement and post-processing statistics declarations which can be referred to in
the main script.

These files also use the `.xir` file ending and are currently only linked to by name, and are not
imported in any way.

Example of two included external files:

```
use xstd;
use xqc/device/X8;
```

The former would be used to import a local file, `xstd.xir` located in the same folder as the main
script. The latter would be used to include specific device gate/measurement declarations from the XQC.

## Declarations

There are no gates, operators or output types included in the IR, and there are very few keywords
(see 'Keyword reference' below) in use. Due to this, all such operations should to be declared
withing this section. Note that this is currently not validated in any way by the parser.

These include:

* Gates are declared with a name followed by the number of parameters taken and number of wires
  which the gate is applicable to; e.g., `gate rx, 1, 1`.

* Functions are declared with a name followed by the number of parameters taken; e.g., `func sin, 1`.

* Outputs and measurement types are declared with a name, e.g., `output samples` or `output expval`.

The parser does not care about which names are used for the declarations. It is up to the device on
which the circuit is to be run to interpret them correctly and to support that particular gate.

Example of the three supported declaration types:
```
gate rz, 1, 1;
gate cnot, 0, 2;

func sin, 1;
func log, 1;

output sample;
output expval;
```

Circuit
-------
The main circuit supports three different type of statements: gate statements, which apply a gate
to one or more wires; gate definitions, consisting of one or several gate statements; and operator
definitions.

All statements must end with a semicolon.

### Statements

A gate statement consists of the name of the gate that is to be applied, optionally followed by
parameters enclosed in parentheses, a vertical bar, signifying a gate application, and finally the
wires on which the gate should be applied. The parameters may contain arithmetics, previously
declared functions and/or any variables accessible in the specific scope (e.g., inside a gate
definition).

Wires are always represented by either integers or names, enclosed in square brackets and separated
by commas; e.g., `[1, a, wire_three, 4]`.

An example of a couple of statements in a circuit:
```
ry(1.23) | [0];
rot(0.1, 0.2, 0.3) | [1];
h(0.2) | [0, 1, 2];
```

### User-defined gates

Users can also define their own gates consisting of one or more gate statements following the syntax
described above. A gate definition starts with the keyword `gate` followed by the name for the new
gate, any parameter variables needed in the following statements, and potentially the wires which
the gate is applicable to, ending with a colon. Preferably, but not necessarily, the statements are
followed on separate lines, ending with the `end` statement.

An example of a gate definition:
```
gate h(a)[0, 1, 2, a]:
    rz(-2.3932) | [0];
    rz(a) | [1];
    cnot | [a, 2];
end;
```

and another one:

```
gate rx42:
    rx(0.42) | [0];
    rx(0.42) | [2];
    rx(0.42) | [3];
end;
```

Note that if no wires are defined in the gate declaration, consecutive integers are assumed. The
latter example would thus be assumed to have 4 wires `[0, 1, 2, 3]`.

### User-defined operators

It's also possible to define operators in a similar way to the gate definitions. An operator
definition starts with the keyword `operator` followed by the name for the operator, any necessary
parameters, and potentially the wires which the operator is using, ending with a colon. Preferably,
but not necessarily, the operator statements are followed on separate lines, ending with the `end`
statement.

Each operator statement consists of two terms separated by a comma: a prefactor and the
operator/tensor product of operators, along with the wires they should be applied to; e.g.,
`X[0] @ Z[1]`.

```
operator o(a):
    0.7 * sin(a), X[0] @ Z[1];
    -1.6, X[0];
    2.45, Y[0] @ X[1];
end;
```

### Variables

Variables only exist within definitions, defined as an argument and used within the block of
statements. They are only defined within the scope of the definition, and do not exist outside of
it. It's even possible to name an operation with the same name as its argument, although, for obvious
reasons, this is not recommended.

Variable outside of definitions and classical variables do not currently exist.

### Comments

XIR uses C++ style comments where everthing after `//` until the end of the line is considered a
comment. Multiline comments simply consist of several single line comments.

```
rx(0.42) | [0];  // this is a comment

// these are also comments
// spread out over multiple lines
cnot | [0, 1];
```

### Notes

* Since all main parts of the XIR script are optional, an empty file is also valid.

* XIR is written in a way that's it's not reliant on any indentation or newlines. It's possible to
  remove all indentations and line-breaks, and it wouldn't change how the file is parsed.

* Everything referred to as a "name" above can contain letters (uppercase or lowercase), digits and
  underscores, and must start with a letter or an underscore.

* All names and keywords are case-sensitive.

* Basic arithmetic (passed as parameters) is handled by the parser, although more complicated
  mathematical expressions are not. For example, `3*(6+4)/2` is stored as a Decimal object `15.0`
  and `a+2+4` is stored as the string `a+6`. A caveat with this simple model is that it cannot
  simplify arithmetics separated by variables. E.g., `2+a+4` is parsed as the string `(2+a)+4`, and
  then beautified and stored as the string `2+a+4`.

  Further note that:

  - floats are stored as `decimal.Decimal` objects, to save the exact representation of
    what is written in the script

  - integers are stored as `int`

  - variables, and expressions containing variables, are stored as strings

Keyword reference
-----------------
- `use` (include external files)
- `gate` (gate defintition)
- `operator` (operator definition)
- `end` (end of definition/declaration)

- `true` (boolean true)
- `false` (boolean false)

- `pi` (mathematical constant pi)

- `gates` (declaration of gates)
- `math` (declaration of mathematical functions)
- `statistics` (declaration of measurement and post-processing statistics)
