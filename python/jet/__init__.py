"""The Jet Python API includes bindings for the core Jet C++ library along
with an interpreter for Xanadu IR (XIR) scripts and a set of classes to model
quantum gates, states, and circuits.
"""

# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *
from .bindings import version

# The rest of the modules control their exports using `__all__`.
from .circuit import *
from .factory import *
from .gate import *
from .interpreter import *
from .state import *

# Grab the current Jet version from the C++ headers.
__version__ = version()
