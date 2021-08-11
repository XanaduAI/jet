"""The Jet Python API includes bindings for the core Jet C++ library along
with an interpreter for Xanadu IR (XIR) scripts and a set of classes to model
quantum gates, states, and circuits.
"""

# The existence of a Python binding is proof of its intention to be exposed.
from .bindings import *  # noqa: F403

# The rest of the modules control their exports using `__all__`.
from .circuit import *  # noqa: F403
from .factory import *  # noqa: F403
from .gate import *  # noqa: F403
from .interpreter import *  # noqa: F403
from .state import *  # noqa: F403

# Grab the current Jet version from the C++ headers.
__version__ = version()  # noqa: F405
