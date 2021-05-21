# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .bindings import PathInfo as PathInfo
from .bindings import Tensor64 as Tensor
from .bindings import TensorNetwork64 as TensorNetwork
from .bindings import TensorNetworkFile64 as TensorNetworkFile
from .bindings import TensorNetworkSerializer64 as TensorNetworkSerializer
from .bindings import (
    add_tensors,
    conj,
    contract_tensors,
    reshape,
    slice_index,
    transpose,
    version,
)

# Grab the current Jet version from the C++ headers.
__version__ = version()
