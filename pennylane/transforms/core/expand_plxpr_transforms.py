# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains expand_plxpr_transforms function for applying captured
transforms to functions with program capture enabled.
"""
from functools import wraps
from typing import Callable

from pennylane.capture import PlxprInterpreter

# from .transform_dispatcher import TransformDispatcher

# has_jax = True
# try:
#     import jax
# except ImportError:
#     has_jax = False


class ExpandTransformsInterpreter(PlxprInterpreter):
    pass


def expand_plxpr_transforms(f: Callable) -> Callable:

    expansion_interpreter = ExpandTransformsInterpreter()
    transformed_f = wraps(f)(expansion_interpreter(f))
    return transformed_f
