# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the ``DecompositionRule`` class to represent a decomposition rule."""

from __future__ import annotations

from typing import Callable

from .resources import Resources


def decomposition(qfunc: Callable) -> DecompositionRule:
    """Decorator that wraps a qfunc in a ``DecompositionRule``."""
    return DecompositionRule(qfunc)


class DecompositionRule:
    """Represents a decomposition rule for an operator."""

    def __init__(self, func: Callable):
        self.impl = func
        self._compute_resources = None

    def compute_resources(self, *args, **kwargs) -> Resources:
        """Computes the resources required to implement this decomposition rule."""
        if self._compute_resources is None:
            raise NotImplementedError("No resource estimation found for this decomposition rule.")
        gate_counts: dict = self._compute_resources(*args, **kwargs)
        num_gates = sum(gate_counts.values())
        return Resources(num_gates, gate_counts)

    @property
    def resources(self):
        """Registers a function as the resource estimator of this decomposition rule."""

        def _compute_resources_decorator(resource_func):
            self._compute_resources = resource_func

        return _compute_resources_decorator
