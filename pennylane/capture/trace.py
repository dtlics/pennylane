# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trace implementations for program capture"""

from functools import lru_cache

has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.core import Trace, Tracer
except ImportError as e:
    has_jax = False


@lru_cache
def _get_transform_trace():
    if not has_jax:
        return None, None

    class TransformTracer(Tracer):
        """Tracer for tracing PennyLane transforms"""

        def __init__(self, trace, val, idx):
            self._trace = trace
            self.val = val
            self.idx = idx

        def __repr__(self):
            return f"TransformTracer({self._trace}, {self.val}, {self.idx})"

        @property
        def aval(self):
            return jax.core.ShapedArray((), jnp.float64)

        def full_lower(self):
            return self

    class TransformTrace(Trace):
        """Trace for processing primitives for PennyLane transforms"""

        def __init__(
            self,
            main: jax.core.MainTrace,
            sublevel: int,
            transform_program: "pennylane.transforms.core.TransformProgram",
        ):
            super().__init__(main, sublevel)
            self.transform_program = transform_program

        def pure(self, x):
            return TransformTracer(self, x, 0)

        lift = sublift = pure

        def process_primitive(self, primitive, tracers, params):
            idx = max(t.idx for t in tracers if isinstance(t, TransformTracer))
            if idx >= len(self.transform_program) or primitive.name[:4] != "qml.":
                tracers = [t.val if isinstance(t, TransformTracer) else t for t in tracers]
                return primitive.bind(*tracers, **params)

            transform = self.transform_program[idx]
            binder = transform.plxpr_transform
            targs, tkwargs = transform.args, transform.kwargs
            return binder(primitive, tracers, params, targs, tkwargs)

    return TransformTrace, TransformTracer
