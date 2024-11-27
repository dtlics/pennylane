# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains experimental features for
resource estimation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.resource_estimation

Resource Estimation Base Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Resources
    ~CompressedResourceOp
    ~ResourceOperator

Resource Object Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~add_in_series
    ~add_in_parallel
    ~mul_in_series
    ~mul_in_parallel
    ~substitute

Operators
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceAdjoint
    ~ResourceCH
    ~ResourceCNOT
    ~ResourceControlled
    ~ResourceControlledPhaseShift
    ~ResourceDoubleExcitation
    ~ResourceDoubleExcitationMinus
    ~ResourceDoubleExcitationPlus
    ~ResourceCRY
    ~ResourceCY
    ~ResourceCZ
    ~ResourceFermionicSWAP
    ~ResourceGlobalPhase
    ~ResourceHadamard
    ~ResourceIdentity
    ~ResourceIsingXX
    ~ResourceIsingXY
    ~ResourceIsingYY
    ~ResourceIsingZZ
    ~ResourceMultiControlledX
    ~ResourceMultiRZ
    ~ResourceOrbitalRotation
    ~ResourcePauliRot
    ~ResourceMultiControlledX
    ~ResourcePhaseShift
    ~ResourcePow
    ~ResourcePSWAP
    ~ResourceRot
    ~ResourceRX
    ~ResourceRY
    ~ResourceRZ
    ~ResourceS
    ~ResourceSingleExcitation
    ~ResourceSingleExcitationMinus
    ~ResourceSingleExcitationPlus
    ~ResourceSWAP
    ~ResourceT
    ~ResourceToffoli
    ~ResourceX
    ~ResourceY
    ~ResourceZ

Templates
~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourceQFT

Tracking Resources
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~get_resources
    ~DefaultGateSet
    ~resource_config

Exceptions
~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~ResourcesNotDefined
"""

from .resource_operator import ResourceOperator, ResourcesNotDefined
<<<<<<< HEAD
from .resource_container import CompressedResourceOp, Resources
from .resource_tracking import DefaultGateSet, get_resources, resource_config
=======

from .resource_container import (
    CompressedResourceOp,
    Resources,
    add_in_series,
    add_in_parallel,
    mul_in_series,
    mul_in_parallel,
    substitute,
)
>>>>>>> labs-resource-sub

from .ops import (
    ResourceAdjoint,
    ResourceExp,
    ResourceCH,
    ResourceCNOT,
    ResourceControlled,
    ResourceControlledPhaseShift,
    ResourceCRY,
    ResourceCY,
    ResourceCZ,
    ResourceDoubleExcitation,
    ResourceDoubleExcitationMinus,
    ResourceDoubleExcitationPlus,
    ResourceFermionicSWAP,
    ResourceGlobalPhase,
    ResourceHadamard,
    ResourceIdentity,
    ResourceIsingXX,
    ResourceIsingXY,
    ResourceIsingYY,
    ResourceIsingZZ,
    ResourceMultiControlledX,
    ResourceMultiRZ,
    ResourceOrbitalRotation,
    ResourcePauliRot,
    ResourcePow,
    ResourcePSWAP,
    ResourcePhaseShift,
    ResourceRot,
    ResourceRX,
    ResourceRY,
    ResourceRZ,
    ResourceS,
    ResourceSingleExcitation,
    ResourceSingleExcitationMinus,
    ResourceSingleExcitationPlus,
    ResourceSWAP,
    ResourceT,
    ResourceToffoli,
    ResourceX,
    ResourceY,
    ResourceZ,
)

from .templates import (
    ResourceQFT,
    ResourceTrotterProduct,
    ResourceTrotterizedQfunc,
    resource_trotterize,
)
