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
r"""Resource operators for non parametric single qubit operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ


class ResourceHadamard(qml.Hadamard, re.ResourceOperator):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceS(qml.S, re.ResourceOperator):
    """Resource class for S"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.resource_rep(**kwargs)
        gate_types[t] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceSWAP(qml.SWAP, re.ResourceOperator):
    """Resource class for SWAP"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        cnot = re.ResourceCNOT.resource_rep()
        gate_types[cnot] = 3

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceT(qml.T, re.ResourceOperator):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceX(qml.X, re.ResourceOperator):
    """Resource class for X"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)
        h = re.ResourceHadamard.resource_rep(**kwargs)

        gate_types = {}
        gate_types[s] = 2
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceY(qml.Y, re.ResourceOperator):
    """Resource class for Y"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)
        h = re.ResourceHadamard.resource_rep(**kwargs)

        gate_types = {}
        gate_types[s] = 6
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceZ(qml.Z, re.ResourceOperator):
    """Resource class for Z"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        s = re.ResourceS.resource_rep(**kwargs)

        gate_types = {}
        gate_types[s] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
