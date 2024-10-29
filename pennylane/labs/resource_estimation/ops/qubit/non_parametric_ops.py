from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

class ResourceHadamard(qml.Hadamard, re.ResourceConstructor):
    """Resource class for Hadamard"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.Hadamard, {})

class ResourceS(qml.S, re.ResourceConstructor):
    """Resource class for S"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        t = ResourceT.resource_rep()
        gate_types[t] = 2

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.S, {})


class ResourceSWAP(qml.SWAP, re.ResourceConstructor):
    """Resource class for SWAP"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}
        cnot = re.ResourceCNOT.resource_rep()
        gate_types[cnot] = 3

        return gate_types


class ResourceT(qml.T, re.ResourceConstructor):
    """Resource class for T"""

    @staticmethod
    def _resource_decomp(*args, **kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    @staticmethod
    def resource_rep() -> re.CompressedResourceOp:
        return re.CompressedResourceOp(qml.T, {})
