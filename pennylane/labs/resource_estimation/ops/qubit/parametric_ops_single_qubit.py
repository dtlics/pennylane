from typing import Dict

import numpy as np

import pennylane as qml
import pennylane.labs.resource_estimation as re


def _rotation_resources(epsilon=10e-3):
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.ResourceT.resource_rep()
    gate_types[t] = num_gates

    return gate_types


class ResourceRZ(qml.RZ, re.ResourceConstructor):
    """Resource class for RZ"""

    @staticmethod
    def _resource_decomp(epsilon=10e-3) -> Dict[re.CompressedResourceOp, int]:
        return _rotation_resources(epsilon=epsilon)

    def resource_params():
        return {}

    @classmethod
    def resource_rep(cls, epsilon=10e-3) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"epsilon": epsilon})
