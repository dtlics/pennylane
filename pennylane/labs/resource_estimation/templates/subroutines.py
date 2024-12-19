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
r"""Resource operators for PennyLane subroutine templates."""
from typing import Dict

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceAdjoint,
    ResourceCNOT,
    ResourceControlled,
    ResourceControlledPhaseShift,
    ResourceHadamard,
    ResourceOperator,
    ResourceRZ,
    ResourceSWAP,
    ResourcePhaseShift,
    ResourceSingleExcitation,
)

# pylint: disable=arguments-differ


class ResourceQFT(qml.QFT, ResourceOperator):
    """Resource class for QFT.

    Resources:
        The resources are obtained from the standard decomposition of QFT as presented
        in (chapter 5) `Nielsen, M.A. and Chuang, I.L. (2011) Quantum Computation and Quantum Information
        <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        hadamard = ResourceHadamard.resource_rep()
        swap = ResourceSWAP.resource_rep()
        ctrl_phase_shift = ResourceControlledPhaseShift.resource_rep()

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires * (num_wires - 1) // 2

        return gate_types

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)


class ResourceQuantumPhaseEstimation(qml.QuantumPhaseEstimation, ResourceOperator):
    """Resource class for QPE"""

    # TODO: Add a secondary resource decomp which falls back to op.pow_resource_decomp

    @staticmethod
    def _resource_decomp(
        base_class, base_params, num_estimation_wires, **kwargs
    ) -> Dict[CompressedResourceOp, int]:
        gate_types = {}

        hadamard = ResourceHadamard.resource_rep()
        adj_qft = ResourceAdjoint.resource_rep(ResourceQFT, {"num_wires": num_estimation_wires})
        ctrl_op = ResourceControlled.resource_rep(base_class, base_params, 1, 0, 0)

        gate_types[hadamard] = num_estimation_wires
        gate_types[adj_qft] = 1
        gate_types[ctrl_op] = (2**num_estimation_wires) - 1

        return gate_types

    def resource_params(self) -> dict:
        op = self.hyperparameters["unitary"]
        num_estimation_wires = len(self.hyperparameters["estimation_wires"])

        return {
            "base_class": type(op),
            "base_params": op.resource_params(),
            "num_estimation_wires": num_estimation_wires,
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_estimation_wires, **kwargs
    ) -> CompressedResourceOp:
        params = {
            "base_class": base_class,
            "base_params": base_params,
            "num_estimation_wires": num_estimation_wires,
        }
        return CompressedResourceOp(cls, params)

    @staticmethod
    def tracking_name(base_class, base_params, num_estimation_wires, **kwargs) -> str:
        return f"QuantumPhaseEstimation({base_class}, {num_estimation_wires})"


ResourceQPE = ResourceQuantumPhaseEstimation  # Alias for ease of typing


class ResourceStatePrep(qml.StatePrep, ResourceOperator):
    """Resource class for StatePrep.

    Resources:
        TODO: add the resources here
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        rz = ResourceRZ.resource_rep()
        cnot = ResourceCNOT.resource_rep()

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        gate_types[rz] = r_count
        gate_types[cnot] = cnot_count
        return gate_types

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)


class ResourceBasisRotation(qml.BasisRotation, ResourceOperator):

    @staticmethod
    def _resource_decomp(num_wires, dim_N) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        phase_shift = ResourcePhaseShift.resource_rep()
        single_excitation = ResourceSingleExcitation.resource_rep()

        se_count = (dim_N * (dim_N - 1) / 2)
        ps_count = dim_N + se_count

        gate_types[phase_shift] = ps_count
        gate_types[single_excitation] = se_count
        return gate_types

    def resource_params(self) -> dict:
        unitary_matrix = self.hyperparameters["unitary_matrix"]
        return {"dim_N": qml.math.shape(unitary_matrix)[0]}

    @classmethod
    def resource_rep(cls, dim_N) -> CompressedResourceOp:
        params = {"dim_N": dim_N}
        return CompressedResourceOp(cls, params)


# class ResourcePrepSelPrep(qml.PrepSelPrep, ResourceOperator):

#     @staticmethod
#     def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
#         gate_types = {}
#         rz = ResourceRZ.resource_rep()
#         cnot = ResourceCNOT.resource_rep()

#         r_count = 2 ** (num_wires + 2) - 5
#         cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

#         gate_types[rz] = r_count
#         gate_types[cnot] = cnot_count
#         return gate_types

#     def resource_params(self) -> dict:
#         return {"num_wires": len(self.wires)}

#     @classmethod
#     def resource_rep(cls, num_wires) -> CompressedResourceOp:
#         params = {"num_wires": num_wires}
#         return CompressedResourceOp(cls, params)


# class ResourceReflection(qml.Reflection, ResourceOperator):

#     @staticmethod
#     def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
#         gate_types = {}
#         rz = ResourceRZ.resource_rep()
#         cnot = ResourceCNOT.resource_rep()

#         r_count = 2 ** (num_wires + 2) - 5
#         cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

#         gate_types[rz] = r_count
#         gate_types[cnot] = cnot_count
#         return gate_types

#     def resource_params(self) -> dict:
#         return {"num_wires": len(self.wires)}

#     @classmethod
#     def resource_rep(cls, num_wires) -> CompressedResourceOp:
#         params = {"num_wires": num_wires}
#         return CompressedResourceOp(cls, params)


# class ResourceQubitization(qml.Qubitization, ResourceOperator):
#     @staticmethod
#     def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
#         gate_types = {}
#         rz = ResourceRZ.resource_rep()
#         cnot = ResourceCNOT.resource_rep()

#         r_count = 2 ** (num_wires + 2) - 5
#         cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

#         gate_types[rz] = r_count
#         gate_types[cnot] = cnot_count
#         return gate_types

#     def resource_params(self) -> dict:
#         return {"num_wires": len(self.wires)}

#     @classmethod
#     def resource_rep(cls, num_wires) -> CompressedResourceOp:
#         params = {"num_wires": num_wires}
#         return CompressedResourceOp(cls, params)