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
r"""Resource operators for controlled operations."""
from typing import Dict

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=arguments-differ,too-many-ancestors


class ResourceCH(qml.CH, re.ResourceOperator):
    r"""Resource class for CH gate.
    
    Resources:
        The resources are derived from the following identities:
        
        .. math:: 
            
            \begin{align}
                \hat{H} &= \hat{R}_{y}(\frac{\pi}{4}) \dot \hat{Z}  \dot \hat{R}_{y}(\frac{-\pi}{4}), \\
                \hat{Z} &= \hat{H} \dot \hat{X}  \dot \hat{H}
            \end{align}
        

        We can control on the Pauli-X gate to obtain our controlled Hadamard gate. 

    """
    # TODO: Reference this:
    # https://quantumcomputing.stackexchange.com/questions/15734/how-to-construct-a-controlled-hadamard-gate-using-single-qubit-gates-and-control

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        ry = re.ResourceRY.resource_rep()
        h = re.ResourceHadamard.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[h] = 2
        gate_types[ry] = 2
        gate_types[cnot] = 1

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCY(qml.CY, re.ResourceOperator):
    r"""Resource class for CY gate.

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Y} = \hat{S} \dot \hat{X} \dot \hat{S}^{\dagger}.

        We can control on the Pauli-X gate to obtain our controlled-Y gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        s = re.ResourceS.resource_rep()

        gate_types[cnot] = 1
        gate_types[s] = 2  # Assuming S^dagg ~ S in cost!

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCZ(qml.CZ, re.ResourceOperator):
    r"""Resource class for CZ

    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \dot \hat{X} \dot \hat{H}.

        We can control on the Pauli-X gate to obtain our controlled-Z gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 1
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCSWAP(qml.CSWAP, re.ResourceOperator):
    r"""Resource class for CSWAP

    Resources:
        The resources are taken (figure 1d) from the paper `Shallow unitary decompositions
        of quantum Fredkin and Toffoli gates for connectivity-aware equivalent circuit averaging 
        <https://arxiv.org/pdf/2305.18128>`_.

        The circuit which applies the SWAP operation on wires (1, 2) and controlled on wire (0) is
        given by: 

        .. code-block:: bash

            0: ────╭●────┤
            1: ─╭X─├●─╭X─┤   
            2: ─╰●─╰X─╰●─┤

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        tof = re.ResourceToffoli.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        gate_types[tof] = 1
        gate_types[cnot] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCCZ(qml.CCZ, re.ResourceOperator):
    r"""Resource class for CCZ
    
    Resources:
        The resources are derived from the following identity:

        .. math:: \hat{Z} = \hat{H} \dot \hat{X} \dot \hat{H}.

        We replace the Pauli-X gate with a Toffoli gate to obtain our control-control-Z gate.
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        toffoli = re.ResourceToffoli.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[toffoli] = 1
        gate_types[h] = 2

        return gate_types

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCNOT(qml.CNOT, re.ResourceOperator):
    r"""Resource class for CNOT
    
    Resources:
        There is no further decomposition provided for this gate.

    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceToffoli(qml.Toffoli, re.ResourceOperator):
    """Resource class for Toffoli
    
    Resources:
        The resources are obtained from the paper `Novel constructions for the fault-tolerant 
        Toffoli gate <https://arxiv.org/pdf/1212.5069>`_. We summarize the cost as: 
    """

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    @staticmethod
    def textbook_resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        """Resources for the Toffoli gate
    
        Resources:
            The resources are taken (figure 4.9) from the textbook `Quantum Computation and Quantum Information
            <https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview>`_.

            The circuit is given by: 
        
            .. code-block:: bash

                0: ───────────╭●───────────╭●────╭●──T──╭●─┤
                1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤     
                2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤  

        """
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        gate_types[cnot] = 6
        gate_types[h] = 2
        gate_types[t] = 7

        return gate_types
    
    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceMultiControlledX(qml.MultiControlledX, re.ResourceOperator):
    """Resource class for MultiControlledX"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRX(qml.CRX, re.ResourceOperator):
    """Resource class for CRX"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRY(qml.CRY, re.ResourceOperator):
    """Resource class for CRY"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRZ(qml.CRZ, re.ResourceOperator):
    """Resource class for CRZ"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceCRot(qml.CRot, re.ResourceOperator):
    """Resource class for CRot"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        raise re.ResourcesNotDefined

    def resource_params(self) -> dict:
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})


class ResourceControlledPhaseShift(qml.ControlledPhaseShift, re.ResourceOperator):
    """Resource class for ControlledPhaseShift"""

    @staticmethod
    def _resource_decomp(**kwargs) -> Dict[re.CompressedResourceOp, int]:
        gate_types = {}

        cnot = re.ResourceCNOT.resource_rep()
        rz = re.ResourceRZ.resource_rep()

        gate_types[cnot] = 2
        gate_types[rz] = 3

        return gate_types

    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {})
