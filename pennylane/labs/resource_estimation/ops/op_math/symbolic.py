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
r"""Resource operators for symbolic operations."""
from collections import defaultdict
from typing import Dict

import pennylane.labs.resource_estimation as re
from pennylane import math
from pennylane.labs.resource_estimation.resource_container import _combine_dict, _scale_dict
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.op_math.controlled import ControlledOp
from pennylane.ops.op_math.exp import Exp
from pennylane.ops.op_math.pow import PowOperation

# pylint: disable=too-many-ancestors,arguments-differ,protected-access,too-many-arguments,too-many-positional-arguments


class ResourceAdjoint(AdjointOperation, re.ResourceOperator):
    """Resource class for the Adjoint symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.adjoint_resource_decomp(**base_params)
        except re.ResourcesNotDefined:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params)
                gate_types[rep] = count

            return gate_types

    def resource_params(self) -> dict:
        return {"base_class": type(self.base), "base_params": self.base.resource_params()}

    @classmethod
    def resource_rep(cls, base_class, base_params) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(cls, {"base_class": base_class, "base_params": base_params})

    @staticmethod
    def adjoint_resource_decomp(base_class, base_params) -> Dict[re.CompressedResourceOp, int]:
        return {base_class.resource_rep(**base_params): 1}

    @staticmethod
    def tracking_name(base_class, base_params) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"Adjoint({base_name})"


class ResourceControlled(ControlledOp, re.ResourceOperator):
    """Resource class for the Controlled symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **base_params
            )
        except re.ResourcesNotDefined:
            pass

        gate_types = defaultdict(int)

        if num_ctrl_values == 0:
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, gate.params, num_ctrl_wires, 0, num_work_wires)
                gate_types[rep] = count

            return gate_types

        no_control = cls.resource_rep(base_class, base_params, num_ctrl_wires, 0, num_work_wires)
        x = re.ResourceX.resource_rep()
        gate_types[no_control] = 1
        gate_types[x] = 2 * num_ctrl_values

        return gate_types

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_ctrl_wires": len(self.control_wires),
            "num_ctrl_values": len([val for val in self.control_values if not val]),
            "num_work_wires": len(self.work_wires),
        }

    @classmethod
    def resource_rep(
        cls, base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires
    ) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "num_ctrl_wires": num_ctrl_wires,
                "num_ctrl_values": num_ctrl_values,
                "num_work_wires": num_work_wires,
            },
        )

    @classmethod
    def controlled_resource_decomp(
        cls,
        outer_num_ctrl_wires,
        outer_num_ctrl_values,
        outer_num_work_wires,
        base_class,
        base_params,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
    ) -> Dict[re.CompressedResourceOp, int]:
        return {
            cls.resource_rep(
                base_class,
                base_params,
                outer_num_ctrl_wires + num_ctrl_wires,
                outer_num_ctrl_values + num_ctrl_values,
                outer_num_work_wires + num_work_wires,
            ): 1
        }

    @staticmethod
    def tracking_name(base_class, base_params, num_ctrl_wires, num_ctrl_values, num_work_wires):
        base_name = base_class.tracking_name(**base_params)
        return f"C({base_name},{num_ctrl_wires},{num_ctrl_values},{num_work_wires})"


class ResourcePow(PowOperation, re.ResourceOperator):
    """Resource class for the Pow symbolic operation."""

    @classmethod
    def _resource_decomp(
        cls, base_class, z, base_params, **kwargs
    ) -> Dict[re.CompressedResourceOp, int]:
        try:
            return base_class.pow_resource_decomp(z, **base_params)
        except re.ResourcesNotDefined:
            pass

        try:
            gate_types = defaultdict(int)
            decomp = base_class.resources(**base_params, **kwargs)
            for gate, count in decomp.items():
                rep = cls.resource_rep(gate.op_type, z, gate.params)
                gate_types[rep] = count

            return gate_types
        except re.ResourcesNotDefined:
            pass

        return {base_class.resource_rep(**base_params): z}

    def resource_params(self) -> dict:
        return {
            "base_class": type(self.base),
            "z": self.z,
            "base_params": self.base.resource_params(),
        }

    @classmethod
    def resource_rep(cls, base_class, z, base_params) -> re.CompressedResourceOp:
        return re.CompressedResourceOp(
            cls, {"base_class": base_class, "z": z, "base_params": base_params}
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, z, base_params
    ) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(base_class, z0 * z, base_params): 1}

    @staticmethod
    def tracking_name(base_class, z, base_params) -> str:
        base_name = base_class.tracking_name(**base_params)
        return f"Pow({base_name}, {z})"


class ResourceExp(Exp, re.ResourceOperator):
    """Resource class for Exp"""

    @staticmethod
    def _resource_decomp(base_class, base_params, coeff, num_steps, **kwargs):

        # Custom exponential operator resources:
        if issubclass(base_class, re.ResourceOperator):
            try:
                return base_class.exp_resource_decomp(coeff, num_steps, **base_params)
            except re.ResourcesNotDefined:
                pass

        if (pauli_sentence := base_params["pauli_rep"]) and math.real(coeff) == 0:
            if len(pauli_sentence) == 1:
                num_wires = len(pauli_sentence.wires)
                pauli_word = tuple(pauli_sentence.keys())[0]  # only one term in the sum
                return _resources_from_pauli_word(pauli_word, num_wires)

            scalar = num_steps or 1  # 1st-order Trotter-Suzuki with 'num_steps' trotter steps:
            return _scale_dict(
                _resources_from_pauli_sentence(pauli_sentence), scalar=scalar, in_place=True
            )

        raise re.ResourcesNotDefined

    def resource_params(self):
        return _extract_exp_params(self.base, self.scalar, self.num_steps)

    @classmethod
    def resource_rep(cls, base_class, base_params, coeff, num_steps, **kwargs):
        base_name = (
            base_class.tracking_name(**base_params)
            if issubclass(base_class, re.ResourceOperator)
            else base_class.__name__
        )

        name = f"Exp({base_name}, {coeff}, num_steps={num_steps})".replace("Resource", "")
        return re.CompressedResourceOp(
            cls,
            {
                "base_class": base_class,
                "base_params": base_params,
                "coeff": coeff,
                "num_steps": num_steps,
            },
            name=name,
        )

    @classmethod
    def pow_resource_decomp(
        cls, z0, base_class, base_params, coeff, num_steps
    ) -> Dict[re.CompressedResourceOp, int]:
        return {cls.resource_rep(base_class, base_params, z0 * coeff, num_steps): 1}

    @classmethod
    def controlled_resource_decomp(
        cls,
        num_ctrl_wires,
        num_ctrl_values,
        num_work_wires,
        base_class,
        base_params,
        coeff,
        num_steps,
    ) -> Dict[re.CompressedResourceOp, int]:
        """The controlled exponential decomposition of a Pauli hamiltonian is symmetric, thus we only need to control
        on the RZ gate in the middle."""
        if (pauli_sentence := base_params["pauli_rep"]) and math.real(coeff) == 0:

            if len(pauli_sentence) > 1:
                base_gate_types = cls.resources(base_class, base_params, coeff, num_steps)

                rz_counts = base_gate_types.pop(re.ResourceRZ.resource_rep())
                ctrl_rz = re.ResourceControlled.resource_rep(
                    re.ResourceRZ, {}, num_ctrl_wires, num_ctrl_values, num_work_wires
                )

                base_gate_types[ctrl_rz] = rz_counts
                return base_gate_types

        raise re.ResourcesNotDefined


def _extract_exp_params(base_op, scalar, num_steps):
    pauli_rep = base_op.pauli_rep
    isinstance_resource_op = isinstance(base_op, re.ResourceOperator)

    if not isinstance_resource_op and (pauli_rep is None):
        raise ValueError(
            f"Cannot obtain resources for the exponential of {base_op}, if it is not a ResourceOperator and it doesn't have a Pauli decomposition."
        )

    base_class = type(base_op)
    base_params = base_op.resource_params() if isinstance_resource_op else {}

    base_params["pauli_rep"] = pauli_rep

    return {
        "base_class": base_class,
        "base_params": base_params,
        "coeff": scalar,
        "num_steps": num_steps,
    }


def _resources_from_pauli_word(pauli_word, num_wires):
    pauli_string = "".join((str(v) for v in pauli_word.values()))
    len_str = len(pauli_string)

    if len_str == 0:
        return {re.ResourceGlobalPhase.resource_rep(): 1}

    if len_str == 1:
        if pauli_string == "X":
            return {re.ResourceRX.resource_rep(): 1}
        if pauli_string == "Y":
            return {re.ResourceRY.resource_rep(): 1}
        if pauli_string == "Z":
            return {re.ResourceRZ.resource_rep(): 1}

    counter = {"X": 0, "Y": 0, "Z": 0}
    for c in pauli_string:
        counter[c] += 1

    num_x = counter["X"]
    num_y = counter["Y"]

    s = re.ResourceS.resource_rep()
    h = re.ResourceHadamard.resource_rep()
    rz = re.ResourceRZ.resource_rep()
    cnot = re.ResourceCNOT.resource_rep()

    gate_types = {}
    gate_types[rz] = 1
    gate_types[s] = 2 * num_y
    gate_types[h] = 2 * (num_x + num_y)
    gate_types[cnot] = 2 * (num_wires - 1)

    return gate_types


def _resources_from_pauli_sentence(pauli_sentence):
    gate_types = defaultdict(int)
    rx = re.ResourceRX.resource_rep()
    ry = re.ResourceRY.resource_rep()
    rz = re.ResourceRZ.resource_rep()

    for pauli_word in iter(pauli_sentence.keys()):
        num_wires = len(pauli_word.wires)

        if num_wires == 1:
            pauli_string = "".join((str(v) for v in pauli_word.values()))
            op_type = {"Z": rz, "X": rx, "Y": ry}[pauli_string]
            gate_types[op_type] += 1

            continue

        pw_gates = _resources_from_pauli_word(pauli_word, num_wires)
        _ = _combine_dict(gate_types, pw_gates, in_place=True)

    return gate_types
