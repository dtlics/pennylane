# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the OutMultiplier template.
"""

import pennylane as qml
from pennylane.operation import Operation


class OutMultiplier(Operation):
    r"""Performs the out-place modular multiplication operation.

    This operator performs the modular multiplication of integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::
        \text{OutMultiplier}(mod) |x \rangle |y \rangle |b \rangle = |x \rangle |y \rangle |b + x \cdot y \; (mod) \rangle,
    
    This operation can be represented in a quantum circuit as:

    .. figure:: ../../_static/templates/arithmetic/outmultiplier.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        :math:`x` and :math:`y` must be smaller than :math:`mod` to get the correct result.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.Multiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        output_wires (Sequence[int]): the wires that store the multiplication result
        mod (int): the modulus for performing the multiplication, default value is :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): the two auxiliary wires to be used for the multiplication when :math:`mod \neq 2^{\text{len(output_wires)}}`

    **Example**

    This example performs the multiplication of two integers :math:`x=2` and :math:`y=7` modulo :math:`mod=12`.

    .. code-block::

        x = 2
        y = 7
        mod = 12

        x_wires = [0, 1]
        y_wires = [2, 3, 4]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]

        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    .. code-block:: pycon

        >>> print(circuit())
        [0 0 1 0]

    The result :math:`[0 0 1 0]`, is the ket representation of
    :math:`2 \cdot 7 \, \text{modulo} \, 12 = 2`.

    .. details::
        :title: Usage Details

        This template takes as input four different sets of wires. 
        
        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis. Therefore, we need at least 
        :math:`\lceil \log_2(x)\rceil` ``x_wires`` to represent :math:`x`.

        The second one is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis. Therefore, we need at least 
        :math:`\lceil \log_2(y)\rceil` ``y_wires`` to represent :math:`y`.

        The third one is ``output_wires`` which is used
        to encode the integer :math:`b+ x \cdot y (mod)` in the computational basis. Therefore, we need at least 
        :math:`\lceil \log_2(mod)\rceil` ``output_wires`` to represent :math:`b + x \cdot y (mod)`.  Note that these wires can be initialized with any integer 
        :math:`b`, but the most common choice is :math:`b=0` to obtain as a final result :math:`x \cdot y (mod)`.

        The fourth set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular multiplication operation. 

        - If :math:`mod = 2^{\text{len(output_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=None``. This is the case by default.
        
        - If :math:`mod \neq 2^{\text{len(output_wires)}}`, two ``work_wires`` have to be provided.

        Note that the OutMultiplier template allows us to perform modular multiplication in the computational basis. However if one just wants to perform 
        standard multiplication (with no modulo), the modulo :math:`mod` has to be set large enough to ensure that :math:`x \cdot k < mod`.
    """

    grad_method = None

    def __init__(
        self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if mod is None:
            mod = 2 ** len(output_wires)
        if mod != 2 ** len(output_wires) and work_wires is None:
            raise ValueError(
                f"If mod is not 2^{len(output_wires)}, two work wires should be provided."
            )
        if (not hasattr(output_wires, "__len__")) or (mod > 2 ** (len(output_wires))):
            raise ValueError("OutMultiplier must have enough wires to represent mod.")

        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")

        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")

        wires_list = ["x_wires", "y_wires", "output_wires", "work_wires"]

        for key in wires_list:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])
        self.hyperparameters["mod"] = mod
        all_wires = sum(self.hyperparameters[key] for key in wires_list)
        super().__init__(wires=all_wires, id=id)

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        }

        return OutMultiplier(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["output_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return (
            self.hyperparameters["x_wires"]
            + self.hyperparameters["y_wires"]
            + self.hyperparameters["output_wires"]
            + self.hyperparameters["work_wires"]
        )

    def decomposition(self):  # pylint: disable=arguments-differ
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            output_wires (Sequence[int]): the wires that store the multiplication result
            mod (int): the modulus for performing the multiplication, default value is :math:`2^{\text{len(output_wires)}}`
            work_wires (Sequence[int]): the two auxiliary wires to be used for the multiplication when :math:`mod \neq 2^{\text{len(output_wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutMultiplier.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [QFT(wires=[5, 6]),
        ControlledSequence(ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]), control=[2, 3]),
        Adjoint(QFT(wires=[5, 6]))]
        """
        op_list = []
        if mod != 2 ** len(output_wires):
            qft_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_output_wires = output_wires
            work_wire = None
        op_list.append(qml.QFT(wires=qft_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.ControlledSequence(
                    qml.PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires
                ),
                control=y_wires,
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_output_wires))

        return op_list
