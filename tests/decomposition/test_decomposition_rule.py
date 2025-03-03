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

"""Unit tests for the DecompositionRule class."""

import pytest

import pennylane as qml
from pennylane.decomposition.resources import Resources, CompressedResourceOp
from pennylane.decomposition.decomposition_rule import DecompositionRule, decomposition
from pennylane.decomposition.decomposition_rule import _decompositions


class TestDecompositionRule:
    """Unit tests for DecompositionRule."""

    def test_create_decomposition_rule(self):
        """Test that a DecompositionRule object can be created."""

        def _multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        def _multi_rz_resources(num_wires):
            return {
                CompressedResourceOp(qml.RZ): 1,
                CompressedResourceOp(qml.CNOT): 2 * (num_wires - 1),
            }

        rule = decomposition(_multi_rz_decomposition, resource_fn=_multi_rz_resources)

        assert isinstance(rule, DecompositionRule)

        with qml.queuing.AnnotatedQueue() as q:
            rule.impl(0.5, wires=[0, 1, 2])

        assert q.queue == [
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        assert rule.compute_resources(num_wires=3) == Resources(
            num_gates=5,
            gate_counts={
                CompressedResourceOp(qml.RZ): 1,
                CompressedResourceOp(qml.CNOT): 4,
            },
        )

    def test_decomposition_decorator(self):
        """Tests creating a decomposition rule using the decorator syntax."""

        @qml.decomposition
        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        @multi_rz_decomposition.resources
        def _(num_wires):
            return {
                CompressedResourceOp(qml.RZ): 1,
                CompressedResourceOp(qml.CNOT): 2 * (num_wires - 1),
            }

        assert isinstance(multi_rz_decomposition, DecompositionRule)

        with qml.queuing.AnnotatedQueue() as q:
            multi_rz_decomposition.impl(0.5, wires=[0, 1, 2])

        assert q.queue == [
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        assert multi_rz_decomposition.compute_resources(num_wires=3) == Resources(
            num_gates=5,
            gate_counts={
                CompressedResourceOp(qml.RZ): 1,
                CompressedResourceOp(qml.CNOT): 4,
            },
        )

    def test_error_raised_with_no_resource_fn(self):
        """Tests that an error is raised when no resource fn is provided."""

        @qml.decomposition
        def multi_rz_decomposition(theta, wires, **__):
            for w0, w1 in zip(wires[-1:0:-1], wires[-2::-1]):
                qml.CNOT(wires=(w0, w1))
            qml.RZ(theta, wires=wires[0])
            for w0, w1 in zip(wires[1:], wires[:-1]):
                qml.CNOT(wires=(w0, w1))

        with pytest.raises(NotImplementedError, match="No resource estimation found"):
            multi_rz_decomposition.compute_resources(num_wires=3)

    def test_decomposition_dictionary(self):
        """Tests that decomposition rules can be registered for an operator."""

        class CustomOp(qml.operation.Operation):
            pass

        assert not qml.has_decomposition(CustomOp)

        @qml.decomposition
        def custom_decomp(theta, wires, **__):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RZ(theta, wires=wires[0])

        @custom_decomp.resources
        def _():
            return {
                CompressedResourceOp(qml.RZ): 2,
                CompressedResourceOp(qml.CNOT): 1,
            }

        @qml.decomposition
        def custom_decomp2(theta, wires, **__):
            qml.RX(theta, wires=wires[0])
            qml.CZ(wires=[wires[0], wires[1]])
            qml.RX(theta, wires=wires[0])

        @custom_decomp2.resources
        def _():
            return {
                CompressedResourceOp(qml.RX): 2,
                CompressedResourceOp(qml.CZ): 1,
            }

        qml.add_decomposition(CustomOp, custom_decomp)
        qml.add_decomposition(CustomOp, custom_decomp2)

        assert qml.has_decomposition(CustomOp)

        assert qml.get_decompositions(CustomOp) == [custom_decomp, custom_decomp2]

        _decompositions.pop(CustomOp)  # cleanup

    def test_auto_wrap_in_resource_op(self):
        """Tests that simply classes can be auto-wrapped in a ``CompressionResourceOp``."""

        class DummyOp(qml.operation.Operator):

            resource_param_keys = ()

        @qml.decomposition
        def custom_decomp(*_, **__):
            raise NotImplementedError

        @custom_decomp.resources
        def _():
            return {
                DummyOp: 1,
            }

        assert custom_decomp.compute_resources() == Resources(
            num_gates=1,
            gate_counts={
                CompressedResourceOp(DummyOp): 1,
            },
        )

    def test_auto_wrap_fails(self):
        """Tests that an op with non-empty resource_param_keys cannot be auto-wrapped."""

        class DummyOp(qml.operation.Operator):

            resource_param_keys = {"foo"}

        @qml.decomposition
        def custom_decomp(*_, **__):
            raise NotImplementedError

        @custom_decomp.resources
        def _():
            return {
                DummyOp: 1,
            }

        with pytest.raises(TypeError, match="Operator DummyOp has non-empty resource_param_keys"):
            custom_decomp.compute_resources()
