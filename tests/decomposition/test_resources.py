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

"""Unit tests for the data structures used for resource estimation in the decomposition module."""

import pytest

import pennylane as qml
from pennylane.decomposition.resources import CompressedResourceOp, Resources


class TestResources:
    """Unit tests for the Resources data structure."""

    def test_resource_initialize(self):
        """Tests initializing a Resources object."""
        resources = Resources()
        assert resources.num_gates == 0
        assert resources.gate_counts == {}

    def test_inconsistent_gate_counts(self):
        """Tests that an error is raised of the gate count is inconsistent
        with the number of gates."""
        with pytest.raises(AssertionError):
            Resources(
                num_gates=2,
                gate_counts={
                    CompressedResourceOp(qml.RX, {}): 2,
                    CompressedResourceOp(qml.RZ, {}): 1,
                },
            )

    def test_negative_gate_counts(self):
        """Tests that an error is raised if the gate count is negative."""
        with pytest.raises(AssertionError):
            Resources(
                num_gates=1,
                gate_counts={
                    CompressedResourceOp(qml.RX, {}): 2,
                    CompressedResourceOp(qml.RZ, {}): -1,
                },
            )

    def test_add_resources(self):
        """Tests adding two Resources objects."""

        resources1 = Resources(
            num_gates=3,
            gate_counts={
                CompressedResourceOp(qml.RX, {}): 2,
                CompressedResourceOp(qml.RZ, {}): 1,
            },
        )
        resources2 = Resources(
            num_gates=2,
            gate_counts={
                CompressedResourceOp(qml.RX, {}): 1,
                CompressedResourceOp(qml.RY, {}): 1,
            },
        )

        resources = resources1 + resources2
        assert resources.num_gates == 5
        assert resources.gate_counts == {
            CompressedResourceOp(qml.RX, {}): 3,
            CompressedResourceOp(qml.RZ, {}): 1,
            CompressedResourceOp(qml.RY, {}): 1,
        }

    def test_mul_resource_with_scalar(self):
        """Tests multiplying a Resources object with a scalar."""

        resources = Resources(
            num_gates=3,
            gate_counts={
                CompressedResourceOp(qml.RX, {}): 2,
                CompressedResourceOp(qml.RZ, {}): 1,
            },
        )

        resources = resources * 2
        assert resources.num_gates == 6
        assert resources.gate_counts == {
            CompressedResourceOp(qml.RX, {}): 4,
            CompressedResourceOp(qml.RZ, {}): 2,
        }


class TestCompressedResourceOp:
    """Unit tests for the CompressedResourceOp data structure."""

    def test_initialization(self):
        """Tests creating a CompressedResourceOp object."""

        op = CompressedResourceOp(qml.QFT, {"num_wires": 5})
        assert op.op_type is qml.QFT
        assert op.params == {"num_wires": 5}

        op = CompressedResourceOp(qml.RX)
        assert op.op_type is qml.RX
        assert op.params == {}

    def test_invalid_op_type(self):
        """Tests that an error is raised if the op_type is invalid."""

        with pytest.raises(TypeError, match="op_type must be a type"):
            CompressedResourceOp(qml.RX(0.5, wires=0), {})

        with pytest.raises(TypeError, match="op_type must be a subclass of Operator"):
            CompressedResourceOp(int, {})

    def test_hash(self):
        """Tests that a CompressedResourceOp object is hashable."""

        op = CompressedResourceOp(qml.RX, {})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(qml.QFT, {"num_wires": 5})
        assert isinstance(hash(op), int)

        op = CompressedResourceOp(
            qml.ops.Controlled,
            {
                "base_class": qml.QFT,
                "base_params": {"num_wires": 5},  # nested dictionary in params
                "num_control_wires": 1,
                "num_zero_control_values": 1,
                "num_work_wires": 1,
            },
        )
        assert isinstance(hash(op), int)

    def test_empty_params_same_hash(self):
        """Tests that CompressedResourceOp objects initialized with or without empty
        parameters have the same hash."""
        op1 = CompressedResourceOp(qml.RX)
        op2 = CompressedResourceOp(qml.RX, {})
        assert hash(op1) == hash(op2)

    def test_different_params_different_hash(self):
        """Tests that CompressedResourceOp objects initialized with different parameters
        have different hashes."""
        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 6})
        assert hash(op1) != hash(op2)

    def test_equal(self):
        """Tests comparing two CompressedResourceOp objects."""

        op1 = CompressedResourceOp(qml.RX, {})
        op2 = CompressedResourceOp(qml.RX, {})
        assert op1 == op2

        op1 = CompressedResourceOp(qml.RX, {})
        op2 = CompressedResourceOp(qml.RZ, {})
        assert op1 != op2

        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 3})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 3})
        assert op1 == op2

        op1 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        op2 = CompressedResourceOp(qml.MultiRZ, {"num_wires": 6})
        assert op1 != op2

    def test_repr(self):
        """Tests the repr defined for debugging purposes."""

        op = CompressedResourceOp(qml.RX, {})
        assert repr(op) == "RX"

        op = CompressedResourceOp(qml.MultiRZ, {"num_wires": 5})
        assert repr(op) == "MultiRZ"
