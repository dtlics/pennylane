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
"""Unit tests for the ``CommuteControlledInterpreter`` class"""

# pylint:disable=wrong-import-position, unused-argument, protected-access
import numpy as np
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

from pennylane.capture.primitives import cond_prim, for_loop_prim
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.optimization.commute_controlled import (
    CommuteControlledInterpreter,
    commute_controlled,
    commute_controlled_plxpr_to_plxpr,
)


class TestCommuteControlledInterpreter:
    """Unit tests for the CommuteControlledInterpreter for commuting controlled gates."""

    def test_invalid_direction(self):
        """Test that any direction other than 'left' or 'right' raises an error."""

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)

        with pytest.raises(ValueError, match="must be 'left' or 'right'"):
            CommuteControlledInterpreter(direction="sideways")(circuit)

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_gate_with_no_basis(self, direction):
        """Test that gates with no basis specified are ignored."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliX(wires=2)
            qml.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2])
            qml.PauliX(wires=2)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=2),
            qml.ControlledQubitUnitary(jax.numpy.array([[0, 1], [1, 0]]), wires=[0, 2]),
            qml.PauliX(wires=2),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            # jax inserts a dtype for the array which confuses qml.equal
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_gate_blocked_different_basis(self, direction):
        """Test that gates do not get pushed through controlled gates whose target bases don't match."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 1])
            qml.PauliY(wires=1)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 3

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliZ(wires=1),
            qml.CNOT(wires=[2, 1]),
            qml.PauliY(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_x_gates_right(self):
        """Test that X-basis gates before controlled-X-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=2)
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CNOT(wires=[0, 2]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.PauliX(wires=2),
            qml.RX(0.2, wires=2),
            qml.CRX(0.1, wires=[0, 1]),
            qml.SX(wires=1),
            qml.PauliX(wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_x_gates_left(self):
        """Test that X-basis gates after controlled-X-type gates on targets get pushed back."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.PauliX(wires=2)
            qml.RX(0.2, wires=2)
            qml.CNOT(wires=[0, 2])
            qml.Toffoli(wires=[0, 1, 2])
            qml.SX(wires=1)
            qml.PauliX(wires=1)
            qml.CRX(0.1, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 7

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=2),
            qml.RX(0.2, wires=2),
            qml.CNOT(wires=[0, 2]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.SX(wires=1),
            qml.PauliX(wires=1),
            qml.CRX(0.1, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_x_gates(self, direction):
        """Test that X-basis gates before controlled-X-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliX(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.RX(0.2, wires=0)
            qml.Toffoli(wires=[2, 0, 1])

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 4

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliX(wires=0),
            qml.CNOT(wires=[0, 2]),
            qml.RX(0.2, wires=0),
            qml.Toffoli(wires=[2, 0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_y_gates_right(self):
        """Test that Y-basis gates before controlled-Y-type gates on targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliY(wires=2)
            qml.CRY(-0.5, wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            qml.RY(0.3, wires=1)
            qml.CY(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CRY(-0.5, wires=[0, 2]),
            qml.PauliY(wires=2),
            qml.CNOT(wires=[1, 2]),
            qml.CY(wires=[0, 1]),
            qml.RY(0.3, wires=1),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_y_gates_left(self):
        """Test that Y-basis gates after controlled-Y-type gates on targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.CRY(-0.5, wires=[0, 2])
            qml.PauliY(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=1)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliY(wires=2),
            qml.CRY(-0.5, wires=[0, 2]),
            qml.CNOT(wires=[1, 2]),
            qml.RY(0.3, wires=1),
            qml.CY(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize("direction", [("left"), ("right")])
    def test_dont_push_y_gates(self, direction):
        """Test that Y-basis gates next to controlled-Y-type gates on controls don't get pushed."""

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.CRY(-0.2, wires=[0, 2])
            qml.PauliY(wires=0)
            qml.CNOT(wires=[1, 2])
            qml.CY(wires=[0, 1])
            qml.RY(0.3, wires=0)

        # This circuit should be unchanged

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 5

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CRY(-0.2, wires=[0, 2]),
            qml.PauliY(wires=0),
            qml.CNOT(wires=[1, 2]),
            qml.CY(wires=[0, 1]),
            qml.RY(0.3, wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_z_gates_right(self):
        """Test that Z-basis gates before controlled-Z-type gates on controls *and* targets get pushed ahead."""

        @CommuteControlledInterpreter(direction="right")
        def circuit():
            qml.PauliZ(wires=2)
            qml.S(wires=0)
            qml.CZ(wires=[0, 2])

            qml.CNOT(wires=[0, 1])

            qml.PhaseShift(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)
            qml.CRZ(0.5, wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.CZ(wires=[0, 2]),
            qml.PauliZ(wires=2),
            qml.CNOT(wires=[0, 1]),
            qml.PhaseShift(0.2, wires=2),
            qml.CRZ(0.5, wires=[0, 1]),
            qml.S(wires=0),
            qml.T(wires=0),
            qml.PauliZ(wires=0),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    def test_push_z_gates_left(self):
        """Test that Z-basis after before controlled-Z-type gates on controls *and*
        targets get pushed behind."""

        @CommuteControlledInterpreter(direction="left")
        def circuit():
            qml.CZ(wires=[0, 2])
            qml.PauliZ(wires=2)
            qml.S(wires=0)

            qml.CNOT(wires=[0, 1])

            qml.CRZ(0.5, wires=[0, 1])
            qml.RZ(0.2, wires=2)
            qml.T(wires=0)
            qml.PauliZ(wires=0)

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 8

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        expected_ops = [
            qml.PauliZ(wires=2),
            qml.S(wires=0),
            qml.RZ(0.2, wires=2),
            qml.T(wires=0),
            qml.PauliZ(wires=0),
            qml.CZ(wires=[0, 2]),
            qml.CNOT(wires=[0, 1]),
            qml.CRZ(0.5, wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)

    @pytest.mark.parametrize(
        "direction, expected_ops",
        [
            (
                "right",
                [
                    qml.X(1),
                    qml.CZ(wires=[0, 1]),
                    qml.S(0),
                    qml.CNOT(wires=[1, 0]),
                    qml.Y(1),
                    qml.CRY(0.5, wires=[1, 0]),
                    qml.Y(1),
                    qml.CRZ(-0.3, wires=[0, 1]),
                    qml.PhaseShift(0.2, wires=[0]),
                    qml.T(0),
                    qml.RZ(0.2, wires=[0]),
                    qml.Z(0),
                    qml.X(1),
                    qml.CRY(0.2, wires=[1, 0]),
                ],
            ),
            (
                "left",
                [
                    qml.X(1),
                    qml.S(0),
                    qml.CZ(wires=[0, 1]),
                    qml.CNOT(wires=[1, 0]),
                    qml.Y(1),
                    qml.CRY(0.5, wires=[1, 0]),
                    qml.PhaseShift(0.2, wires=[0]),
                    qml.Y(1),
                    qml.T(0),
                    qml.RZ(0.2, wires=[0]),
                    qml.Z(0),
                    qml.CRZ(-0.3, wires=[0, 1]),
                    qml.X(1),
                    qml.CRY(0.2, wires=[1, 0]),
                ],
            ),
        ],
    )
    def test_push_mixed_with_matrix(self, direction, expected_ops):
        """Test that arbitrary gates after controlled gates on controls *and*
        targets get properly pushed."""
        # pylint:disable=too-many-function-args

        @CommuteControlledInterpreter(direction=direction)
        def circuit():
            qml.PauliX(wires=1)
            qml.S(wires=0)
            qml.CZ(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.PauliY(wires=1)
            qml.CRY(0.5, wires=[1, 0])
            qml.PhaseShift(0.2, wires=0)
            qml.PauliY(wires=1)
            qml.T(wires=0)
            qml.CRZ(-0.3, wires=[0, 1])
            qml.RZ(0.2, wires=0)
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            qml.CRY(0.2, wires=[1, 0])

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.jaxpr.eqns) == 14

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert op1.wires == op2.wires
            assert qml.math.allclose(op1.parameters, op2.parameters)


class TestCommuteControlledHigherOrderPrimitives:
    """Unit tests for the CommuteControlledInterpreter for higher order primitives."""

    @pytest.mark.parametrize("direction", ["left", "right"])
    def test_qnode(self, direction):
        """Test that the CommuteControlledInterpreter can be used with a QNode."""

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit():
            qml.PauliX(wires=1)
            qml.S(wires=0)
            qml.CZ(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.PauliY(wires=1)
            qml.CRY(0.5, wires=[1, 0])
            qml.PhaseShift(0.2, wires=0)
            qml.PauliY(wires=1)
            qml.T(wires=0)
            qml.CRZ(-0.3, wires=[0, 1])
            qml.RZ(0.2, wires=0)
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            qml.CRY(0.2, wires=[1, 0])
            return qml.expval(qml.PauliZ(0))

        transformed_circuit = CommuteControlledInterpreter(direction=direction)(circuit)

        jaxpr = jax.make_jaxpr(transformed_circuit)()
        assert len(jaxpr.eqns) == 1
        circuit_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert len(circuit_jaxpr.eqns) == 16

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        with qml.capture.pause():
            # pylint: disable=not-callable
            expected_result = circuit()

        assert qml.math.allclose(result, expected_result)

    def test_transform_higher_order_primitive(self):
        """Test that the inner_jaxpr of transform primitives is not transformed."""

        @qml.transform
        def fictitious_transform(tape):
            """Fictitious transform"""
            return [tape], lambda res: res[0]

        @CommuteControlledInterpreter()
        def circuit():
            @fictitious_transform
            def g():
                qml.RX(0.1, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.Toffoli(wires=[0, 1, 2])

            qml.RX(0.1, 0)
            g()
            qml.RY(0.2, 0)

        jaxpr = jax.make_jaxpr(circuit)()

        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.RX._primitive
        assert jaxpr.eqns[1].primitive == fictitious_transform._primitive
        assert jaxpr.eqns[2].primitive == qml.RY._primitive

        inner_jaxpr = jaxpr.eqns[1].params["inner_jaxpr"]
        assert len(inner_jaxpr.eqns) == 3
        assert inner_jaxpr.eqns[0].primitive == qml.RX._primitive
        assert inner_jaxpr.eqns[1].primitive == qml.CNOT._primitive
        assert inner_jaxpr.eqns[2].primitive == qml.Toffoli._primitive

    @pytest.mark.parametrize(
        "selector, expected_ops",
        [
            (
                0.2,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[0, 2]),
                    qml.Toffoli(wires=[0, 1, 2]),
                    qml.RX(np.pi, wires=2),
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
            (
                0.4,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.CNOT(wires=[0, 2]),
                    qml.Toffoli(wires=[0, 1, 2]),
                    qml.RX(np.pi, wires=2),
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
            (
                0.8,
                [
                    qml.CNOT(wires=[0, 1]),
                    qml.RY(np.pi, wires=2),
                    qml.CNOT(wires=[0, 2]),
                    qml.Toffoli(wires=[0, 1, 2]),
                    qml.CNOT(wires=[0, 1]),
                ],
            ),
        ],
    )
    def test_cond(self, selector, expected_ops):
        """Test that operations inside a conditional block are correctly pushed."""

        @CommuteControlledInterpreter()
        def circuit(selector, x):

            qml.CNOT(wires=[0, 1])

            def true_branch(x):
                qml.RY(x, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.Toffoli(wires=[0, 1, 2])

            # pylint: disable=unused-argument
            def false_branch(x):
                qml.RX(x, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.Toffoli(wires=[0, 1, 2])

            qml.cond(selector > 0.5, true_branch, false_branch)(x)
            qml.CNOT(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)(selector, np.pi)

        assert len(jaxpr.eqns) == 4
        assert jaxpr.eqns[0].primitive == jax.lax.gt_p
        assert jaxpr.eqns[1].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[2].primitive == cond_prim
        assert jaxpr.eqns[3].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, selector, np.pi)
        jaxpr_ops = collector.state["ops"]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires

    def test_for_loop(self):
        """Test that for operators inside a for loop are correctly pushed."""

        @CommuteControlledInterpreter()
        def circuit(x):

            qml.CNOT(wires=[0, 1])

            @qml.for_loop(0, 1)
            # pylint: disable=unused-argument
            def loop(i, x):
                qml.RX(x, wires=2)
                qml.CNOT(wires=[0, 2])
                qml.Toffoli(wires=[0, 1, 2])
                return qml.Hadamard(wires=0)

            # pylint: disable=no-value-for-parameter
            loop(x)

            qml.CNOT(wires=[0, 1])

        jaxpr = jax.make_jaxpr(circuit)(np.pi)
        assert len(jaxpr.eqns) == 3

        assert jaxpr.eqns[0].primitive == qml.CNOT._primitive
        assert jaxpr.eqns[1].primitive == for_loop_prim
        assert jaxpr.eqns[2].primitive == qml.CNOT._primitive

        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, np.pi)
        jaxpr_ops = collector.state["ops"]
        assert len(jaxpr_ops) == 6

        expected_ops = [
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=[0, 2]),
            qml.Toffoli(wires=[0, 1, 2]),
            qml.RX(np.pi, wires=[2]),
            qml.H(0),
            qml.CNOT(wires=[0, 1]),
        ]

        for op1, op2 in zip(jaxpr_ops, expected_ops):
            assert op1.name == op2.name
            assert qml.math.allclose(op1.parameters, op2.parameters)
            assert op1.wires == op2.wires
