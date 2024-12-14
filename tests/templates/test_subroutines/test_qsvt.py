# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Tests for the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments, import-outside-toplevel, no-self-use
from copy import copy

import pytest
from numpy.linalg import matrix_power

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates.subroutines.qsvt import _complementary_poly


def qfunc(A):
    """Used to test queuing in the next test."""
    return qml.RX(A[0][0], wires=0)


def qfunc2(A):
    """Used to test queuing in the next test."""
    return qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0))


def lst_phis(phis):
    """Used to test queuing in the next test."""
    return [qml.PCPhase(i, 2, wires=[0, 1]) for i in phis]


class TestQSVT:
    """Test the qml.QSVT template."""

    def test_standard_validity(self):
        """Test standard validity criteria with assert_valid."""
        projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        op = qml.QSVT(qml.PauliX(wires=0), projectors)
        qml.ops.functions.assert_valid(op)

    def test_init_error(self):
        """Test that an error is raised if a non-operation object is passed
        for the block-encoding."""
        with pytest.raises(ValueError, match="Input block encoding must be an Operator"):
            qml.QSVT(1.23, [qml.Identity(wires=0)])

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "operations"),
        [
            (
                qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.5, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.3, dim=2, wires=[0, 1])],
                [0, 1],
                [
                    qml.PCPhase(0.5, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                    qml.PCPhase(0.3, dim=2, wires=[0, 1]),
                ],
            ),
            (
                qml.Hadamard(wires=0),
                [qml.RZ(-2 * theta, wires=0) for theta in [1.23, -0.5, 4]],
                [0],
                [
                    qml.RZ(-2.46, wires=0),
                    qml.Hadamard(0),
                    qml.RZ(1, wires=0),
                    qml.Hadamard(0),
                    qml.RZ(-8, wires=0),
                ],
            ),
        ],
    )
    def test_output(self, U_A, lst_projectors, wires, operations):
        """Test that qml.QSVT produces the intended measurements."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.QSVT(U_A, lst_projectors)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev)
        def circuit_correct():
            for op in operations:
                qml.apply(op)
            return qml.expval(qml.PauliZ(wires=0))

        assert pnp.isclose(circuit(), circuit_correct())

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "results"),
        [
            (
                qml.BlockEncode(0.1, wires=0),
                [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)],
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.BlockEncode(pnp.array([[0.1]]), wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qml.PauliZ(wires=0),
                [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=1)],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.PauliZ(wires=[0]),
                    qml.RY(0.2, wires=[0]),
                    qml.adjoint(qml.PauliZ(wires=[0])),
                    qml.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U_A, lst_projectors, results):
        """Test that qml.QSVT queues operations in the correct order."""
        with qml.tape.QuantumTape() as tape:
            qml.QSVT(U_A, lst_projectors)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_ops_defined_in_circuit(self):
        """Test that qml.QSVT queues operations correctly when they are called in the qnode."""
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        results = [
            qml.PCPhase(0.2, dim=1, wires=[0]),
            qml.PauliX(wires=[0]),
            qml.PCPhase(0.3, dim=1, wires=[0]),
        ]

        with qml.queuing.AnnotatedQueue() as q:
            qml.QSVT(qml.PauliX(wires=0), lst_projectors)

        tape = qml.tape.QuantumScript.from_queue(q)

        for expected, val in zip(results, tape.expand().operations):
            qml.assert_equal(expected, val)

    def test_decomposition_queues_its_contents(self):
        """Test that the decomposition method queues the decomposition in the correct order."""
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        op = qml.QSVT(qml.PauliX(wires=0), lst_projectors)
        with qml.queuing.AnnotatedQueue() as q:
            decomp = op.decomposition()

        ops, _ = qml.queuing.process_queue(q)
        for op1, op2 in zip(ops, decomp):
            qml.assert_equal(op1, op2)

    def test_wire_order(self):
        """Test that the wire order is preserved."""

        op1 = qml.GroverOperator(wires=[0, 3])
        op2 = qml.QFT(wires=[2, 1])
        qsvt_wires = qml.QSVT(op2, [op1]).wires
        assert qsvt_wires == op1.wires + op2.wires

    @pytest.mark.parametrize(
        ("quantum_function", "phi_func", "A", "phis", "results"),
        [
            (
                qfunc,
                lst_phis,
                pnp.array([[0.1, 0.2], [0.3, 0.4]]),
                pnp.array([0.2, 0.3]),
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.RX(0.1, wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qfunc2,
                lst_phis,
                pnp.array([[0.1, 0.2], [0.3, 0.4]]),
                pnp.array([0.1, 0.2]),
                [
                    qml.PCPhase(0.1, dim=2, wires=[0]),
                    qml.prod(qml.PauliX(wires=0), qml.RZ(0.1, wires=0)),
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                ],
            ),
        ],
    )
    def test_queuing_callables(self, quantum_function, phi_func, A, phis, results):
        """Test that qml.QSVT queues operations correctly when a function is called"""

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(quantum_function(A), phi_func(phis))

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_ltorch(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for torch."""
        import torch

        angles = qml.poly_to_angles(poly, "QSVT")
        default_matrix = qml.matrix(
            qml.qsvt(input_matrix, poly, encoding_wires=wires, block_encoding="embedding")
        )

        input_matrix = torch.tensor(input_matrix, dtype=float)
        angles = torch.tensor(angles, dtype=float)

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert pnp.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_jax(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for jax."""
        import jax.numpy as jnp

        angles = qml.poly_to_angles(poly, "QSVT")
        default_matrix = qml.matrix(
            qml.qsvt(input_matrix, poly, encoding_wires=wires, block_encoding="embedding")
        )

        input_matrix = jnp.array(input_matrix)
        angles = jnp.array(angles)

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert pnp.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "jax"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_jax_with_identity(self, input_matrix, poly, wires):
        """Test that applying the identity operation before the qsvt function in
        a JIT context does not affect the matrix for jax.

        The main purpose of this test is to ensure that the types of the block
        encoding and projector-controlled phase shift data in a QSVT instance
        are taken into account when inferring the backend of a QuantumScript.
        """
        import jax

        def identity_and_qsvt(angles):
            qml.Identity(wires=wires[0])
            qml.QSVT(
                qml.BlockEncode(input_matrix, wires=wires),
                [
                    qml.PCPhase(angles[i], dim=len(input_matrix), wires=wires)
                    for i in range(len(angles))
                ],
            )

        @jax.jit
        def get_matrix_with_identity(angles):
            return qml.matrix(identity_and_qsvt, wire_order=wires)(angles)

        angles = qml.poly_to_angles(poly, "QSVT")
        matrix = qml.matrix(qml.qsvt(input_matrix, poly, wires, "embedding"))
        matrix_with_identity = get_matrix_with_identity(angles)

        assert pnp.allclose(matrix, matrix_with_identity)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "poly", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0, 0.2], [0, 1])],
    )
    def test_QSVT_tensorflow(self, input_matrix, poly, wires):
        """Test that the qsvt function matrix is correct for tensorflow."""
        import tensorflow as tf

        angles = qml.poly_to_angles(poly, "QSVT")
        default_matrix = qml.matrix(qml.qsvt(input_matrix, poly, wires, "embedding"))

        input_matrix = tf.Variable(input_matrix)
        angles = tf.Variable(angles)

        op = qml.QSVT(
            qml.BlockEncode(input_matrix, wires),
            [qml.PCPhase(phi, 2, wires) for phi in angles],
        )

        assert pnp.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "tensorflow"

    @pytest.mark.parametrize(
        ("A", "phis"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [0.1, 0.2, 0.3],
            )
        ],
    )
    def test_QSVT_grad(self, A, phis):
        """Test that qml.grad results are the same as finite difference results"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(A, phis):
            qml.QSVT(
                qml.BlockEncode(A, wires=[0, 1]),
                [qml.PCPhase(phi, 2, wires=[0, 1]) for phi in phis],
            )
            return qml.expval(qml.PauliZ(wires=0))

        A = pnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex, requires_grad=True)
        phis = pnp.array([0.1, 0.2, 0.3], dtype=complex, requires_grad=True)
        y = circuit(A, phis)

        mat_grad_results, phi_grad_results = qml.grad(circuit)(A, phis)

        diff = 1e-8

        manual_mat_results = [
            (circuit(A + pnp.array([[diff, 0], [0, 0]]), phis) - y) / diff,
            (circuit(A + pnp.array([[0, diff], [0, 0]]), phis) - y) / diff,
            (circuit(A + pnp.array([[0, 0], [diff, 0]]), phis) - y) / diff,
            (circuit(A + pnp.array([[0, 0], [0, diff]]), phis) - y) / diff,
        ]

        for idx, result in enumerate(manual_mat_results):
            assert pnp.isclose(result, pnp.real(mat_grad_results.flatten()[idx]), atol=1e-6)

        manual_phi_results = [
            (circuit(A, phis + pnp.array([diff, 0, 0])) - y) / diff,
            (circuit(A, phis + pnp.array([0, diff, 0])) - y) / diff,
            (circuit(A, phis + pnp.array([0, 0, diff])) - y) / diff,
        ]

        for idx, result in enumerate(manual_phi_results):
            assert pnp.isclose(result, pnp.real(phi_grad_results[idx]), atol=1e-6)

    def test_label(self):
        """Test that the label method returns the correct string label"""
        op = qml.QSVT(qml.Hadamard(0), [qml.Identity(0)])
        assert op.label() == "QSVT"
        assert op.label(base_label="custom_label") == "custom_label"

    def test_data(self):
        """Test that the data property gets and sets the correct values"""
        op = qml.QSVT(qml.RX(1, wires=0), [qml.RY(2, wires=0), qml.RZ(3, wires=0)])
        assert op.data == (1, 2, 3)
        op.data = [4, 5, 6]
        assert op.data == (4, 5, 6)

    def test_copy(self):
        """Test that a QSVT operator can be copied."""
        orig_op = qml.QSVT(qml.RX(1, wires=0), [qml.RY(2, wires=0), qml.RZ(3, wires=0)])
        copy_op = copy(orig_op)
        qml.assert_equal(orig_op, copy_op)

        # Ensure the (nested) operations are copied instead of aliased.
        assert orig_op is not copy_op
        assert orig_op.hyperparameters["UA"] is not copy_op.hyperparameters["UA"]

        orig_projectors = orig_op.hyperparameters["projectors"]
        copy_projectors = copy_op.hyperparameters["projectors"]
        assert all(p1 is not p2 for p1, p2 in zip(orig_projectors, copy_projectors))


class Testqsvt_legacy:
    """Test the qml.qsvt_legacy function."""

    def test_qsvt_legacy_deprecated(self):
        """Test that my_feature is deprecated."""
        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):
            _ = qml.qsvt_legacy(0.3, [0.1, 0.2], [0])

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "true_mat"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [0.2, 0.3],
                [0, 1],
                # mathematical order of gates:
                qml.matrix(qml.PCPhase(0.2, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]))
                @ qml.matrix(qml.PCPhase(0.3, dim=2, wires=[0, 1])),
            ),
            (
                [[0.3, 0.1], [0.2, 0.9]],
                [0.1, 0.2, 0.3],
                [0, 1],
                # mathematical order of gates:
                qml.matrix(qml.PCPhase(0.1, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.adjoint(qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1])))
                @ qml.matrix(qml.PCPhase(0.2, dim=2, wires=[0, 1]))
                @ qml.matrix(qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]))
                @ qml.matrix(qml.PCPhase(0.3, dim=2, wires=[0, 1])),
            ),
        ],
    )
    def test_output(self, A, phis, wires, true_mat):
        """Test that qml.qsvt_legacy produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt_legacy(A, phis, wires)
            return qml.expval(qml.PauliZ(wires=0))

        observable_mat = pnp.kron(qml.matrix(qml.PauliZ(0)), pnp.eye(2))
        true_expval = (pnp.conj(true_mat).T @ observable_mat @ true_mat)[0, 0]

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):
            assert pnp.isclose(circuit(), true_expval)
            assert pnp.allclose(qml.matrix(circuit)(), true_mat)

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "result"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [-1.520692517929803, 0.05010380886509347],
                [0, 1],
                0.01,
            ),  # angles from pyqsp give 0.1*x
            (
                0.3,
                [-0.8104500678299933, 1.520692517929803, 0.7603462589648997],
                [0],
                0.009,
            ),  # angles from pyqsp give 0.1*x**2
            (
                -1,
                [-1.164, 0.3836, 0.383, 0.406],
                [0],
                -1,
            ),  # angles from pyqsp give 0.5 * (5 * x**3 - 3 * x)
        ],
    )
    def test_output_wx(self, A, phis, wires, result):
        """Test that qml.qsvt_legacy produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt_legacy(A, phis, wires, convention="Wx")
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):
            assert pnp.isclose(pnp.real(qml.matrix(circuit)())[0][0], result, rtol=1e-3)

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "result"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [-1.520692517929803, 0.05010380886509347],
                [0, 1],
                0.01,
            ),  # angles from pyqsp give 0.1*x
            (
                0.3,
                [-0.8104500678299933, 1.520692517929803, 0.7603462589648997],
                [0],
                0.009,
            ),  # angles from pyqsp give 0.1*x**2
            (
                -1,
                [-1.164, 0.3836, 0.383, 0.406],
                [0],
                -1,
            ),  # angles from pyqsp give 0.5 * (5 * x**3 - 3 * x)
        ],
    )
    def test_matrix_wx(self, A, phis, wires, result):
        """Assert that the matrix method produces the expected result using both call signatures."""

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):
            m1 = qml.matrix(qml.qsvt_legacy(A, phis, wires, convention="Wx"))
            m2 = qml.matrix(qml.qsvt_legacy, wire_order=wires)(A, phis, wires, convention="Wx")

            assert pnp.isclose(pnp.real(m1[0, 0]), result, rtol=1e-3)
            assert pnp.allclose(m1, m2)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_torch(self, input_matrix, angles, wires):
        """Test that the qsvt_legacy function matrix is correct for torch."""
        import torch

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):
            default_matrix = qml.matrix(qml.qsvt_legacy(input_matrix, angles, wires))

            input_matrix = torch.tensor(input_matrix, dtype=float)
            angles = torch.tensor(angles, dtype=float)

            op = qml.qsvt_legacy(input_matrix, angles, wires)

            assert pnp.allclose(qml.matrix(op), default_matrix)
            assert qml.math.get_interface(qml.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_jax(self, input_matrix, angles, wires):
        """Test that the qsvt_legacy function matrix is correct for jax."""
        import jax.numpy as jnp

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):

            default_matrix = qml.matrix(qml.qsvt_legacy(input_matrix, angles, wires))

            input_matrix = jnp.array(input_matrix)
            angles = jnp.array(angles)

            op = qml.qsvt_legacy(input_matrix, angles, wires)

            assert pnp.allclose(qml.matrix(op), default_matrix)
            assert qml.math.get_interface(qml.matrix(op)) == "jax"

    @pytest.mark.tf
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_tensorflow(self, input_matrix, angles, wires):
        """Test that the qsvt_legacy function matrix is correct for tensorflow."""
        import tensorflow as tf

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):

            default_matrix = qml.matrix(qml.qsvt_legacy(input_matrix, angles, wires))

            input_matrix = tf.Variable(input_matrix)
            angles = tf.Variable(angles)

            op = qml.qsvt_legacy(input_matrix, angles, wires)

            assert pnp.allclose(qml.matrix(op), default_matrix)
            assert qml.math.get_interface(qml.matrix(op)) == "tensorflow"

    def test_qsvt_grad(self):
        """Test that qml.grad results are the same as finite difference results"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(A, phis):
            qml.qsvt_legacy(
                A,
                phis,
                wires=[0, 1],
            )
            return qml.expval(qml.PauliZ(wires=0))

        with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):

            A = pnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=complex, requires_grad=True)
            phis = pnp.array([0.1, 0.2, 0.3], dtype=complex, requires_grad=True)
            y = circuit(A, phis)

            mat_grad_results, phi_grad_results = qml.grad(circuit)(A, phis)

            diff = 1e-8

            manual_mat_results = [
                (circuit(A + pnp.array([[diff, 0], [0, 0]]), phis) - y) / diff,
                (circuit(A + pnp.array([[0, diff], [0, 0]]), phis) - y) / diff,
                (circuit(A + pnp.array([[0, 0], [diff, 0]]), phis) - y) / diff,
                (circuit(A + pnp.array([[0, 0], [0, diff]]), phis) - y) / diff,
            ]

            for idx, result in enumerate(manual_mat_results):
                assert pnp.isclose(result, pnp.real(mat_grad_results.flatten()[idx]), atol=1e-6)

            manual_phi_results = [
                (circuit(A, phis + pnp.array([diff, 0, 0])) - y) / diff,
                (circuit(A, phis + pnp.array([0, diff, 0])) - y) / diff,
                (circuit(A, phis + pnp.array([0, 0, diff])) - y) / diff,
            ]

            for idx, result in enumerate(manual_phi_results):
                assert pnp.isclose(result, pnp.real(phi_grad_results[idx]), atol=1e-6)


phase_angle_data = (
    (
        [0, 0, 0],
        [3 * pnp.pi / 4, pnp.pi / 2, -pnp.pi / 4],
    ),
    (
        [1.0, 2.0, 3.0, 4.0],
        [1.0 + 3 * pnp.pi / 4, 2.0 + pnp.pi / 2, 3.0 + pnp.pi / 2, 4.0 - pnp.pi / 4],
    ),
)


@pytest.mark.jax
@pytest.mark.parametrize("initial_angles, expected_angles", phase_angle_data)
def test_private_qsp_to_qsvt_jax(initial_angles, expected_angles):
    """Test that the _qsp_to_qsvt function is jax compatible"""
    import jax.numpy as jnp

    from pennylane.templates.subroutines.qsvt import _qsp_to_qsvt

    initial_angles = jnp.array(initial_angles)
    expected_angles = jnp.array(expected_angles)

    computed_angles = _qsp_to_qsvt(initial_angles)
    jnp.allclose(computed_angles, expected_angles)


def test_global_phase_not_alway_applied():
    """Test that the global phase is not applied if it is 0"""

    with pytest.warns(qml.PennyLaneDeprecationWarning, match="`qml.qsvt_legacy` is deprecated"):

        decomposition = qml.qsvt_legacy(
            [1], [0, 1, 2, 3, 4], wires=[0], convention="Wx"
        ).decomposition()
        for op in decomposition:
            assert not isinstance(op, qml.GlobalPhase)


class Testqsvt:
    """Test the qml.qsvt function."""

    def test_qsvt_warning(self):
        """Test that qsvt through warning."""
        with pytest.warns(
            qml.PennyLaneDeprecationWarning, match="You may be trying to use the old `qsvt`"
        ):
            qml.qsvt([[0.1, 0.2], [0.2, -0.1]], [0.1, 0, 0.1], [0, 1, 2])

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires"),
        [
            (
                [[0.1, 0.2], [0.2, -0.4]],
                [0.2, 0, 0.3],
                "fable",
                [0, 1, 2],
            ),
            (
                [[0.1, 0.2], [0.2, -0.4]],
                [0.2, 0, 0.3],
                "embedding",
                [0, 1],
            ),
            (
                [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]],
                [0.2, 0, 0.3],
                "embedding",
                [0, 1, 2],
            ),
            (
                [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]],
                [0.2, 0, 0.3],
                "fable",
                [0, 1, 2, 3, 4],
            ),
            (
                0.3,
                [0.2, 0, 0.3],
                "embedding",
                [0],
            ),
        ],
    )
    def test_matrix_input(self, A, poly, encoding_wires, block_encoding):
        """Test that qml.qsvt produces the correct output when A is a matrix."""
        dev = qml.device("default.qubit", wires=encoding_wires)

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, poly, encoding_wires, block_encoding)
            return qml.state()

        A_matrix = qml.math.atleast_2d(A)
        # Calculation of the polynomial transformation on the input matrix
        expected = sum(coef * matrix_power(A_matrix, i) for i, coef in enumerate(poly))

        assert pnp.allclose(qml.matrix(circuit)()[: len(A_matrix), : len(A_matrix)].real, expected)

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires"),
        [
            (
                qml.Z(1) + qml.X(1),
                [0.2, 0, 0.3],
                "prepselprep",
                [0],
            ),
            (
                qml.Z(2) + qml.X(2) - 0.2 * qml.X(3) @ qml.Z(2),
                [0, -0.2, 0, 0.5],
                "prepselprep",
                [0, 1],
            ),
            (
                qml.Z(1) + qml.X(1),
                [0.2, 0, 0.3],
                "qubitization",
                [0],
            ),
            (
                qml.Z(2) + qml.X(2) - 0.2 * qml.X(3) @ qml.Z(2),
                [0, -0.2, 0, 0.5],
                "qubitization",
                [0, 1],
            ),
        ],
    )
    def test_ham_input(self, A, poly, encoding_wires, block_encoding):
        """Test that qml.qsvt produces the correct output when A is a hamiltonian."""

        coeffs = A.terms()[0]
        coeffs /= pnp.linalg.norm(coeffs, 1)

        A = qml.dot(coeffs, A.terms()[1])
        A_matrix = qml.matrix(A)
        dev = qml.device("default.qubit", wires=encoding_wires + A.wires)

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, poly, encoding_wires, block_encoding)
            return qml.state()

        # Calculation of the polynomial transformation on the input matrix
        expected = sum(coef * matrix_power(A_matrix, i) for i, coef in enumerate(poly))

        assert pnp.allclose(qml.matrix(circuit)()[: len(A_matrix), : len(A_matrix)].real, expected)

    @pytest.mark.parametrize(
        ("A", "poly", "block_encoding", "encoding_wires", "msg_match"),
        [
            (
                [[0.1, 0], [0, -0.1]],
                [0.3, 0, 0.4],
                "prepselprep",
                [0, 1],
                "block_encoding should take",
            ),
            (
                [[1, 0], [0, 1]],
                [0.3, 0, 0.4],
                "fable",
                [0, 1],
                "The subnormalization factor should be lower than 1",
            ),
            (qml.Z(0) - qml.X(0), [0.3, 0, 0.4], "fable", [1], "block_encoding should take"),
            (qml.Z(0) - qml.X(0), [0.3, 0, 0.4], "prepselprep", [0], "Control wires in"),
        ],
    )
    def test_raise_error(
        self, A, poly, block_encoding, encoding_wires, msg_match
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        with pytest.raises(ValueError, match=msg_match):

            qml.qsvt(A, poly, encoding_wires=encoding_wires, block_encoding=block_encoding)

    @pytest.mark.torch
    def test_qsvt_torch(self):
        """Test that the qsvt function generates the correct matrix with torch."""
        import torch

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qml.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = torch.tensor(qml.matrix(default_op))

        torch_op = qml.qsvt(torch.tensor(A), torch.tensor(poly), [0, 1, 2], "embedding")
        torch_matrix = qml.matrix(torch_op)

        assert qml.math.allclose(default_matrix, torch_matrix, atol=1e-6)
        assert qml.math.get_interface(torch_matrix) == "torch"

    @pytest.mark.jax
    def test_qsvt_jax(self):
        """Test that the qsvt function generates the correct matrix with jax."""
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qml.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = jnp.array(qml.matrix(default_op))

        jax_op = qml.qsvt(jnp.array(A), jnp.array(poly), [0, 1, 2], "embedding")
        jax_matrix = qml.matrix(jax_op)

        assert qml.math.allclose(default_matrix, jax_matrix, atol=1e-6)
        assert qml.math.get_interface(jax_matrix) == "jax"

    @pytest.mark.tf
    def test_qsvt_tensorflow(self):
        """Test that the qsvt function generates the correct matrix with tensorflow."""
        import tensorflow as tf

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        default_op = qml.qsvt(A, poly, [0, 1, 2], "embedding")
        default_matrix = qml.matrix(default_op)

        tf_op = qml.qsvt(tf.Variable(A), poly, [0, 1, 2], "embedding")
        tf_matrix = qml.matrix(tf_op)

        assert qml.math.allclose(default_matrix, tf_matrix, atol=1e-6)
        assert qml.math.get_interface(tf_matrix) == "tensorflow"

    @pytest.mark.jax
    def test_qsvt_grad(self):
        """Test that the qsvt function generates the correct output with qml.grad and jax.grad."""
        import jax
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(A):
            qml.qsvt(A, poly, [0, 1, 2], "embedding")
            return qml.expval(qml.Z(0) @ qml.Z(1))

        assert pnp.allclose(qml.grad(circuit)(pnp.array(A)), jax.grad(circuit)(jnp.array(A)))
        assert not pnp.allclose(qml.grad(circuit)(pnp.array(A)), 0.0)

    @pytest.mark.jax
    def test_qsvt_jit(self):
        """
        Test that the qsvt function works with jax.jit.
        Note that the traceable argument is A.
        """

        import jax
        import jax.numpy as jnp

        poly = [-0.1, 0, 0.2, 0, 0.5]
        A = [[-0.1, 0, 0, 0.1], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.1, 0, -0.2, -0.1]]

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(A):
            qml.qsvt(A, poly, [0, 1, 2], "embedding")
            return qml.expval(qml.Z(0) @ qml.Z(1))

        not_jitted_output = circuit(jnp.array(A))

        jitted_circuit = jax.jit(circuit)
        jitted_output = jitted_circuit(jnp.array(A))
        assert jnp.allclose(not_jitted_output, jitted_output)


class TestRootFindingSolver:
    @pytest.mark.parametrize(
        "P",
        [
            ([0.1, 0, 0.3, 0, -0.1]),
            ([0, 0.2, 0, 0.3]),
            ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
        ],
    )
    def test_complementary_polynomial(self, P):
        """Checks that |P(z)|^2 + |Q(z)|^2 = 1 in the unit circle"""

        Q = _complementary_poly(P)  # Calculate complementary polynomial Q

        # Define points on the unit circle
        theta_vals = pnp.linspace(0, 2 * pnp.pi, 100)
        unit_circle_points = pnp.exp(1j * theta_vals)

        for z in unit_circle_points:
            P_val = pnp.polyval(P, z)
            P_magnitude_squared = pnp.abs(P_val) ** 2

            Q_val = pnp.polyval(Q, z)
            Q_magnitude_squared = pnp.abs(Q_val) ** 2

            assert pnp.isclose(P_magnitude_squared + Q_magnitude_squared, 1, atol=1e-7)

    @pytest.mark.parametrize(
        "angles",
        [
            ([0.1, 2, 0.3, 3, -0.1]),
            ([0, 0.2, 1, 0.3, 4, 2.4]),
            ([-0.4, 2, 0.4, 0, -0.1, 0, 0.1]),
        ],
    )
    def test_transform_angles(self, angles):
        """Test the transform_angles function"""

        new_angles = qml.transform_angles(angles, "QSP", "QSVT")
        assert pnp.allclose(angles, qml.transform_angles(new_angles, "QSVT", "QSP"))

        new_angles = qml.transform_angles(angles, "QSVT", "QSP")
        assert pnp.allclose(angles, qml.transform_angles(new_angles, "QSP", "QSVT"))

        with pytest.raises(AssertionError, match="Invalid conversion"):
            _ = qml.transform_angles(angles, "QFT", "QSVT")

    @pytest.mark.parametrize(
        "poly",
        [
            ([0.1, 0, 0.3, 0, -0.1]),
            ([0, 0.2, 0, 0.3]),
            ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
        ],
    )
    def test_correctness_QSP_angles_root_finding(self, poly):
        """Tests that angles generate desired poly"""

        angles = qml.poly_to_angles(poly, "QSP", angle_solver="root-finding")
        x = 0.5

        @qml.qnode(qml.device("default.qubit"))
        def circuit_qsp():
            qml.RX(2 * angles[0], wires=0)
            for angle in angles[1:]:
                qml.RZ(-2 * pnp.arccos(x), wires=0)
                qml.RX(2 * angle, wires=0)

            return qml.state()

        output = qml.matrix(circuit_qsp, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert pnp.isclose(output.real, expected.real)

    @pytest.mark.parametrize(
        "poly",
        [
            ([0.1, 0, 0.3, 0, -0.1]),
            ([0, 0.2, 0, 0.3]),
            ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
        ],
    )
    def test_correctness_QSVT_angles(self, poly):
        """Tests that angles generate desired poly"""

        angles = qml.poly_to_angles(poly, "QSVT")
        x = 0.5

        block_encoding = qml.RX(-2 * pnp.arccos(x), wires=0)
        projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in angles]

        @qml.qnode(qml.device("default.qubit"))
        def circuit_qsvt():
            qml.QSVT(block_encoding, projectors)
            return qml.state()

        output = qml.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert pnp.isclose(output.real, expected.real)

    @pytest.mark.parametrize(
        ("poly", "routine", "angle_solver", "msg_match"),
        [
            (
                [0.0, 0.1, 0.2],
                "QSVT",
                "root-finding",
                "The polynomial has no definite parity",
            ),
            (
                [0, 0.1j, 0, 0.3, 0, 0.2, 0.0],
                "QSVT",
                "root-finding",
                "Array must not have an imaginary part",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QFT",
                "root-finding",
                "Invalid routine",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QSVT",
                "Pitagoras",
                "Invalid angle solver",
            ),
            (
                [0, 0.1, 0, 0.3, 0, 0.2],
                "QSP",
                "Pitagoras",
                "Invalid angle solver",
            ),
            (
                [0, 2, 0, 0.3, 0, 0.2],
                "QSP",
                "root-finding",
                "The polynomial must satisfy that |P(x)| ≤ 1",
            ),
            (
                [1],
                "QSP",
                "root-finding",
                "The polynomial must have at least degree 1",
            ),
        ],
    )
    def test_raise_error(self, poly, routine, angle_solver, msg_match):
        """Test that proper errors are raised"""

        with pytest.raises(AssertionError, match=msg_match):
            _ = qml.poly_to_angles(poly, routine, angle_solver)
