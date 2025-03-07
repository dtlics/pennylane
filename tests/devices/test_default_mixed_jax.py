# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Tests for the ``default.mixed`` device for the JAX interface
"""

# pylint: disable=protected-access
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices.default_mixed import DefaultMixed

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


decorators = [lambda x: x, jax.jit]


class TestQNodeIntegration:
    """Integration tests for default.mixed with JAX. This test ensures it integrates
    properly with the PennyLane UI, in particular the QNode."""

    def test_load_device(self):
        """Test that the plugin device loads correctly"""
        dev = qml.device("default.mixed", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == qml.measurements.Shots(None)
        assert dev.short_name == "default.mixed"
        assert dev.target_device.capabilities()["passthru_devices"]["jax"] == "default.mixed"

    def test_qubit_circuit(self, tol):
        """Test that the device provides the correct
        result for a simple circuit."""
        p = jnp.array(0.543)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_correct_state(self, tol):
        """Test that the device state is correct after evaluating a
        quantum function on the device"""
        dev = qml.device("default.mixed", wires=2)

        state = dev.state
        expected = np.zeros((4, 4))
        expected[0, 0] = 1
        assert np.allclose(state, expected, atol=tol, rtol=0)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(a):
            qml.Hadamard(wires=0)
            qml.RZ(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit(jnp.array(np.pi / 4))
        state = dev.state

        amplitude = np.exp(-1j * np.pi / 4) / 2
        expected = np.array(
            [[0.5, 0, amplitude, 0], [0, 0, 0, 0], [np.conj(amplitude), 0, 0.5, 0], [0, 0, 0, 0]]
        )

        assert np.allclose(state, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_qubit_density_matrix_jit_compatible(self, n_qubits, mocker):
        """Test that _apply_density_matrix works with jax-jit"""

        dev = qml.device("default.mixed", wires=n_qubits)
        spy = mocker.spy(dev.target_device, "_apply_density_matrix")

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(state_ini):
            qml.QubitDensityMatrix(state_ini, wires=[0])
            return qml.state()

        state_ini = jnp.array([1, 0])
        rho_ini = jnp.tensordot(state_ini, state_ini, axes=0)
        rho_out = circuit(rho_ini)
        spy.assert_called_once()
        assert qml.math.get_interface(rho_out) == "jax"

        dim = 2**n_qubits
        expected = np.zeros((dim, dim))
        expected[0, 0] = 1.0
        assert np.array_equal(rho_out, expected)

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "backprop", "finite-diff"])
    def test_channel_jit_compatible(self, diff_method):
        """Test that `default.mixed` is compatible with jax-jit"""

        a, b, c = jnp.array([0.1, 0.2, 0.1])

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(a, b, c):
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.DepolarizingChannel(c, wires=[0])
            qml.DepolarizingChannel(c, wires=[1])
            return qml.expval(qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliY(1)]))

        grad_fn = jax.jit(jax.grad(circuit, argnums=[0, 1, 2]))
        res1 = grad_fn(a, b, c)

        assert len(res1) == 3
        assert all(isinstance(r_, jax.Array) for r_ in res1)

        # make the second QNode argument a constant
        grad_fn = jax.grad(circuit, argnums=[0, 1])
        res2 = grad_fn(a, b, c)

        assert len(res2) == 2
        assert all(isinstance(r_, jax.Array) for r_ in res2)
        assert qml.math.allclose(res1[:2], res2)

    @pytest.mark.parametrize("gradient_func", [qml.gradients.param_shift, None])
    def test_jit_with_shots(self, gradient_func):
        """Test that jitted execution works when shots are given."""

        dev = qml.device("default.mixed", wires=1, shots=10)

        @jax.jit
        def wrapper(x):
            with qml.queuing.AnnotatedQueue() as q:
                qml.RX(x, wires=0)
                qml.expval(qml.PauliZ(0))
            tape = qml.tape.QuantumScript.from_queue(q)
            return qml.execute([tape], dev, diff_method=gradient_func)

        assert jnp.allclose(wrapper(jnp.array(0.0))[0], 1.0)

    @pytest.mark.parametrize("shots", [10, 100, 1000])
    def test_jit_sampling_with_broadcasting(self, shots):
        """Tests that the sampling method works with broadcasting with jax-jit"""

        dev = qml.device("default.mixed", wires=1, shots=shots)

        number_of_states = 4
        state_probability = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.1, 0.2]])

        @partial(jax.jit, static_argnums=0)
        def func(number_of_states, state_probability):
            return dev.sample_basis_states(number_of_states, state_probability)

        res = func(number_of_states, state_probability)
        assert qml.math.shape(res) == (2, shots)
        assert set(qml.math.unwrap(res.flatten())).issubset({0, 1, 2, 3})

    @pytest.mark.parametrize("shots", [10, 100, 1000])
    def test_jit_with_qnode(self, shots):
        """Test that qnode can be jitted when shots are given"""

        dev = qml.device("default.mixed", wires=2, shots=shots)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit(state_ini, a, b, c):
            qml.QubitDensityMatrix(state_ini, wires=[0, 1])
            qml.RY(a, wires=0)
            qml.RX(b, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.DepolarizingChannel(c, wires=[0])
            qml.DepolarizingChannel(c, wires=[1])
            return qml.probs(wires=[0, 1])

        state_ini = jnp.array([1, 0, 0, 0])
        a, b, c = jnp.array([0.1, 0.2, 0.1])

        rho_ini = jnp.tensordot(state_ini, state_ini, axes=0)
        res = circuit(rho_ini, a, b, c)
        jacobian = jax.jacobian(circuit, argnums=[1, 2, 3])(rho_ini, a, b, c)

        assert qml.math.get_interface(res) == "jax"
        assert all(isinstance(r_, jax.Array) for r_ in jacobian)


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize("enable_x64, r_dtype", [(False, np.float32), (True, np.float64)])
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_real_dtype(self, enable_x64, r_dtype, measurement):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with real-valued outputs"""
        jax.config.update("jax_enable_x64", enable_x64)
        p = jnp.array(0.543)

        dev = qml.device("default.mixed", wires=3, r_dtype=r_dtype)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    @pytest.mark.parametrize("enable_x64, c_dtype", [(False, np.complex64), (True, np.complex128)])
    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_complex_dtype(self, enable_x64, c_dtype, measurement):
        """Test that the user-defined dtype of the device is preserved
        for QNodes with complex-valued outputs"""
        jax.config.update("jax_enable_x64", enable_x64)
        p = jnp.array(0.543)

        dev = qml.device("default.mixed", wires=3, c_dtype=c_dtype)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype


class TestOps:
    """Unit tests for operations supported by the default.mixed device with JAX"""

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    def test_multirz_jacobian(self, jacobian_fn):
        """Test that the patched numpy functions are used for the MultiRZ
        operation and the jacobian can be computed."""
        wires = 4
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(param):
            qml.MultiRZ(param, wires=[0, 1])
            return qml.probs(wires=list(range(wires)))

        param = jnp.array(0.3)

        res = jacobian_fn(circuit)(param)
        assert np.allclose(res, np.zeros(wires**2))

    def test_full_subsystem(self, mocker):
        """Test applying a state vector to the full subsystem"""
        dev = DefaultMixed(wires=["a", "b", "c"])
        state = jnp.array([1, 0, 0, 0, 1, 0, 1, 1]) / 2.0
        state_wires = qml.wires.Wires(["a", "b", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = np.outer(state, np.conj(state))

        assert np.all(jnp.reshape(dev._state, (-1,)) == jnp.reshape(state, (-1,)))
        spy.assert_not_called()

    def test_partial_subsystem(self, mocker):
        """Test applying a state vector to a subset of wires of the full subsystem"""

        dev = DefaultMixed(wires=["a", "b", "c"])
        state = jnp.array([1, 0, 1, 0]) / jnp.sqrt(2.0)
        state_wires = qml.wires.Wires(["a", "c"])

        spy = mocker.spy(qml.math, "scatter")
        dev._apply_state_vector(state=state, device_wires=state_wires)

        state = jnp.kron(jnp.outer(state, jnp.conj(state)), jnp.array([[1, 0], [0, 0]]))

        assert np.all(jnp.reshape(dev._state, (8, 8)) == state)
        spy.assert_called()


class TestApplyChannelMethodChoice:
    """Test that the right method between _apply_channel and _apply_channel_tensordot
    is chosen."""

    @pytest.mark.parametrize(
        "op, exp_method, dev_wires",
        [
            (qml.RX(jnp.array(0.2), 0), "_apply_channel", 1),
            (qml.RX(jnp.array(0.2), 0), "_apply_channel", 8),
            (qml.CNOT([0, 1]), "_apply_channel", 3),
            (qml.CNOT([0, 1]), "_apply_channel", 8),
            (qml.MultiControlledX(wires=list(range(2))), "_apply_channel", 3),
            (qml.MultiControlledX(wires=list(range(3))), "_apply_channel_tensordot", 3),
            (qml.MultiControlledX(wires=list(range(8))), "_apply_channel_tensordot", 8),
            (qml.PauliError("X", jnp.array(0.5), 0), "_apply_channel", 2),
            (qml.PauliError("XXX", jnp.array(0.5), [0, 1, 2]), "_apply_channel", 4),
            (
                qml.PauliError("X" * 8, jnp.array(0.5), list(range(8))),
                "_apply_channel_tensordot",
                8,
            ),
        ],
    )
    def test_with_numpy_state(self, mocker, op, exp_method, dev_wires):
        """Test with a numpy array as device state."""

        methods = ["_apply_channel", "_apply_channel_tensordot"]
        del methods[methods.index(exp_method)]
        unexp_method = methods[0]
        spy_exp = mocker.spy(DefaultMixed, exp_method)
        spy_unexp = mocker.spy(DefaultMixed, unexp_method)
        dev = qml.device("default.mixed", wires=dev_wires)
        state = np.zeros((2**dev_wires, 2**dev_wires))
        state[0, 0] = 1.0
        dev._state = np.array(state).reshape([2] * (2 * dev_wires))
        dev._apply_operation(op)

        spy_unexp.assert_not_called()
        spy_exp.assert_called_once()

    @pytest.mark.parametrize(
        "op, exp_method, dev_wires",
        [
            (qml.RX(jnp.array(0.2), 0), "_apply_channel", 1),
            (qml.RX(jnp.array(0.2), 0), "_apply_channel", 8),
            (qml.CNOT([0, 1]), "_apply_channel", 3),
            (qml.CNOT([0, 1]), "_apply_channel", 8),
            (qml.MultiControlledX(wires=list(range(2))), "_apply_channel", 3),
            (qml.MultiControlledX(wires=list(range(3))), "_apply_channel", 3),
            (qml.MultiControlledX(wires=list(range(8))), "_apply_channel_tensordot", 8),
            (qml.PauliError("X", jnp.array(0.5), 0), "_apply_channel", 2),
            (qml.PauliError("XXX", jnp.array(0.5), [0, 1, 2]), "_apply_channel", 4),
            (
                qml.PauliError("X" * 8, jnp.array(0.5), list(range(8))),
                "_apply_channel_tensordot",
                8,
            ),
        ],
    )
    def test_with_jax_state(self, mocker, op, exp_method, dev_wires):
        """Test with a JAX array as device state."""

        methods = ["_apply_channel", "_apply_channel_tensordot"]
        del methods[methods.index(exp_method)]
        unexp_method = methods[0]
        spy_exp = mocker.spy(DefaultMixed, exp_method)
        spy_unexp = mocker.spy(DefaultMixed, unexp_method)
        dev = qml.device("default.mixed", wires=dev_wires)
        state = np.zeros((2**dev_wires, 2**dev_wires))
        state[0, 0] = 1.0
        dev.target_device._state = jnp.array(state).reshape([2] * (2 * dev_wires))
        dev.target_device._apply_operation(op)

        spy_unexp.assert_not_called()
        spy_exp.assert_called_once()


class TestPassthruIntegration:
    """Tests for integration with the PassthruQNode"""

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_jacobian_variable_multiply(self, jacobian_fn, decorator, tol):
        """Test that jacobian of a QNode with an attached default.mixed device with JAX
        gives the correct result in the case of parameters multiplied by scalars"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        weights = jnp.array([x, y, z])

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(p):
            qml.RX(3 * p[0], wires=0)
            qml.RY(p[1], wires=0)
            qml.RX(p[2] / 2, wires=0)
            return qml.expval(qml.PauliZ(0))

        res = decorator(circuit)(weights)

        expected = np.cos(3 * x) * np.cos(y) * np.cos(z / 2) - np.sin(3 * x) * np.sin(z / 2)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = decorator(jacobian_fn(circuit, 0))(weights)

        expected = np.array(
            [
                -3 * (np.sin(3 * x) * np.cos(y) * np.cos(z / 2) + np.cos(3 * x) * np.sin(z / 2)),
                -np.cos(3 * x) * np.sin(y) * np.cos(z / 2),
                -0.5 * (np.sin(3 * x) * np.cos(z / 2) + np.cos(3 * x) * np.cos(y) * np.sin(z / 2)),
            ]
        )

        assert qml.math.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_jacobian_repeated(self, jacobian_fn, decorator, tol):
        """Test that jacobian of a QNode with an attached default.mixed device with JAX
        gives the correct result in the case of repeated parameters"""
        x = 0.43316321
        y = 0.2162158
        z = 0.75110998
        p = jnp.array([x, y, z])
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RX(x[1], wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            return qml.expval(qml.PauliZ(0))

        res = decorator(circuit)(p)

        expected = np.cos(y) ** 2 - np.sin(x) * np.sin(y) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

        res = decorator(jacobian_fn(circuit, 0))(p)

        expected = np.array(
            [-np.cos(x) * np.sin(y) ** 2, -2 * (np.sin(x) + 1) * np.sin(y) * np.cos(y), 0]
        )
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_backprop_jacobian_agrees_parameter_shift(self, jacobian_fn, decorator, tol):
        """Test that jacobian of a QNode with an attached default.mixed device with JAX
        gives the correct result with respect to the parameter-shift method"""
        p = pnp.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])
        p_jax = jnp.array(p)

        def circuit(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

        dev1 = qml.device("default.mixed", wires=3)
        dev2 = qml.device("default.mixed", wires=3)

        circuit1 = qml.QNode(circuit, dev1, diff_method="backprop", interface="jax")
        circuit2 = qml.QNode(circuit, dev2, diff_method="parameter-shift", interface="jax")

        res = decorator(circuit1)(p_jax)
        assert np.allclose(res, circuit2(p), atol=tol, rtol=0)

        res = decorator(jacobian_fn(circuit1, 0))(p_jax)
        assert np.allclose(res, jax.jacobian(circuit2)(p), atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, wire_ids, exp_fn",
        [
            (qml.RY, [0], lambda a: -jnp.sin(a)),
            (qml.AmplitudeDamping, [0], lambda a: -2),
            (qml.DepolarizingChannel, [-1], lambda a: -4 / 3),
            (lambda a, wires: qml.ResetError(p0=a, p1=0.1, wires=wires), [0], lambda a: -2),
            (lambda a, wires: qml.ResetError(p0=0.1, p1=a, wires=wires), [0], lambda a: 0),
        ],
    )
    @pytest.mark.parametrize("decorator", decorators)
    def test_state_differentiability(self, decorator, op, wire_ids, exp_fn, tol):
        """Test that the device state can be differentiated"""
        # pylint: disable=too-many-arguments
        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(a):
            qml.PauliX(dev.wires[wire_ids[0]])
            op(a, wires=[dev.wires[idx] for idx in wire_ids])
            return qml.state()

        a = jnp.array(0.54)

        def cost(a):
            res = jnp.abs(circuit(a)) ** 2
            return res[1][1] - res[0][0]

        grad = decorator(jax.grad(cost))(a)
        expected = exp_fn(a)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_state_vector_differentiability(self, jacobian_fn, decorator, tol):
        """Test that the device state vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(a):
            qml.RY(a, wires=0)
            return qml.state()

        a = jnp.array(0.54).astype(np.complex64)

        grad = decorator(jacobian_fn(circuit, 0, holomorphic=True))(a)
        expected = 0.5 * np.array([[-np.sin(a), np.cos(a)], [np.cos(a), np.sin(a)]])

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("wires", [range(2), [-12.32, "abc"]])
    @pytest.mark.parametrize("decorator", decorators)
    def test_density_matrix_differentiability(self, decorator, wires, tol):
        """Test that the density matrix can be differentiated"""
        dev = qml.device("default.mixed", wires=wires)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a):
            qml.RY(a, wires=wires[0])
            qml.CNOT(wires=[wires[0], wires[1]])
            return qml.density_matrix(wires=wires[1])

        a = jnp.array(0.54)

        def cost(a):
            res = jnp.abs(circuit(a)) ** 2
            return res[1][1] - res[0][0]

        grad = decorator(jax.grad(cost))(a)
        expected = np.sin(a)

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator", decorators)
    def test_prob_differentiability(self, decorator, tol):
        """Test that the device probability can be differentiated"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = jnp.array(0.54)
        b = jnp.array(0.12)

        def cost(a, b):
            prob_wire_1 = circuit(a, b)
            return prob_wire_1[1] - prob_wire_1[0]

        res = decorator(cost)(a, b)
        expected = -np.cos(a) * np.cos(b)
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = decorator(jax.grad(cost, (0, 1)))(a, b)
        expected = [np.sin(a) * np.cos(b), np.cos(a) * np.sin(b)]
        assert np.allclose(grad, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_prob_vector_differentiability(self, jacobian_fn, decorator, tol):
        """Test that the device probability vector can be differentiated directly"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[1])

        a = jnp.array(0.54)
        b = jnp.array(0.12)

        res = decorator(circuit)(a, b)
        expected = [
            np.cos(a / 2) ** 2 * np.cos(b / 2) ** 2 + np.sin(a / 2) ** 2 * np.sin(b / 2) ** 2,
            np.cos(a / 2) ** 2 * np.sin(b / 2) ** 2 + np.sin(a / 2) ** 2 * np.cos(b / 2) ** 2,
        ]
        assert np.allclose(res, expected, atol=tol, rtol=0)

        grad = decorator(jacobian_fn(circuit, (0, 1)))(a, b)
        expected = 0.5 * np.array(
            [
                [-np.sin(a) * np.cos(b), np.sin(a) * np.cos(b)],
                [-np.cos(a) * np.sin(b), np.cos(a) * np.sin(b)],
            ]
        )

        assert np.allclose(grad, expected, atol=tol, rtol=0)

    def test_sample_backprop_error(self):
        """Test that sampling in backpropagation mode raises an error"""
        # pylint: disable=unused-variable
        dev = qml.device("default.mixed", wires=1, shots=100)

        msg = "does not support backprop with requested circuit"

        with pytest.raises(qml.QuantumFunctionError, match=msg):

            @qml.qnode(dev, diff_method="backprop", interface="jax")
            def circuit(a):
                qml.RY(a, wires=0)
                return qml.sample(qml.PauliZ(0))

    @pytest.mark.parametrize("decorator", decorators)
    def test_expval_gradient(self, decorator, tol):
        """Tests that the gradient of expval is correct"""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = jnp.array(-0.234)
        b = jnp.array(0.654)

        res = decorator(circuit)(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res = decorator(jax.grad(circuit, (0, 1)))(a, b)
        expected_grad = np.array(
            [-0.5 * np.sin(a) * (np.cos(b) + 1), 0.5 * np.sin(b) * (1 - np.cos(a))]
        )
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize("decorator", decorators)
    @pytest.mark.parametrize("x, shift", [(0.0, 0.0), (0.5, -0.5)])
    def test_hessian_at_zero(self, decorator, x, shift):
        """Tests that the Hessian at vanishing state vector amplitudes
        is correct."""
        dev = qml.device("default.mixed", wires=1)

        shift = jnp.array(shift)
        x = jnp.array(x)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(x):
            qml.RY(shift, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert qml.math.isclose(decorator(jax.grad(circuit))(x), 0.0)
        assert qml.math.isclose(decorator(jax.jacobian(jax.jacobian(circuit)))(x), -1.0)
        assert qml.math.isclose(decorator(jax.grad(jax.grad(circuit)))(x), -1.0)

    @pytest.mark.parametrize("operation", [qml.U3, qml.U3.compute_decomposition])
    @pytest.mark.parametrize("diff_method", ["backprop", "parameter-shift", "finite-diff"])
    def test_jax_interface_gradient(self, operation, diff_method, tol):
        """Tests that the gradient of an arbitrary U3 gate is correct
        using the JAX interface, using a variety of differentiation methods."""
        if diff_method == "finite-diff":
            jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.mixed", wires=1)
        state = jnp.array(1j * np.array([1, -1]) / np.sqrt(2))

        @qml.qnode(dev, diff_method=diff_method, interface="jax")
        def circuit(x, weights, w):
            """In this example, a mixture of scalar
            arguments, array arguments, and keyword arguments are used."""
            qml.StatePrep(state, wires=w)
            operation(x, weights[0], weights[1], wires=w)
            return qml.expval(qml.PauliX(w))

        def cost(params):
            """Perform some classical processing"""
            return circuit(params[0], params[1:], w=0) ** 2

        theta = 0.543
        phi = -0.234
        lam = 0.654

        params = jnp.array([theta, phi, lam])

        res = cost(params)
        expected_cost = (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi)) ** 2
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res = jax.grad(cost)(params)

        expected_grad = (
            np.array(
                [
                    np.sin(theta) * np.cos(lam) * np.cos(phi),
                    np.cos(theta) * np.cos(lam) * np.sin(phi) + np.sin(lam) * np.cos(phi),
                    np.cos(theta) * np.sin(lam) * np.cos(phi) + np.cos(lam) * np.sin(phi),
                ]
            )
            * 2
            * (np.sin(lam) * np.sin(phi) - np.cos(theta) * np.cos(lam) * np.cos(phi))
        )
        assert np.allclose(res, expected_grad, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "dev_name,diff_method,grad_on_execution",
        [
            ["default.mixed", "finite-diff", False],
            ["default.mixed", "parameter-shift", False],
            ["default.mixed", "backprop", True],
        ],
    )
    def test_ragged_differentiation(self, dev_name, diff_method, grad_on_execution, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""

        if diff_method == "finite-diff":
            jax.config.update("jax_enable_x64", True)

        dev = qml.device(dev_name, wires=2)
        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        @qml.qnode(
            dev, diff_method=diff_method, grad_on_execution=grad_on_execution, interface="jax"
        )
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[1])

        res = circuit(x, y)
        expected = np.array(
            [
                np.cos(x),
                (1 + np.cos(x) * np.cos(y)) / 2,
                (1 - np.cos(x) * np.cos(y)) / 2,
            ]
        )

        assert np.allclose(qml.math.hstack(res), expected, atol=tol, rtol=0)

        res = jax.jacobian(circuit, (0, 1))(x, y)

        expected = np.array(
            [
                [-np.sin(x), -np.sin(x) * np.cos(y) / 2, np.cos(y) * np.sin(x) / 2],
                [0, -np.cos(x) * np.sin(y) / 2, np.cos(x) * np.sin(y) / 2],
            ]
        )

        assert np.allclose(qml.math.hstack([res[0][0], res[1][0]]), expected[0], atol=tol, rtol=0)
        assert np.allclose(qml.math.hstack([res[0][1], res[1][1]]), expected[1], atol=tol, rtol=0)

    @pytest.mark.parametrize("jacobian_fn", [jax.jacfwd, jax.jacrev])
    @pytest.mark.parametrize("decorator", decorators)
    def test_batching(self, jacobian_fn, decorator, tol):
        """Tests that the gradient of the qnode is correct with batching"""
        dev = qml.device("default.mixed", wires=2)

        @partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(a, b):
            qml.RX(a, wires=0)
            qml.CRX(b, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        a = jnp.array([-0.234, 0.678])
        b = jnp.array([0.654, 1.236])

        res = decorator(circuit)(a, b)
        expected_cost = 0.5 * (np.cos(a) * np.cos(b) + np.cos(a) - np.cos(b) + 1)
        assert np.allclose(res, expected_cost, atol=tol, rtol=0)

        res_a, res_b = decorator(jacobian_fn(circuit, (0, 1)))(a, b)
        expected_a, expected_b = [
            -0.5 * np.sin(a) * (np.cos(b) + 1),
            0.5 * np.sin(b) * (1 - np.cos(a)),
        ]

        assert np.allclose(jnp.diag(res_a), expected_a, atol=tol, rtol=0)
        assert np.allclose(jnp.diag(res_b), expected_b, atol=tol, rtol=0)


class TestHighLevelIntegration:
    """Tests for integration with higher level components of PennyLane."""

    def test_template_integration(self):
        """Test that a PassthruQNode default.mixed with JAX works with templates."""
        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(weights):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
        weights = jnp.array(np.random.random(shape))

        grad = jax.grad(circuit)(weights)
        assert grad.shape == weights.shape

    def test_vmap_channel_ops(self):
        """Test that jax.vmap works for a QNode with channel ops"""
        dev = qml.device("default.mixed", wires=1)

        @qml.qnode(dev, diff_method="backprop", interface="jax")
        def circuit(p):
            qml.AmplitudeDamping(p, wires=0)
            qml.GeneralizedAmplitudeDamping(p, p, wires=0)
            qml.PhaseDamping(p, wires=0)
            qml.DepolarizingChannel(p, wires=0)
            qml.BitFlip(p, wires=0)
            qml.ResetError(p, p, wires=0)
            qml.PauliError("X", p, wires=0)
            qml.PhaseFlip(p, wires=0)
            qml.ThermalRelaxationError(p, p, p, 0.0001, wires=0)
            return qml.expval(qml.PauliZ(0))

        vcircuit = jax.vmap(circuit)

        x = jnp.array([0.005, 0.01, 0.02, 0.05])
        res = vcircuit(x)

        # compare vmap results to results of individually executed circuits
        expected = []
        for x_indiv in x:
            expected.append(circuit(x_indiv))

        assert np.allclose(expected, res)

    @pytest.mark.parametrize("gradient_func", [qml.gradients.param_shift, "device", None])
    def test_tapes(self, gradient_func):
        """Test that jitted execution works with tapes."""

        def cost(x, y, interface, gradient_func):
            """Executes tapes"""
            device = qml.device("default.mixed", wires=2)

            with qml.queuing.AnnotatedQueue() as q1:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(0))
                qml.expval(qml.PauliZ(1))

            tape1 = qml.tape.QuantumScript.from_queue(q1)

            with qml.queuing.AnnotatedQueue() as q2:
                qml.RX(x, wires=[0])
                qml.RY(y, wires=[1])
                qml.CNOT(wires=[0, 1])
                qml.probs(wires=[0])
                qml.probs(wires=[1])

            tape2 = qml.tape.QuantumScript.from_queue(q2)
            return [
                device.execute(
                    tape,
                    qml.devices.ExecutionConfig(interface=interface, gradient_method=gradient_func),
                )
                for tape in [tape1, tape2]
            ]

        x = jnp.array(0.543)
        y = jnp.array(-0.654)

        x_ = np.array(0.543)
        y_ = np.array(-0.654)

        res = cost(x, y, interface="jax-jit", gradient_func=gradient_func)
        exp = cost(x_, y_, interface="numpy", gradient_func=gradient_func)

        for r, e in zip(res, exp):
            assert jnp.allclose(qml.math.array(r), qml.math.array(e), atol=1e-7)


class TestMeasurements:
    """Tests for measurements with default.mixed"""

    @pytest.mark.parametrize(
        "measurement",
        [
            qml.counts(qml.PauliZ(0)),
            qml.counts(wires=[0]),
            qml.sample(qml.PauliX(0)),
            qml.sample(wires=[1]),
        ],
    )
    def test_measurements_jax(self, measurement):
        """Test sampling-based measurements work with `default.mixed` for trainable interfaces"""
        num_shots = 1024
        dev = qml.device("default.mixed", wires=2, shots=num_shots)

        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        res = circuit(jnp.array(0.5))

        assert len(res) == 2 if isinstance(measurement, qml.measurements.CountsMP) else num_shots

    @pytest.mark.parametrize(
        "meas_op",
        [qml.PauliX(0), qml.PauliZ(0)],
    )
    def test_measurement_diff(self, meas_op):
        """Test sequence of single-shot expectation values work for derivatives"""
        num_shots = 64
        dev = qml.device("default.mixed", shots=[(1, num_shots)], wires=2)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.expval(meas_op)

        def cost(angle):
            return qml.math.hstack(circuit(angle))

        angle = jnp.array(0.1234)
        assert isinstance(jax.jacobian(cost)(angle), jax.Array)
        assert len(cost(angle)) == num_shots
