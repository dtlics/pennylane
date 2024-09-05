# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Pytest configuration file for PennyLane test suite.
"""
# pylint: disable=unused-import
import contextlib
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import DefaultGaussian
from pennylane.operation import disable_new_opmath_cm, enable_new_opmath_cm

sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))

# defaults
TOL = 1e-3
TF_TOL = 2e-2
TOL_STOCHASTIC = 0.05


# pylint: disable=too-few-public-methods
class DummyDevice(DefaultGaussian):
    """Dummy device to allow Kerr operations"""

    _operation_map = DefaultGaussian._operation_map.copy()
    _operation_map["Kerr"] = lambda *x, **y: np.identity(2)


@pytest.fixture(autouse=True)
def set_numpy_seed():
    np.random.seed(9872653)
    yield


@pytest.fixture(scope="function", autouse=True)
def capture_legacy_device_deprecation_warnings():
    with warnings.catch_warnings(record=True) as recwarn:
        warnings.simplefilter("always")
        yield

        for w in recwarn:
            if isinstance(w, qml.PennyLaneDeprecationWarning):
                assert "Use of 'default.qubit." in str(w.message)
                assert "is deprecated" in str(w.message)
                assert "use 'default.qubit'" in str(w.message)

    for w in recwarn:
        if "Use of 'default.qubit." not in str(w.message):
            warnings.warn(message=w.message, category=w.category)


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def tol_stochastic():
    """Numerical tolerance for equality tests of stochastic values."""
    return TOL_STOCHASTIC


@pytest.fixture(scope="session")
def tf_tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TF_TOL", TF_TOL))


@pytest.fixture(scope="session", params=[1, 2])
def n_layers(request):
    """Number of layers."""
    return request.param


@pytest.fixture(scope="session", params=[2, 3], name="n_subsystems")
def n_subsystems_fixture(request):
    """Number of qubits or qumodes."""
    return request.param


@pytest.fixture(scope="session")
def qubit_device(n_subsystems):
    with pytest.warns(qml.PennyLaneDeprecationWarning, match="Use of 'default.qubit.legacy'"):
        return qml.device("default.qubit.legacy", wires=n_subsystems)


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qubit_device_1_wire(request):
    return qml.device(
        "default.qubit.legacy", wires=1, r_dtype=request.param[0], c_dtype=request.param[1]
    )


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qubit_device_2_wires(request):
    return qml.device(
        "default.qubit.legacy", wires=2, r_dtype=request.param[0], c_dtype=request.param[1]
    )


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qubit_device_3_wires(request):
    return qml.device(
        "default.qubit.legacy", wires=3, r_dtype=request.param[0], c_dtype=request.param[1]
    )


# The following 3 fixtures are for default.qutrit devices to be used
# for testing with various real and complex dtypes.


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_1_wire(request):
    return qml.device("default.qutrit", wires=1, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_2_wires(request):
    return qml.device("default.qutrit", wires=2, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="function", params=[(np.float32, np.complex64), (np.float64, np.complex128)])
def qutrit_device_3_wires(request):
    return qml.device("default.qutrit", wires=3, r_dtype=request.param[0], c_dtype=request.param[1])


@pytest.fixture(scope="session")
def gaussian_device(n_subsystems):
    """Number of qubits or modes."""
    return DummyDevice(wires=n_subsystems)


@pytest.fixture(scope="session")
def gaussian_dummy():
    """Gaussian device with dummy Kerr gate."""
    return DummyDevice


@pytest.fixture(scope="session")
def gaussian_device_2_wires():
    """A 2-mode Gaussian device."""
    return DummyDevice(wires=2)


@pytest.fixture(scope="session")
def gaussian_device_4modes():
    """A 4 mode Gaussian device."""
    return DummyDevice(wires=4)


#######################################################################


@pytest.fixture(scope="module", params=[1, 2, 3])
def seed(request):
    """Different seeds."""
    return request.param


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""

    with monkeypatch.context() as m:
        dev = qml.Device
        m.setattr(dev, "__abstractmethods__", frozenset())
        m.setattr(dev, "short_name", "mock_device")
        m.setattr(dev, "capabilities", lambda cls: {"model": "qubit"})
        m.setattr(dev, "operations", {"RX", "RY", "RZ", "CNOT", "SWAP"})
        yield qml.Device(wires=2)  # pylint:disable=abstract-class-instantiated


# pylint: disable=protected-access
@pytest.fixture
def tear_down_hermitian():
    yield None
    qml.Hermitian._eigs = {}


# pylint: disable=protected-access
@pytest.fixture
def tear_down_thermitian():
    yield None
    qml.THermitian._eigs = {}


#######################################################################
# Fixtures for testing under new and old opmath


def pytest_addoption(parser):
    parser.addoption(
        "--disable-opmath", action="store", default="False", help="Whether to disable new_opmath"
    )


# pylint: disable=eval-used
@pytest.fixture(scope="session", autouse=True)
def disable_opmath_if_requested(request):
    disable_opmath = request.config.getoption("--disable-opmath")
    # value from yaml file is a string, convert to boolean
    if eval(disable_opmath):
        qml.operation.disable_new_opmath(warn=True)


@pytest.fixture(scope="function")
def use_legacy_opmath():
    with disable_new_opmath_cm() as cm:
        yield cm


# pylint: disable=contextmanager-generator-missing-cleanup
@pytest.fixture(scope="function")
def use_new_opmath():
    with enable_new_opmath_cm() as cm:
        yield cm


@pytest.fixture(params=[disable_new_opmath_cm, enable_new_opmath_cm], scope="function")
def use_legacy_and_new_opmath(request):
    with request.param() as cm:
        yield cm


@pytest.fixture
def new_opmath_only():
    if not qml.operation.active_new_opmath():
        pytest.skip("This feature only works with new opmath enabled")


@pytest.fixture
def legacy_opmath_only():
    if qml.operation.active_new_opmath():
        pytest.skip("This test exclusively tests legacy opmath")


#######################################################################

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
else:
    tf_available = True

try:
    import torch
    from torch.autograd import Variable

    torch_available = True
except ImportError as e:
    torch_available = False

try:
    import jax
    import jax.numpy as jnp

    jax_available = True
except ImportError as e:
    jax_available = False


# pylint: disable=unused-argument
def pytest_generate_tests(metafunc):
    if jax_available:
        jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(items, config):
    rootdir = pathlib.Path(config.rootdir)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        if "qchem" in rel_path.parts:
            mark = getattr(pytest.mark, "qchem")
            item.add_marker(mark)
        if "finite_diff" in rel_path.parts:
            mark = getattr(pytest.mark, "finite-diff")
            item.add_marker(mark)
        if "parameter_shift" in rel_path.parts:
            mark = getattr(pytest.mark, "param-shift")
            item.add_marker(mark)
        if "data" in rel_path.parts:
            mark = getattr(pytest.mark, "data")
            item.add_marker(mark)

    # Tests that do not have a specific suite marker are marked `core`
    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        if (
            not any(
                elem
                in [
                    "autograd",
                    "data",
                    "torch",
                    "tf",
                    "jax",
                    "qchem",
                    "qcut",
                    "all_interfaces",
                    "finite-diff",
                    "param-shift",
                    "external",
                ]
                for elem in markers
            )
            or not markers
        ):
            item.add_marker(pytest.mark.core)


def pytest_runtest_setup(item):
    """Automatically skip tests if interfaces are not installed"""
    # Autograd is assumed to be installed
    interfaces = {"tf", "torch", "jax"}
    available_interfaces = {
        "tf": tf_available,
        "torch": torch_available,
        "jax": jax_available,
    }

    allowed_interfaces = [
        allowed_interface
        for allowed_interface in interfaces
        if available_interfaces[allowed_interface] is True
    ]

    # load the marker specifying what the interface is
    all_interfaces = {"tf", "torch", "jax", "all_interfaces"}
    marks = {mark.name for mark in item.iter_markers() if mark.name in all_interfaces}

    for b in marks:
        if b == "all_interfaces":
            required_interfaces = {"tf", "torch", "jax"}
            for interface in required_interfaces:
                if interface not in allowed_interfaces:
                    pytest.skip(
                        f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                    )
        else:
            if b not in allowed_interfaces:
                pytest.skip(
                    f"\nTest {item.nodeid} only runs with {allowed_interfaces} interfaces(s) but {b} interface provided",
                )
