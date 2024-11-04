import numpy as np
import pytest

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    _rotation_resources,
)


@pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
def test_rotation_resources(epsilon):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.CompressedResourceOp(re.ResourceT, {})
    gate_types[t] = num_gates
    assert gate_types == _rotation_resources(epsilon=epsilon)


class TestPauliRotation:
    """Test ResourceRX, ResourceRY, and ResourceRZ"""

    params = list(zip([re.ResourceRX, re.ResourceRY, re.ResourceRZ], [10e-3, 10e-4, 10e-5]))

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resources(self, resource_class, epsilon):
        """Test the resources method"""

        label = "error_" + resource_class.__name__.replace("Resource", '').lower()
        config = {label: epsilon}
        op = resource_class(1.24, wires=0)
        assert op.resources(config) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resource_rep(self, resource_class, epsilon): #pylint: disable=unused-argument
        """Test the compact representation"""
        op = resource_class(1.24, wires=0)
        expected = re.CompressedResourceOp(resource_class, {})
        assert op.resource_rep() == expected

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resources_from_rep(self, resource_class, epsilon):
        """Test the resources can be obtained from the compact representation"""

        label = "error_" + resource_class.__name__.replace("Resource", '').lower()
        config = {label: epsilon}
        op = resource_class(1.24, wires=0)
        expected = _rotation_resources(epsilon=epsilon)
        assert resource_class.resources(config, **op.resource_rep().params) == expected


class TestRot:
    """Test ResourceRot"""

    def test_resources(self):
        """Test the resources method"""

        op = re.ResourceRot(0.1, 0.2, 0.3, wires=0)
        config = {"error_rx": 10e-3, "error_ry": 10e-3, "error_rz": 10e-3}

    def test_resource_rep(self):
        """Test the compressed representation"""

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
