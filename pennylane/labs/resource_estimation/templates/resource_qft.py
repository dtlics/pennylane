import pennylane as qml

from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceConstructor

class ResourceQFT(qml.QFT, ResourceConstructor):
    """Resource class for QFT"""

    @staticmethod
    def _resource_decomp(num_wires) -> dict:
        if not isinstance(num_wires, int):
            raise TypeError("num_wires must be an int.")

        if num_wires < 1:
            raise ValueError("num_wires must be greater than 0.")

        gate_types = {}

        hadamard = CompressedResourceOp(qml.Hadamard, {})
        swap = CompressedResourceOp(qml.SWAP, {})
        ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, {})

        gate_types[hadamard] = num_wires
        gate_types[swap] = num_wires // 2
        gate_types[ctrl_phase_shift] = num_wires*(num_wires - 1) // 2

        return gate_types

    def resource_rep(self) -> CompressedResourceOp:
        params = {"num_wires": len(self.wires)}
        return CompressedResourceOp(qml.QFT, params)
