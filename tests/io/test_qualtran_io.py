# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.io.qualtran_io` module.
"""
import numpy as np
import pennylane as qml


class TestFromBloq:
    """Test that FromBloq accurately wraps around Bloqs."""

    def test_repr(self):
        """Tests that FromBloq has the correct __repr__"""

        from qualtran.bloqs.basic_gates import XGate

        assert qml.FromBloq(XGate(), 1).__repr__() == "FromBloq(XGate, wires=Wires([1]))"

    def test_composite_bloq_advanced(self):
        """Tests that a composite bloq with higher level abstract bloqs has the correct
        decomposition after wrapped with `FromBloq`"""
        from qualtran import BloqBuilder
        from qualtran import QUInt
        from qualtran.bloqs.arithmetic import Product, Add
        from pennylane.wires import Wires

        bb = BloqBuilder()

        w1 = bb.add_register("p1", 3)
        w2 = bb.add_register("p2", 3)
        w3 = bb.add_register("q1", 3)
        w4 = bb.add_register("q2", 3)

        w1, w2, res1 = bb.add(Product(3, 3), a=w1, b=w2)
        w3, w4, res2 = bb.add(Product(3, 3), a=w3, b=w4)
        p1p2, p1p2_plus_q1q2 = bb.add(Add(QUInt(bitsize=6), QUInt(bitsize=6)), a=res1, b=res2)

        cbloq = bb.finalize(p1=w1, p2=w2, q1=w3, q2=w4, p1p2=p1p2, p1p2_plus_q1q2=p1p2_plus_q1q2)

        expected = [
            qml.FromBloq(Product(3, 3), wires=Wires([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])),
            qml.FromBloq(Product(3, 3), wires=Wires([6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23])),
            qml.FromBloq(
                Add(QUInt(bitsize=6), QUInt(bitsize=6)),
                wires=Wires([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
            ),
        ]
        assert qml.FromBloq(cbloq, wires=range(24)).decomposition() == expected

    def test_composite_bloq(self):
        """Tests that a simple composite bloq has the correct decomposition after wrapped with `FromBloq`"""
        from qualtran import BloqBuilder
        from qualtran.bloqs.basic_gates import Hadamard, CNOT, Toffoli

        bb = BloqBuilder()  # bb is the circuit like object

        w1 = bb.add_register("wire1", 1)
        w2 = bb.add_register("wire2", 1)
        aux = bb.add_register("aux_wires", 2)

        aux_wires = bb.split(aux)

        w1 = bb.add(Hadamard(), q=w1)
        w2 = bb.add(Hadamard(), q=w2)

        w1, aux1 = bb.add(CNOT(), ctrl=w1, target=aux_wires[0])
        w2, aux2 = bb.add(CNOT(), ctrl=w2, target=aux_wires[1])

        ctrl_aux, w1 = bb.add(Toffoli(), ctrl=(aux1, aux2), target=w1)
        ctrl_aux, w2 = bb.add(Toffoli(), ctrl=ctrl_aux, target=w2)
        aux_wires = bb.join(ctrl_aux)

        circuit_bloq = bb.finalize(wire1=w1, wire2=w2, aux_wires=aux_wires)

        decomp = qml.FromBloq(circuit_bloq, wires=list(range(4))).decomposition()
        expected_decomp = [
            qml.H(0),
            qml.H(1),
            qml.CNOT([0, 2]),
            qml.CNOT([1, 3]),
            qml.Toffoli([2, 3, 0]),
            qml.Toffoli([2, 3, 1]),
        ]
        assert decomp == expected_decomp

        mapped_decomp = qml.FromBloq(circuit_bloq, wires=[3, 0, 1, 2]).decomposition()
        mapped_expected_decomp = [
            qml.H(3),
            qml.H(0),
            qml.CNOT([3, 1]),
            qml.CNOT([0, 2]),
            qml.Toffoli([1, 2, 3]),
            qml.Toffoli([1, 2, 0]),
        ]
        assert mapped_decomp == mapped_expected_decomp

    def test_atomic_bloqs(self):
        """Tests that atomic bloqs have the correct PennyLane equivalent after wrapped with `FromBloq`"""
        from qualtran.bloqs.basic_gates import Hadamard, CNOT, Toffoli

        assert Hadamard().as_pl_op(0) == qml.Hadamard(0)
        assert CNOT().as_pl_op([0, 1]) == qml.CNOT([0, 1])
        assert Toffoli().as_pl_op([0, 1, 2]) == qml.Toffoli([0, 1, 2])

        assert np.allclose(qml.FromBloq(Hadamard(), 0).matrix(), qml.Hadamard(0).matrix())
        assert np.allclose(qml.FromBloq(CNOT(), [0, 1]).matrix(), qml.CNOT([0, 1]).matrix())
        assert np.allclose(
            qml.FromBloq(Toffoli(), [0, 1, 2]).matrix(), qml.Toffoli([0, 1, 2]).matrix()
        )

    def test_bloqs(self):
        """Tests that bloqs with decompositions have the correct PennyLane decompositions after
        being wrapped with `FromBloq`"""

        from qualtran.bloqs.basic_gates import Swap

        assert qml.FromBloq(Swap(3), wires=range(6)).decomposition() == [
            qml.SWAP(wires=[0, 3]),
            qml.SWAP(wires=[1, 4]),
            qml.SWAP(wires=[2, 5]),
        ]

    def test_get_bloq_registers_info(self):
        """Tests that get_bloq_registers_info returns the expected dictionary with the correct
        registers and wires."""

        from qualtran import BloqBuilder
        from qualtran import QUInt
        from qualtran.bloqs.arithmetic import Product, Add
        from pennylane.wires import Wires

        bb = BloqBuilder()

        w1 = bb.add_register("p1", 3)
        w2 = bb.add_register("p2", 3)
        w3 = bb.add_register("q1", 3)
        w4 = bb.add_register("q2", 3)

        w1, w2, res1 = bb.add(Product(3, 3), a=w1, b=w2)
        w3, w4, res2 = bb.add(Product(3, 3), a=w3, b=w4)
        p1p2, p1p2_plus_q1q2 = bb.add(Add(QUInt(bitsize=6), QUInt(bitsize=6)), a=res1, b=res2)

        circuit_bloq = bb.finalize(
            p1=w1, p2=w2, q1=w3, q2=w4, p1p2=p1p2, p1p2_plus_q1q2=p1p2_plus_q1q2
        )

        expected = {
            "p1": Wires([0, 1, 2]),
            "p2": Wires([3, 4, 5]),
            "q1": Wires([6, 7, 8]),
            "q2": Wires([9, 10, 11]),
            "p1p2": Wires([12, 13, 14, 15, 16, 17]),
            "p1p2_plus_q1q2": Wires([18, 19, 20, 21, 22, 23]),
        }
        actual = qml.get_bloq_registers_info(circuit_bloq)

        assert actual == expected
