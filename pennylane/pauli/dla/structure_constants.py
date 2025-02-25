# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A function to compute the adjoint representation of a Lie algebra"""
from itertools import combinations, combinations_with_replacement
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.typing import TensorLike

from ..pauli_arithmetic import PauliSentence, PauliWord


def _all_commutators(ops):
    commutators = {}
    for (j, op1), (k, op2) in combinations(enumerate(ops), r=2):
        res = op1.commutator(op2)
        if res != PauliSentence({}):
            commutators[(j, k)] = res

    return commutators


def structure_constants(
    g: list[Union[Operator, PauliWord, PauliSentence]],
    pauli: bool = False,
    matrix: bool = False,
    is_orthogonal: bool = True,
) -> TensorLike:
    r"""
    Compute the structure constants that make up the adjoint representation of a Lie algebra.

    Given a DLA :math:`\{iG_1, iG_2, .. iG_d \}` of dimension :math:`d`,
    the structure constants yield the decomposition of all commutators in terms of DLA elements,

    .. math:: [i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{d-1} f^\gamma_{\alpha, \beta} iG_\gamma.

    The adjoint representation :math:`\left(\text{ad}(iG_\gamma)\right)_{\alpha, \beta} = f^\gamma_{\alpha, \beta}` is given by those structure constants,
    which can be computed via

    .. math:: f^\gamma_{\alpha, \beta} = \frac{\text{tr}\left(i G_\gamma \cdot \left[i G_\alpha, i G_\beta \right] \right)}{\text{tr}\left( iG_\gamma iG_\gamma \right)}.

    The inputs are assumed to be orthogonal unless ``is_orthogonal`` is set to ``False``.
    However, we neither assume nor enforce normalization of the DLA elements :math:`G_\alpha`.

    Args:
        g (List[Union[Operator, PauliWord, PauliSentence]]): The (dynamical) Lie algebra for which we want to compute
            its adjoint representation. DLAs can be generated by a set of generators via :func:`~lie_closure`.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or :class:`~.PauliWord` instances are input.
            This can help with performance to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.
        matrix (bool): Whether or not matrix representations are used and output in the structure constants computation. Default is ``False``.
        is_orthogonal (bool): Whether the set of operators in ``g`` is orthogonal with respect to the trace inner product.
            Default is ``True``.

    Returns:
        TensorLike: The adjoint representation of shape ``(d, d, d)``, corresponding to indices ``(gamma, alpha, beta)``.

    .. seealso:: :func:`~lie_closure`, :func:`~center`, :class:`~pennylane.pauli.PauliVSpace`, :doc:`Introduction to Dynamical Lie Algebras for quantum practitioners <demos/tutorial_liealgebra>`

    **Example**

    Let us generate the DLA of the transverse field Ising model using :func:`~lie_closure`.

    >>> n = 2
    >>> gens = [X(i) @ X(i+1) for i in range(n-1)]
    >>> gens += [Z(i) for i in range(n)]
    >>> dla = qml.lie_closure(gens)
    >>> print(dla)
    [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1)), -1.0 * (Y(0) @ Y(1))]

    The dimension of the DLA is :math:`d = 6`. Hence, the structure constants have shape ``(6, 6, 6)``.

    >>> adjoint_rep = qml.structure_constants(dla)
    >>> adjoint_rep.shape
    (6, 6, 6)

    The structure constants tell us the commutation relation between operators in the DLA via

    .. math:: [i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{d-1} f^\gamma_{\alpha, \beta} iG_\gamma.

    Let us confirm those with an example. Take :math:`[iG_1, iG_3] = [iZ_0, -iY_0 X_1] = -i 2 X_0 X_1 = -i 2 G_0`, so
    we should have :math:`f^0_{1, 3} = -2`, which is indeed the case.

    >>> adjoint_rep[0, 1, 3]
    -2.0

    We can also look at the overall adjoint action of the first element :math:`G_0 = X_{0} \otimes X_{1}` of the DLA on other elements.
    In particular, at :math:`\left(\text{ad}(iG_0)\right)_{\alpha, \beta} = f^0_{\alpha, \beta}`, which corresponds to the following matrix.

    >>> adjoint_rep[0]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [-0.,  0.,  0., -2.,  0.,  0.],
           [-0.,  0.,  0.,  0., -2.,  0.],
           [-0.,  2., -0.,  0.,  0.,  0.],
           [-0., -0.,  2.,  0.,  0.,  0.],
           [ 0., -0., -0., -0., -0.,  0.]])

    Note that we neither enforce nor assume normalization by default.

    To compute the structure constants of a non-orthogonal set of operators, use the option
    ``is_orthogonal=False``:

    >>> dla = [qml.X(0), qml.Y(0), qml.X(0) - qml.Z(0)]
    >>> adjoint_rep = qml.structure_constants(dla, is_orthogonal=False)
    >>> adjoint_rep[:, 0, 1] # commutator of X_0 and Y_0 consists of first and last operator
    array([-2.,  0.,  2.])

    We can also use matrix representations for the computation, which is sometimes faster, in particular for sums of many Pauli words.
    This is just affecting how the structure constants are computed internally, it does not change the result.

    >>> adjoint_rep2 = qml.structure_constants(dla, is_orthogonal=False, matrix=True)
    >>> qml.math.allclose(adjoint_rep, adjoint_rep2)
    True

    We can also input the DLA in form of matrices. For that we use :func:`~lie_closure` with the ``matrix=True``.

    >>> n = 4
    >>> gens = [qml.X(i) @ qml.X(i+1) + qml.Y(i) @ qml.Y(i+1) + qml.Z(i) @ qml.Z(i+1) for i in range(n-1)]
    >>> g = qml.lie_closure(gens, matrix=True)
    >>> g.shape
    (12, 16, 16)

    The DLA is represented by a collection of twelve :math:`2^4 \times 2^4` matrices.
    Hence, the dimension of the DLA is :math:`d = 12` and the structure constants have shape ``(12, 12, 12)``.

    >>> from pennylane.labs.dla import structure_constants_matrix
    >>> adj = structure_constants_matrix(g)
    >>> adj.shape
    (12, 12, 12)

    .. details::
        :title: Mathematical details

        Consider a (dynamical) Lie algebra :math:`\{iG_1, iG_2, .. iG_d \}` of dimension :math:`d`.
        The defining property of the structure constants is that they express the decomposition
        of commutators in terms of the DLA elements, as described at the top. This can be written
        as

        .. math::
            [i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{d-1} f^\gamma_{\alpha, \beta} iG_\gamma.

        Now we may multiply this equation with the adjoint of a DLA element and apply the trace:

        .. math::

            \text{tr}\left(-i G_\eta \cdot \left[i G_\alpha, i G_\beta \right] \right)
            &= \text{tr}\left(-i G_\eta
            \sum_{\gamma = 0}^{d-1} f^\gamma_{\alpha, \beta} iG_\gamma\right)\\
            &= \sum_{\gamma = 0}^{d-1} \underset{g_{\eta \gamma}}{\underbrace{
            \text{tr}\left(-i G_\eta iG_\gamma\right)}}
            f^\gamma_{\alpha, \beta} \\
            \Rightarrow\ f^\gamma_{\alpha, \beta} &= (g^{-1})_{\gamma \eta}
            \text{tr}\left(-i G_\eta \cdot \left[i G_\alpha, i G_\beta \right] \right).

        Here we introduced the Gram matrix
        :math:`g_{\alpha\beta} = \text{tr}(-iG_\alpha i G_\beta)` of the DLA elements.
        Note that this is just the projection of the commutator on the DLA element
        :math:`iG_\gamma` via the trace inner product.

        Now, if the DLA elements are orthogonal, as assumed by ``structure_constants`` by default,
        the Gram matrix will be diagonal and simply consist of some rescaling factors, so that the
        above computation becomes the equation from the very top:

        .. math::
            f^\gamma_{\alpha, \beta} =
            \frac{\text{tr}\left(i G_\gamma \cdot \left[i G_\alpha, i G_\beta \right] \right)}
            {\text{tr}\left( iG_\gamma iG_\gamma \right)}.

        This is cheaper than computing the full Gram matrix, inverting it, and multiplying it to
        the trace inner products.

        For the case of an orthonormal set of operators, we even have
        :math:`g_{\alpha\beta}=\delta_{\alpha\beta}`, so that the division in this calculation can
        be skipped.

    """
    if matrix:
        return _structure_constants_matrix(g, is_orthogonal)

    if any((op.pauli_rep is None) for op in g):
        raise ValueError(
            f"Cannot compute adjoint representation of non-pauli operators. Received {g}."
        )

    if not pauli:
        g = [op.pauli_rep for op in g]

    commutators = _all_commutators(g)

    rep = np.zeros((len(g), len(g), len(g)), dtype=float)
    for i, op in enumerate(g):
        # if is_orthogonal is activated we will use the norm_squared of the op, otherwise we won't
        norm_squared = (op @ op).trace() if is_orthogonal else 1
        for (j, k), res in commutators.items():
            # if is_orthogonal is activated, use v = ∑ (v · e_j / ||e_j||^2) * e_j
            value = (1j * (op @ res).trace()).real / norm_squared
            rep[i, j, k] = value
            rep[i, k, j] = -value

    if not is_orthogonal:
        gram = np.zeros((len(g), len(g)), dtype=float)
        for (i, op1), (j, op2) in combinations_with_replacement(enumerate(g), r=2):
            gram[i, j] = gram[j, i] = (op1 @ op2).trace()

        # Contract the structure constants on the upper index with the Gram matrix, see derivation
        rep = np.tensordot(np.linalg.pinv(gram), rep, axes=[[-1], [0]])

    return rep


def _structure_constants_matrix(g: TensorLike, is_orthogonal: bool = True) -> TensorLike:
    r"""
    Compute the structure constants that make up the adjoint representation of a Lie algebra.

    This function computes the structure constants of a Lie algebra provided by their matrix matrix representation,
    obtained from, e.g., :func:`~lie_closure`.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    :class:`~PauliSentence` that are employed in :func:`~structure_constants`, e.g., when there are few generators
    that are sums of many Paulis.

    .. seealso:: For details on the mathematical definitions, see :func:`~structure_constants` and the section "Lie algebra basics" in our `g-sim demo <https://pennylane.ai/qml/demos/tutorial_liesim/#lie-algebra-basics>`__.

    Args:
        g (np.array): The (dynamical) Lie algebra provided as matrix matrices, as generated from :func:`~lie_closure`.
            ``g`` should have shape ``(d, 2**n, 2**n)`` where ``d`` is the dimension of the algebra and ``n`` is the number of qubits. Each matrix ``g[i]`` should be Hermitian.
        is_orthogonal (bool): Whether or not the matrices in ``g`` are orthogonal with respect to the Hilbert-Schmidt inner product on
            (skew-)Hermitian matrices. If the inputs are orthogonal, it is recommended to set ``is_orthogonal`` to ``True`` to reduce
            computational cost. Defaults to ``True``.

    Returns:
        TensorLike: The adjoint representation of shape ``(d, d, d)``, corresponding to
        the indices ``(gamma, alpha, beta)``.

    **Example**

    Let us generate the DLA of the transverse field Ising model using :func:`~lie_closure`.

    >>> n = 4
    >>> gens = [qml.X(i) @ qml.X(i+1) + qml.Y(i) @ qml.Y(i+1) + qml.Z(i) @ qml.Z(i+1) for i in range(n-1)]
    >>> g = qml.lie_closure(gens, matrix=True)
    >>> g.shape
    (12, 16, 16)

    The DLA is represented by a collection of twelve :math:`2^4 \times 2^4` matrices.
    Hence, the dimension of the DLA is :math:`d = 12` and the structure constants have shape ``(12, 12, 12)``.

    >>> from pennylane.labs.dla import structure_constants_matrix
    >>> adj = structure_constants_matrix(g)
    >>> adj.shape
    (12, 12, 12)

    **Internal representation**

    As mentioned above, the input is assumed to be a batch of Hermitian matrices, even though
    algebra elements are usually skew-Hermitian. That is, the input should represent the operators
    :math:`G_\alpha` for an algebra basis :math:`\{iG_\alpha\}_\alpha`.
    In an orthonormal basis of this form, the structure constants can then be computed simply via

    .. math::

        f^\gamma_{\alpha, \beta} = \text{tr}[-i G_\gamma[iG_\alpha, iG_\beta]] = i\text{tr}[G_\gamma [G_\alpha, G_\beta]] \in \mathbb{R}.

    Possible deviations of an orthogonal basis from normalization is taken into account in a
    reduced version of the step for non-orthogonal bases below.

    **Structure constants in non-orthogonal bases**

    Structure constants are often discussed using an orthogonal basis of the algebra.
    This function can deal with non-orthogonal bases as well. For this, the Gram
    matrix :math:`g` between the basis elements is taken into account when computing the overlap
    of a commutator :math:`[iG_\alpha, iG_\beta]` with all algebra elements :math:`iG_\gamma`.
    The resulting formula reads

    .. math::

        f^\gamma_{\alpha, \beta} &= \sum_\eta g^{-1}_{\gamma\eta} i \text{tr}[G_\eta [G_\alpha, G_\beta]]\\
        g_{\gamma \eta} &= \text{tr}[G_\gamma G_\eta] \quad(\in\mathbb{R})

    Internally, the commutators are computed by evaluating all operator products and subtracting
    suitable pairs of products from each other. These products can be reused to evaluate the
    Gram matrix as well.
    For orthogonal but not normalized bases, a reduced version of this step is used, only
    computing (and inverting) the diagonal of the Gram matrix.
    """

    if getattr(g[0], "wires", False):
        # operator input
        all_wires = qml.wires.Wires.all_wires([_.wires for _ in g])
        n = len(all_wires)
        assert all_wires.toset() == set(range(n))

        g = qml.math.array(
            [qml.matrix(op, wire_order=range(n)) for op in g], dtype=complex, like=g[0]
        )
        chi = 2**n
        assert np.shape(g) == (len(g), chi, chi)

    interface = qml.math.get_interface(g[0])

    if isinstance(g[0], TensorLike) and isinstance(g, (list, tuple)):
        # list of matrices
        g = qml.math.stack(g, like=interface)

    chi = qml.math.shape(g[0])[0]
    assert qml.math.shape(g) == (len(g), chi, chi)
    assert qml.math.allclose(
        qml.math.transpose(qml.math.conj(g), (0, 2, 1)), g
    ), "Input matrices to structure_constants not Hermitian"

    # compute all commutators by computing all products first.
    # Axis ordering is (dimg, chi, _chi_) x (dimg, _chi_, chi) -> (dimg, chi, dimg, chi)
    prod = qml.math.tensordot(g, g, axes=[[2], [1]])
    # The commutators now are the difference of prod with itself, with dimg axes swapped
    all_coms = prod - qml.math.transpose(prod, (2, 1, 0, 3))

    # project commutators on the basis of g, see docstring for details.
    # Axis ordering is (dimg, _chi_, *chi*) x (dimg, *chi*, dimg, _chi_) -> (dimg, dimg, dimg)
    # Normalize trace inner product by dimension chi
    adj = qml.math.real(1j * qml.math.tensordot(g / chi, all_coms, axes=[[1, 2], [3, 1]]))

    if is_orthogonal:
        # Orthogonal but not normalized inputs. Need to correct by (diagonal) Gram matrix

        if interface == "tensorflow":
            import keras  # pylint: disable=import-outside-toplevel

            pre_diag = keras.ops.diagonal(
                keras.ops.diagonal(prod, axis1=1, axis2=3), axis1=0, axis2=1
            )
        else:
            # offset, axis1, axis2 arguments are called differently in torch, use positional arguments
            pre_diag = qml.math.diagonal(qml.math.diagonal(prod, 0, 1, 3), 0, 0, 1)

        gram_diag = qml.math.real(qml.math.sum(pre_diag, axis=0))

        adj = (chi / gram_diag[:, None, None]) * adj
    else:
        # Non-orthogonal inputs. Need to correct by (full) Gram matrix
        # Compute the Gram matrix and apply its (pseudo-)inverse to the obtained projections.
        # See the docstring for details.
        # The Gram matrix is just one additional diagonal contraction of the ``prod`` tensor,
        # across the Hilbert space dimensions. (dimg, _chi_, dimg, _chi_) -> (dimg, dimg)
        # This contraction is missing the normalization factor 1/chi of the trace inner product.
        gram_inv = qml.math.linalg.pinv(
            qml.math.real(qml.math.sum(qml.math.diagonal(prod, axis1=1, axis2=3), axis=-1))
        )
        # Axis ordering for contraction with gamma axis of raw structure constants:
        # (dimg, _dimg_), (_dimg_, dimg, dimg) -> (dimg, dimg, dim)
        # Here we add the missing normalization factor of the trace inner product (after inversion)
        adj = qml.math.tensordot(gram_inv * chi, adj, axes=1)

    return adj
