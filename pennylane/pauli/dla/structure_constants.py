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
import warnings
from typing import Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.typing import TensorLike

from ..pauli_arithmetic import PauliSentence, PauliWord


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

    .. seealso:: :func:`~lie_closure`, :func:`~center`, :class:`~pennylane.pauli.PauliVSpace`, `Demo: Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__

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
    warnings.warn(
        "Calling center via ``qml.pauli.structure_constants`` is deprecated. ``structure_constants`` has moved to ``pennylane.liealg``. "
        "Please call ``structure_constants`` from top level as ``qml.structure_constants`` or from the liealg module via ``qml.liealg.structure_constants``.",
        qml.PennyLaneDeprecationWarning,
    )

    return qml.structure_constants(g, pauli=pauli, matrix=matrix, is_orthogonal=is_orthogonal)
