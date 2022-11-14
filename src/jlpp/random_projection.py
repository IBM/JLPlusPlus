#
# Copyright (c) 2022 IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


def compute_row_norms_jl(
    A: Union[np.ndarray, csr_matrix, LinearOperator], m: int
) -> np.ndarray:
    """
    Given a matrix A, return the squared Euclidean norms of the rows of the
    matrix A*G, where G is a n*m matrix with i.i.d. elements from the distribution
    ~N(0,1/sqrt(m)). The returned values are approximations to the squared Euclidean
    norms of the rows of A.

    Args:
        A: input matrix given either as a dense matrix or sparse matrix or linear operator.
        m (int): number of columns of G.
    Returns:
        Vector containing the squared Euclidean norms of the rows of A*G.
    """

    AG = A.dot(np.random.randn(A.shape[1], m)) / np.sqrt(m)
    return np.sum(AG * AG, axis=1)


def compute_row_norms_jlpp(
    A: Union[np.ndarray, csr_matrix, LinearOperator], k: int, m: int
) -> np.ndarray:
    """
    Given a matrix A, compute approximations of the squared Euclidean norms of the rows A
    using adaptive techniques that lead to faster convergence (less matrix-vector queries)
    than standard Johnson-Lindenstrauss random projections (like the ones used in sqnorms_jl).
    The key idea is to first find a good approximation of the top singular subspace of A
    and then use JL approximations only on the orthogonal complement of that subspace.
    For more details see A. Sobczyk & M. Luisier, NeurIPS 2022:
    "Approximate Euclidean lengths and distances beyond Johnson-Lindenstrauss"
    https://openreview.net/forum?id=_N4k45mtnuq

    Args:
        A: input matrix given either as a dense matrix or sparse matrix or linear operator.
        k: dimension of the top singular subspace to approximate.
        m: number of columns of the Gaussian random projection matrix
    Returns:
        Vector containing the approximations of the squared Euclidean norms of A.
    """
    S = np.random.randn(A.shape[1], k) / np.sqrt(k)
    B = A.T.dot(A.dot(S))
    Q, R = np.linalg.qr(B)

    G = np.random.randn(A.shape[1], m) / np.sqrt(m)
    D = A.dot(G - Q.dot(Q.T.dot(G)))
    AQ = A.dot(Q)
    return np.sum(AQ * AQ, axis=1) + np.sum(D * D, axis=1)
