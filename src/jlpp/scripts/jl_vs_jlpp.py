#
# Copyright (c) 2022 IBM Inc. All rights reserved
# SPDX-License-Identifier: MIT
#

import numpy as np
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass
import click
import logging

from jlpp.random_projection import compute_row_norms_jl, compute_row_norms_jlpp

plt.rcParams.update({"font.size": 14})


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@dataclass
class ApproximationErrors:
    mean_frobenius: float
    stdev_frobenius: float
    mean_elementwise: float
    stdev_elementwise: float


def run_x_times(A, m, x=10):

    logger.info(f"Running both algorithms {x} times for {4*m} matrix-vector queries...")

    norms_A = np.linalg.norm(A, axis=1)
    sqnorms_A = norms_A * norms_A
    frobenius_norm_A = np.linalg.norm(A, "fro") ** 2

    frobenius_err_jl = []
    max_err_jl = []
    frobenius_err_jlpp = []
    max_err_jlpp = []

    for i in range(x):

        sqnorms_jlpp = compute_row_norms_jlpp(A, k=m, m=m)
        sqnorms_jl = compute_row_norms_jl(A, 4 * m)

        max_err_jl.append(np.max(np.abs(sqnorms_A - sqnorms_jl) / sqnorms_A))
        max_err_jlpp.append(np.max(np.abs(sqnorms_A - sqnorms_jlpp) / sqnorms_A))
        frobenius_err_jl.append(
            np.abs(frobenius_norm_A - np.sum(sqnorms_jl)) / frobenius_norm_A
        )
        frobenius_err_jlpp.append(
            np.abs(frobenius_norm_A - np.sum(sqnorms_jlpp)) / frobenius_norm_A
        )

    jl_error = ApproximationErrors(
        mean_frobenius=np.mean(frobenius_err_jl),
        stdev_frobenius=np.std(frobenius_err_jl),
        mean_elementwise=np.mean(max_err_jl),
        stdev_elementwise=np.std(max_err_jl),
    )
    jlpp_error = ApproximationErrors(
        mean_frobenius=np.mean(frobenius_err_jlpp),
        stdev_frobenius=np.std(frobenius_err_jlpp),
        mean_elementwise=np.mean(max_err_jlpp),
        stdev_elementwise=np.std(max_err_jlpp),
    )
    return jl_error, jlpp_error


class RandomMatrix:
    @staticmethod
    def random_rotation(D: np.ndarray):
        """
        Given a square matrix D, apply a (symmetric) random rotation on the left and right.
        """
        n = D.shape[0]
        A = np.random.randn(n, n)
        Q, _ = np.linalg.qr(A)
        return Q.dot(D).dot(Q.T)

    @staticmethod
    def get_symmetric_matrix_with_decay(n, c, rotate=False):
        """
        Construct a n*n matrix with decaying eigenvalues.
        lambda_i is equal to i^(-c).
        """
        D = np.arange(n) + 1
        D = pow(1 / D, c)
        if rotate is False:
            return np.diag(D), D
        else:
            return RandomMatrix.random_rotation(np.diag(D)), D


@click.command()
@click.option("--n", default=5000, help="Size of the matrix.", type=click.IntRange(1))
@click.option(
    "--c",
    default=0.5,
    help=(
        "Decay factor for the singular values of the matrix."
        " The i-th singular value is equal to i^(-c)."
    ),
    type=click.FloatRange(0.0),
)
@click.option(
    "--rotate",
    default=True,
    help="Whether to randonly rotate the column space of the test matrix or not.",
    type=bool,
)
@click.option(
    "--max-queries",
    default=1000,
    help=(
        "Maximum number of matrix-vector queries for both algorithms, "
        "passed to: np.arange(min_queries, max_queries+1, step)"
    ),
    type=click.IntRange(1),
)
@click.option(
    "--min-queries",
    default=100,
    help=(
        "Minimum number of matrix-vector queries for both algorithms, "
        "passed to: np.arange(min_queries, max_queries+1, step)"
    ),
    type=click.IntRange(1),
)
@click.option(
    "--step",
    default=100,
    help=(
        "Step between min and max number of queries for np.arange, "
        "passed to: np.arange(min_queries, max_queries+1, step)"
    ),
    type=click.IntRange(1),
)
@click.option(
    "--n-tries",
    default=10,
    help="Number of random tries per experiment.",
    type=click.IntRange(1),
)
def main(n, c, rotate, min_queries, max_queries, step, n_tries):

    A, sing_vals_A = RandomMatrix.get_symmetric_matrix_with_decay(n, c=c, rotate=rotate)
    jl_errors = []
    jlpp_errors = []
    m_values = np.arange(min_queries, max_queries + 1, step)
    m_over_4_values = [int(np.ceil(m_value / 4.0)) for m_value in m_values]
    logger.info(
        f"Requested matrix-vector queries: {[m_value for m_value in m_values]}."
    )
    logger.info(
        "Adjusted matrix-vector queries: "
        f"{[m_over_4 * 4 for m_over_4 in m_over_4_values]} "
        "(need to be divisible by 4)."
    )
    for p in m_over_4_values:
        jl_error, jlpp_error = run_x_times(A, p, x=n_tries)
        jl_errors.append(jl_error)
        jlpp_errors.append(jlpp_error)

    # plots
    capsize = 5
    markersize = 12
    m_values = [4 * v for v in m_over_4_values]
    m_values_str = [f"{v}" for v in m_values]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(
        m_values,
        [err.mean_frobenius for err in jl_errors],
        [err.stdev_frobenius for err in jl_errors],
        color="r",
        linestyle="-",
        marker="*",
        label="JL (norm-wise)",
        capsize=capsize,
        markersize=markersize,
    )
    ax.errorbar(
        m_values,
        [err.mean_frobenius for err in jlpp_errors],
        [err.stdev_frobenius for err in jlpp_errors],
        color="b",
        linestyle="-",
        marker="*",
        label="JL++ (norm-wise)",
        capsize=capsize,
        markersize=markersize,
    )
    ax.errorbar(
        m_values,
        [err.mean_elementwise for err in jl_errors],
        [err.stdev_elementwise for err in jl_errors],
        color="r",
        linestyle="-.",
        marker="x",
        label="JL (element-wise)",
        capsize=capsize,
        markersize=markersize,
    )
    ax.errorbar(
        m_values,
        [err.mean_elementwise for err in jlpp_errors],
        [err.stdev_elementwise for err in jlpp_errors],
        color="b",
        linestyle="-.",
        marker="x",
        label="JL++ (element-wise)",
        capsize=capsize,
        markersize=markersize,
    )

    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower center",
        ncol=2,
        borderaxespad=0.0,
    )
    ax.grid(which="both")
    ax.set_xticks(m_values, rotation=45)
    ax.set_xticklabels(m_values_str, rotation=45)
    ax.set_ylabel("Relative error")
    ax.set_xlabel("Matrix-vector multiplication queries m")

    plt.tight_layout()
    plt.savefig(f"n_{n}_rotate_{rotate}_c_{c}_n_tries_{n_tries}.png", format="png")
    plt.close()


if __name__ == "__main__":

    main()
