# jlplusplus
Python code to reproduce the experimental evaluation of the algorithms proposed in the NeurIPS 2022 publication "Approximate Euclidean lengths and distances beyond Johnson-Lindenstrauss".
If you download or use this code for research purposes please cite the corresponding [publication](https://openreview.net/forum?id=_N4k45mtnuq):

```
@inproceedings{
sobczyk2022approximate,
title={Approximate Euclidean lengths and distances beyond Johnson-Lindenstrauss},
author={Aleksandros Sobczyk and Mathieu Luisier},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=_N4k45mtnuq}
}
```

## Installation
The Python package and the exeutable script can be installed directly with pip:
```
pip install git+https://github.com/IBM/JLPlusPlus
```

## Usage
An executable script, `compare-jl-vs-jlpp`, is installed along with the package, which compare the approximation accuracy of the proposed algorithms with standard Johnson-Lindenstrauss approximations.
Several options can be configured by passing the corresponding arguments arguments:
```
compare-jl-vs-jlpp --help
Usage: compare-jl-vs-jlpp [OPTIONS]

Options:
  --n INTEGER RANGE            Size of the matrix.  [x>=1]
  --c FLOAT RANGE              Decay factor for the singular values of the
                               matrix. The i-th singular value is equal to
                               i^(-c).  [x>=0.0]
  --rotate BOOLEAN             Whether to randonly rotate the column space of
                               the test matrix or not.
  --max-queries INTEGER RANGE  Maximum number of matrix-vector queries for
                               both algorithms, passed to:
                               np.arange(min_queries, max_queries+1, step)
                               [x>=1]
  --min-queries INTEGER RANGE  Minimum number of matrix-vector queries for
                               both algorithms, passed to:
                               np.arange(min_queries, max_queries+1, step)
                               [x>=1]
  --step INTEGER RANGE         Step between min and max number of queries for
                               np.arange, passed to: np.arange(min_queries,
                               max_queries+1, step)  [x>=1]
  --n-tries INTEGER RANGE      Number of random tries per experiment.  [x>=1]
  --help                       Show this message and exit.
```
Running the script will produce a `.png` image with the convergence comparison of the two methods.
For more details on the algorithms see [here](https://openreview.net/forum?id=_N4k45mtnuq).