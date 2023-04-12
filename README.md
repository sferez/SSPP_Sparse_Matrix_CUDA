
# Small Scale Parallel Computing

This study examined the performance of OpenMP and CUDA for Sparse matrix-vector multiplication (SpMV) on a hybrid CPU and GPU platform.

It implemented both programming models on a set of sparse matrices with varying sizes and densities.

## Report

The report is available at Medium: [Article]()

![Article]()

## Structure

The following code folders are included in this project:

- 'input' contains the input matrices in matrix market format
- 'output' contains the sparsity pattern of some output matrices
- 'src/CUDA' contains the CUDA code parallelization
- 'src/OMP' contains the OpenMP code parallelization
- 'src' contains the custom class and python script to run the code

## Requirements

- CMake
- CUDA
- OpenMP
- C++ compiler
- Python 3

## Build

Each folder contains a CMakeLists.txt file to build the code. Each parallelization is built separately.

## Authors

- [@sferez](https://github.com/sferez)