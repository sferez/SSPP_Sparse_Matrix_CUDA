
# Small Scale Parallel Computing

This study examined the performance of OpenMP and CUDA parallelization for Sparse matrix-vector multiplication (SpMV) on a hybrid CPU and GPU platform.

It implemented both programming models on a set of sparse matrices with varying sizes and densities.

## Report

The report is available at Medium: [Article](https://medium.com/@simeon.ferez/sparse-matrix-vector-product-openmp-vs-cuda-on-hybrid-platforms-7ef576fbfc0d)

![Article](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7AOf5Y2Sdi1yYTiJs9caNg.png)

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
- Python 3 (optional)

## Build

Each folder contains a CMakeLists.txt file to build the code. Each parallelization is built separately for testing purpose.

Python script have been provided to run all the configurations and generate the results.

## Authors

- [@sferez](https://github.com/sferez)