/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#include <iostream>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include "../csr.h"

// ----------------------- Global variables ----------------------- //

#define XBD 64
#define YBD 16
const dim3 BLOCK_DIM(XBD,YBD);
int ntimes = 20;

// ----------------------- Function prototypes ----------------------- //

// Simple CPU implementation of matrix multiplication.
void CpuMatrixMultiply(const csr* A, const double* B, double*  C)
{
// Multiply the matrix A by the vector B by CSR format and store the result in C
    for (int i = 0; i < A->M; i++) {
        double t = 0;
        for (int j = A->IRP[i]; j < A->IRP[i+1]; j++) {
            t += B[A->JA[j]] * A->AS[j];
        }
        C[i] = t;
    }
}

__device__ void rowReduce(volatile double *sdata, int tid, int s) {
    switch(s){
        case 16:  sdata[tid] += sdata[tid + 16];
        case  8:  sdata[tid] += sdata[tid +  8];
        case  4:  sdata[tid] += sdata[tid +  4];
        case  2:  sdata[tid] += sdata[tid +  2];
        case  1:  sdata[tid] += sdata[tid +  1];
    }
}

// GPU implementation of matrix_vector product 2D block 1D grid
__global__ void gpuMatrixVector(const int *IRP, const int *JA, const double* AS, const double* B, double*  C, int M) {

    __shared__ double ax[YBD][XBD];
    int tr     = threadIdx.y;
    int tc     = threadIdx.x;
    int row    = blockIdx.x*blockDim.y + tr;
    int offset = IRP[row];
    int end    = IRP[row+1];
    int i      = 0;

    int s;
    ax[tr][tc] = 0.0;

    if (row < M) {
        for (i = offset + tc; i < (end-XBD); i += XBD*2) {
            ax[tr][tc] += AS[i] * B[JA[i]] + AS[i+XBD] * B[JA[i+XBD]];
        }
        for (; i < end; i += XBD) {
            ax[tr][tc] += AS[i] * B[JA[i]];
        }
    }
    __syncthreads();
    // Reduce the row
    for (s=XBD/2; s >=32; s >>=1){
        if (tc<s)
            ax[tr][tc] += ax[tr][tc+s];
        __syncthreads();
    }

    s = min(16,XBD/2);
    if (tc < s) rowReduce(&(ax[tr][0]),tc,s);
    // Write the result
    if (tc == 0) {
        C[row] = ax[tr][0];
    }
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr,"Usage: %s  filename\n",argv[0]);
    }

    csr* A = new csr(argv[1]);

    // ----------------------- Host memory initialisation ----------------------- //

    int nrowsc = A->M;
    int ncolsc = A->N;

    double* h_B = (double*) malloc(ncolsc * sizeof(double));
    double* h_C = (double*) malloc(nrowsc * sizeof(double));
    double *h_C_proof = (double*) malloc(nrowsc * sizeof(double));

    // ----------------------- Matrix Initialization  ----------------------- //

    for (int i = 0; i < ncolsc; i++) {
        h_B[i] = 1.0;
    }
    for (int i = 0; i < nrowsc; i++) {
        h_C[i] = 0.0;
    }
    for (int i = 0; i < nrowsc; i++) {
        h_C_proof[i] = 0.0;
    }

    // ---------------------- Device memory initialisation ---------------------- //

    int *d_JA, *d_IRP;
    double *d_AS, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void**) &d_B, ncolsc * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_C, nrowsc * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_JA, A->nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_IRP, (A->M+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_AS, A->nz * sizeof(double)));

    // Copy matrices from the host (CPU) to the device (GPU).
    checkCudaErrors(cudaMemcpy(d_JA, A->JA, A->nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_IRP, A->IRP, (A->M+1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, A->AS, A->nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, ncolsc * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C, nrowsc * sizeof(double), cudaMemcpyHostToDevice));

    // ------------------------ Calculations on the CPU ------------------------- //

    CpuMatrixMultiply(A,h_B,h_C_proof);

    // ------------------------ Calculations on the GPU ------------------------- //

    dim3 GRID_DIM((nrowsc + BLOCK_DIM.y - 1)/BLOCK_DIM.y, 1, 1);

    double bestTime = 1e30;
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);
    for (int i = 0; i < ntimes; i++) {
        timer->reset();
        timer->start();
        gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>> (d_IRP, d_JA, d_AS, d_B, d_C, nrowsc);
        cudaDeviceSynchronize();
        timer->stop();
        bestTime = min(bestTime, timer->getTime());
    }

    float gpuflops = 2.e-6*A->nz/ timer->getTime();

    // ----------------------- Check results ----------------------- //

    // Download the resulting vector d_y from the device and store it in h_y_d.
    checkCudaErrors(cudaMemcpy(h_C, d_C, nrowsc * sizeof(double), cudaMemcpyDeviceToHost));
    //find max error
    double max_error = 0;
    for (int i = 0; i < nrowsc; i++) {
        double error = h_C[i] - h_C_proof[i];
        if (error < 0) error = -error;
        if (error > max_error) {max_error = error;}
    }

    // ------------------------------- Print results ------------------------------ //

    std::cout << "CSR CUDA 2DB 1DG Unroll 2H\t Time " << bestTime << " ms \t GFLOPS " << gpuflops << " \t Max error " << max_error << "\t Block size " << XBD << "x" << YBD << "Grid size " << GRID_DIM.x << "x" << GRID_DIM.y << std::endl;

    // ------------------------------- Cleaning up ------------------------------ //

    delete timer;
    // Free the device memory.
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_IRP));

    // Free the host memory.

    free(h_B);
    free(h_C);
    free(h_C_proof);

    A->~csr();

    return 0;
}
