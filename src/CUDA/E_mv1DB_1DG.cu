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
#include "../ellpack.h"

// ----------------------- Global variables ----------------------- //

#define XBD 64
const dim3 BLOCK_DIM(XBD);
int ntimes = 20;

// ----------------------- Function prototypes ----------------------- //

// Simple CPU implementation of matrix multiplication.
void CpuMatrixMultiply(const ellpack* A, const double* B, double*  C)
{
    for (int i = 0; i < A->M; i++) {
        double t0 = 0.0;
        for (int j = 0; j < A->max_nz; j++) {
            if (A->JA[i][j] == -1) break;
            t0 += A->AS[i][j] * B[A->JA[i][j]];
        }
        C[i] = t0;
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

// GPU implementation of matrix_vector product 1D block 1D grid
__global__ void gpuMatrixVector(const int *JA, const double* AS, const double* B, double*  C, int M, int max_nz) {

    //create shared memory
    __shared__ double sdata[XBD];
    int tc     = threadIdx.x;
    int row    = blockIdx.x;

    int s;
    sdata[tc] = 0.0;

    if (row < M) {
        // Starting address of indexing within matrix A
        int idxm = row * max_nz + tc;
        int ic   = tc;
        double t  = 0.0;
        for ( ; ic < max_nz; ic += XBD) {
            if (JA[idxm] == -1) break;
            t += AS[idxm] * B[JA[idxm]];
            idxm += XBD;
        }
        sdata[tc] = t;
    }
    __syncthreads();

    //reduce
    for (s = XBD/2; s >= 32; s >>= 1) {
        if (tc < s) {
            sdata[tc] += sdata[tc + s];
        }
        __syncthreads();
    }
    s = min(16,XBD/2);
    if (tc < s) rowReduce(sdata, tc, s);
    if (tc == 0) C[row] = sdata[0];
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr,"Usage: %s  filename\n",argv[0]);
    }

    ellpack* A = new ellpack(argv[1]);

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

    //convert JA and AS to 1D arrays
    int *h_JA = (int*) malloc(nrowsc * A->max_nz * sizeof(int));
    double *h_AS = (double*) malloc(nrowsc * A->max_nz * sizeof(double));
    for (int i = 0; i < nrowsc; i++) {
        for (int j = 0; j < A->max_nz; j++) {
            h_JA[i*A->max_nz + j] = A->JA[i][j];
            h_AS[i*A->max_nz + j] = A->AS[i][j];
        }
    }

    // ---------------------- Device memory initialisation ---------------------- //

    int *d_JA;
    double *d_AS, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void**) &d_JA, nrowsc * A->max_nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &d_AS, nrowsc * A->max_nz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_B, ncolsc * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**) &d_C, nrowsc * sizeof(double)));

    // Copy matrices from the host (CPU) to the device (GPU).
    checkCudaErrors(cudaMemcpy(d_JA, h_JA, nrowsc * A->max_nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, h_AS, nrowsc * A->max_nz * sizeof(double), cudaMemcpyHostToDevice));
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
        gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>> (d_JA, d_AS, d_B, d_C, nrowsc, A->max_nz);
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

    std::cout << "ELLPACK CUDA 1DB 1DG \t\t Time " << bestTime << " ms \t GFLOPS " << gpuflops << " \t Max error " << max_error << "\t Block size " << XBD << "x1 " << "Grid size " << GRID_DIM.x << "x" << GRID_DIM.y << std::endl;

    // ------------------------------- Cleaning up ------------------------------ //

    delete timer;
    // Free the device memory.
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    // Free the host memory.
    free(h_JA);
    free(h_AS);
    free(h_B);
    free(h_C);
    free(h_C_proof);

    A->~ellpack();

    return 0;
}
