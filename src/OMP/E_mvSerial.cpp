/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "../wtime/wtime.h"
#include "../ellpack.h"
#include <random>

const int ntimes = 20;
inline double dmin ( double a, double b ) { return a < b ? a : b; }
inline int max ( int a, int b ) { return a > b ? a : b; }
inline int min ( int a, int b ) { return a < b ? a : b; }

// Simple CPU implementation of matrix multiplication.
void MatrixMultiply(const ellpack* A, const double* B, double*  C)
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

void MatrixMultiplyOMP(const ellpack* A, const double* B, double*  C, int chunk)
{
    // Multiply the matrix A by the vector B by CSR format and store the result in C
#pragma omp parallel for shared(A, B, C) schedule(static, chunk)
    for (int i = 0; i < A->M; i++) {
        double t0 = 0.0;
        for (int j = 0; j < A->max_nz; j++) {
            if (A->JA[i][j] == -1) break;
            t0 += A->AS[i][j] * B[A->JA[i][j]];
        }
        C[i] = t0;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,"Usage: %s  filename\n",argv[0]);
    }

    ellpack* A = new ellpack(argv[1]);

    // ----------------------- Host memory initialisation ----------------------- //

    int nrowsc = A->M;
    int ncolsc = A->N;

    double* B = (double*) malloc(ncolsc * sizeof(double));
    double* C = (double*) malloc(nrowsc * sizeof(double));
    double *C_proof = (double*) malloc(nrowsc * sizeof(double));

    // ----------------------- Matrix Initialization  ----------------------- //

    for (int i = 0; i < ncolsc; i++) {
        B[i] = 1.0;
    }
    for (int i = 0; i < nrowsc; i++) {
        C[i] = 0.0;
    }
    for (int i = 0; i < nrowsc; i++) {
        C_proof[i] = 0.0;
    }

    MatrixMultiply(A,B,C_proof);

    // ----------------------- Matrix multiplication ----------------------- //

    double tmlt = 1e100;
    int *CHUNK = new int[19];
    for (int i = 0; i < 19; i++) {
        CHUNK[i] = pow(2, i);
    }
    int id_chunk = 0;

    for (int i= 0; i < 19; i++) {
        for (int j = 0; j < ntimes; j++) {
            double t1 = wtime();
            MatrixMultiplyOMP(A, B, C, CHUNK[i]);
            double t2 = wtime();
            if(tmlt > (t2 - t1)) {
                tmlt = (t2 - t1);
                id_chunk = i;
            }
        }
    }

    // ----------------------- Check results ----------------------- //

    //find max error
    double max_error = 0;
    for (int i = 0; i < nrowsc; i++) {
        double error = C[i] - C_proof[i];
        if (error < 0) error = -error;
        if (error > max_error) max_error = error;
    }


    // ----------------------- Print results ----------------------- //

    fprintf(stdout,"Number of threads: %d \n",omp_get_max_threads());
    double mflops = 2.0 * A->nz / tmlt * 1e-6;
//    fprintf(stdout,"Multiplying matrices of size %d x %d (%d) : time %lf  MFLOPS %lf \n",
//            nrowsc,ncolsc,ncolsc,tmlt,mflops);
    fprintf(stdout,"M: %d \t N: %d \t nz: %d \n",A->M,A->N,A->nz);
    fprintf(stdout,"ELLPACK Serial \t Time: %lf \t MFLOPS %lf \t GFLOPS %lf \t Error %lf \t CHUNK %d \n",tmlt,mflops,mflops/1000,max_error,CHUNK[id_chunk]);
    // ----------------------- Free memory ----------------------- //

    A->~ellpack();
    free(B);
    free(C);

    return 0;
}
