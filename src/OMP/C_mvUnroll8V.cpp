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
#include "../csr.h"
#include <random>

const int ntimes = 20;
inline double dmin ( double a, double b ) { return a < b ? a : b; }
inline int max ( int a, int b ) { return a > b ? a : b; }
inline int min ( int a, int b ) { return a < b ? a : b; }

// Simple CPU implementation of matrix multiplication.
void MatrixMultiply(const csr* A, const double* B, double*  C)
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

void MatrixMultiplyOMP(const csr* A, const double* B, double*  C , int chunk) {

    int i, j0, j1, j2, j3, j4, j5, j6, j7;
    double t0, t1, t2, t3, t4, t5, t6, t7;
#pragma omp parallel for shared(C,B,A) private(i,j0,j1,j2,j3,j4,j5,j6,j7,t0,t1,t2,t3,t4,t5,t6,t7) schedule(static,chunk)
    for (i = 0; i < A->M - 7; i += 8) {
        t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0;
        j0 = A->IRP[i];
        j1 = A->IRP[i+1];
        j2 = A->IRP[i+2];
        j3 = A->IRP[i+3];
        j4 = A->IRP[i+4];
        j5 = A->IRP[i+5];
        j6 = A->IRP[i+6];
        j7 = A->IRP[i+7];
        for (;  j0 < A->IRP[i+1]; j0++) {
            t0 += B[A->JA[j0]] * A->AS[j0];
            if (j1<A->IRP[i+2]) {
                t1 += B[A->JA[j1]] * A->AS[j1];
                j1++;
            }
            if (j2<A->IRP[i+3]) {
                t2 += B[A->JA[j2]] * A->AS[j2];
                j2++;
            }
            if (j3<A->IRP[i+4]) {
                t3 += B[A->JA[j3]] * A->AS[j3];
                j3++;
            }
            if (j4<A->IRP[i+5]) {
                t4 += B[A->JA[j4]] * A->AS[j4];
                j4++;
            }
            if (j5<A->IRP[i+6]) {
                t5 += B[A->JA[j5]] * A->AS[j5];
                j5++;
            }
            if (j6<A->IRP[i+7]) {
                t6 += B[A->JA[j6]] * A->AS[j6];
                j6++;
            }
            if (j7<A->IRP[i+8]) {
                t7 += B[A->JA[j7]] * A->AS[j7];
                j7++;
            }
        }
        if (j1<A->IRP[i+2]) {
            for (; j1 < A->IRP[i+2]; j1++) {
                t1 += B[A->JA[j1]] * A->AS[j1];
            }
        }
        if (j2<A->IRP[i+3]) {
            for (; j2 < A->IRP[i+3]; j2++) {
                t2 += B[A->JA[j2]] * A->AS[j2];
            }
        }
        j3 = j2;
        if (j3<A->IRP[i+4]) {
            t3 = 0;
            for (; j3 < A->IRP[i+4]; j3++) {
                t3 += B[A->JA[j3]] * A->AS[j3];
            }
        }
        j4 = j3;
        if (j4<A->IRP[i+5]) {
            t4 = 0;
            for (; j4 < A->IRP[i+5]; j4++) {
                t4 += B[A->JA[j4]] * A->AS[j4];
            }
        }
        j5 = j4;
        if (j5<A->IRP[i+6]) {
            t5 = 0;
            for (; j5 < A->IRP[i+6]; j5++) {
                t5 += B[A->JA[j5]] * A->AS[j5];
            }
        }
        j6 = j5;
        if (j6<A->IRP[i+7]) {
            t6 = 0;
            for (; j6 < A->IRP[i+7]; j6++) {
                t6 += B[A->JA[j6]] * A->AS[j6];
            }
        }
        j7 = j6;
        if (j7<A->IRP[i+8]) {
            t7 = 0;
            for (; j7 < A->IRP[i+8]; j7++) {
                t7 += B[A->JA[j7]] * A->AS[j7];
            }
        }
        C[i] = t0;
        C[i+1] = t1;
        C[i+2] = t2;
        C[i+3] = t3;
        C[i+4] = t4;
        C[i+5] = t5;
        C[i+6] = t6;
        C[i+7] = t7;
    }
    if (A->M % 8 != 0) {
        for (i = A->M - A->M % 8; i < A->M; i++) {
            t0 = 0;
            for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
                t0 += B[A->JA[j]] * A->AS[j];
            }
            C[i] = t0;
        }
    }
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,"Usage: %s  filename\n",argv[0]);
    }

    // ----------------------- IDs VERSION ----------------------- //

//    fprintf(stdout,"ID: 100, Unroll8V\n");

    // ----------------------- Read matrix from file ----------------------- //

    csr* A = new csr(argv[1]);

    // ----------------------- Host memory initialisation ----------------------- //

    int nrowsc = A->M;
    int ncolsc = A->N;

    double* B = (double*) malloc(ncolsc * sizeof(double));
    double* C = (double*) malloc(nrowsc * sizeof(double));
    double *C_proof = (double*) malloc(nrowsc * sizeof(double));

    // ----------------------- Matrix Initialization  ----------------------- //

    //random number generator

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    for (int i = 0; i < ncolsc; i++) {
        B[i] = 1;
    }
    for (int i = 0; i < nrowsc; i++) {
        C[i] = 0.0;
    }
    for (int i = 0; i < nrowsc; i++) {
        C_proof[i] = 0.0;
    }

    // -------------------- Matrix multiplication SERIAL -------------------- //

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

//    //print results 2 last elements
//    for (int i = 0; i < nrowsc; i++) {
//        fprintf(stdout,"C[%d] = %lf \t C_proof[%d] = %lf \t error = %lf \n",i,C[i],i,C_proof[i],C[i]-C_proof[i]);
//    }


    // ----------------------- Print results ----------------------- //

    double mflops = 2.0 * A->nz / tmlt * 1e-6;
    //print number of threads
    #pragma omp parallel
        {
        #pragma omp master
            {
//            fprintf(stdout,"Number of threads: %d \n",omp_get_max_threads());
//            fprintf(stdout,"M: %d \t N: %d \t nz: %d \n",A->M,A->N,A->nz);
//            fprintf(stdout,"Multiplying matrices of size %d x %d (%d) : time %lf  MFLOPS %lf GFLOPS %lf\n",
//                    nrowsc,ncolsc,ncolsc,tmlt,mflops,mflops/1000);
//            fprintf(stdout,"Max error: %lf\n",max_error);
            fprintf(stdout,"CSR Unroll 8V \t Time: %lf \t MFLOPS %lf \t GFLOPS %lf \t Error %lf \t CHUNK %d \n",tmlt,mflops,mflops/1000,max_error,CHUNK[id_chunk]);
            }
        }

    // ----------------------- Free memory ----------------------- //

    A->~csr();
    free(B);
    free(C);

    return 0;
}
