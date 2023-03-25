/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#ifndef SS_ASSIGNEMENT_ELLPACK_H
#define SS_ASSIGNEMENT_ELLPACK_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <utility>
#include <tuple>
#include <vector>
#include "matrix.h"

class ellpack : public matrix{

public:
    alignas(64) int **JA;
    alignas(64) double **AS;
    int M, N, nz, max_nz;

    explicit ellpack(char *filename);
    void create_ellpack_matrix(char *filename);
    void print_ellpack_matrix() const;
    static void print_vector(int *v, int size);
    static void print_vector(double *v, int size);
    void createPNG(char *filename) const;
    ~ellpack();

private:
    void convert_coo_to_ellpack( int* I_,  int* J_, double* val_);
    void sort_coo(int* I_, int* J_, double* val_) const;
};


#endif //SS_ASSIGNEMENT_ELLPACK_H
