/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#include "ellpack.h"

ellpack::ellpack(char *filename) {
    create_ellpack_matrix(filename);
}

void ellpack::print_vector(int *v, int size) {

    for(int i = 0; i < size; i++){
        std::cout << v[i] << ' ';
    }
    std::cout << '\n';
}

void ellpack::print_vector(double *v, int size) {

    for(int i = 0; i < size; i++){
        printf("%.2f ", v[i]);
    }
    std::cout << '\n';
}


void ellpack::create_ellpack_matrix(char *filename) {
    //read matrix from file using mmio
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    //get size matrix
    if ((f = fopen(filename, "r")) == NULL){
        std::cout << "Could not open file" << '\n';
        exit(1);
    }

    //initialize matcode
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner");
        exit(1);
    }

    //initialize M, N, nz
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(1);

    int *I_temp = new int[nz];
    int *J_temp = new int[nz];
    double *AS_temp = new double[nz];

    mm_read_mtx_crd_data(f, M, N, nz, I_temp, J_temp, AS_temp, matcode);

    //recalculate nz
    if (mm_is_symmetric(matcode)){
        int min_nz=0;
        for (int i = 0; i < nz; i++){
            if (I_temp[i] == J_temp[i]){
                min_nz++;
            }
        }
        int old_nz = nz;
        nz = 2*nz- min_nz;

        int *I_temp2 = new int[nz];
        int *J_temp2 = new int[nz];
        double *AS_temp2 = new double[nz];

        int k = 0;
        for (int i = 0; i < old_nz; i++){
            I_temp2[k] = I_temp[i];
            J_temp2[k] = J_temp[i];
            AS_temp2[k] = AS_temp[i];
            k++;
        }
        for (int i = 0; i < old_nz; i++){
            if (I_temp[i] != J_temp[i]){
                I_temp2[k] = J_temp[i];
                J_temp2[k] = I_temp[i];
                AS_temp2[k] = AS_temp[i];
                k++;
            }
        }

        free(I_temp);
        free(J_temp);
        free(AS_temp);

        I_temp = I_temp2;
        J_temp = J_temp2;
        AS_temp = AS_temp2;
    }


    sort_coo(I_temp, J_temp, AS_temp);

    //find max_nz
    max_nz = 0;
    int count = 1;
    for (int i = 0; i < nz-1; i++){
        if (I_temp[i] == I_temp[i+1]){
            count++;
        } else{
            if (count > max_nz){
                max_nz = count;
            }
            count = 1;
        }
    }

    convert_coo_to_ellpack(I_temp, J_temp, AS_temp);

    free(I_temp);
    free(J_temp);
    free(AS_temp);
}

void ellpack::print_ellpack_matrix() const {

    int count = 0;
    for (int i = 0; i < M; i++ ){
        count = 0;
        for ( int j = 0 ; j < N ; j++ ){
            if( JA[i][count] != j ){
                std::cout << ".... ";
            } else{
                printf("%.2f ", AS[i][count]);
                count++;
            }
        }
        std::cout << '\n';
    }
}

void ellpack::convert_coo_to_ellpack(int *I_, int *J_, double *val_) {

    //create JA and AS
    JA = new int*[M];
    AS = new double*[M];

    for (int i = 0; i < M; i++){
        JA[i] = new int[max_nz];
        AS[i] = new double[max_nz];
    }

    //complete JA and AS
    int count = 0;
    for (int i = 0; i < M; i++){
        for (int j = 0; j < max_nz; j++){
            if (I_[count] == i){
                JA[i][j] = J_[count];
                AS[i][j] = val_[count];
                count++;
            } else{
                JA[i][j] = -1;
                AS[i][j] = 0;
            }
        }
    }
}

void ellpack::sort_coo(int *I_, int *J_, double *AS_) const {

    std::vector<int> index(nz);
    for (int i = 0; i < nz; ++i) {
        index[i] = i;
    }

// sort the I_temp and index arrays in parallel
    std::sort(index.begin(), index.end(), [&](int a, int b) {
        return I_[a] < I_[b] ||
               (I_[a] == I_[b]) && J_[a] < J_[b];
    });

// create temporary arrays to hold the sorted J_temp and AS_temp
    std::vector<int> J_temp_sorted(nz);
    std::vector<double> AS_temp_sorted(nz);
    std::vector<int> I_temp_sorted(nz);

// recompose the J_temp and AS_temp arrays based on the sorted index array
    for (int i = 0; i < nz; ++i) {
        int j = index[i];
        I_temp_sorted[i] = I_[j]-1;
        J_temp_sorted[i] = J_[j]-1;
        AS_temp_sorted[i] = AS_[j];
    }

// overwrite the original J_temp and AS_temp arrays with the sorted versions
    std::copy(I_temp_sorted.begin(), I_temp_sorted.end(), I_);
    std::copy(J_temp_sorted.begin(), J_temp_sorted.end(), J_);
    std::copy(AS_temp_sorted.begin(), AS_temp_sorted.end(), AS_);

    //free memory
    index.clear();
    index.shrink_to_fit();
    J_temp_sorted.clear();
    J_temp_sorted.shrink_to_fit();
    AS_temp_sorted.clear();
    AS_temp_sorted.shrink_to_fit();
    I_temp_sorted.clear();
    I_temp_sorted.shrink_to_fit();
}

ellpack::~ellpack() {
    free(JA);
    free(AS);
}