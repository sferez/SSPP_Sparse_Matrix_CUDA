/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#include "csr.h"

csr::csr(char *filename) {
    create_csr_matrix(filename);
}

void csr::print_vector(int *v, int size) {

    for(int i = 0; i < size; i++){
        std::cout << v[i] << ' ';
    }
    std::cout << '\n';
}

void csr::print_vector(double *v, int size) {

    for(int i = 0; i < size; i++){
        printf("%.2f ", v[i]);
    }
    std::cout << '\n';
}

void csr::create_csr_matrix(char *filename) {
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

//    std::cout << "Reading matrix from file..." << '\n';
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

//    std::cout << "Sorting... " << nz << '\n';
    sort_coo(I_temp, J_temp, AS_temp);

//    std::cout << "Converting to CSR format..." << '\n';
    convert_coo_to_csr(I_temp, J_temp, AS_temp);

    free(I_temp);
    free(J_temp);
    free(AS_temp);
}

void csr::print_csr_matrix() const {

    //print full matrix
    int k = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(j == JA[k]){
                printf("%.2f ", AS[k]);
//                std::cout << "X" ;
                k++;
            }
            else
                std::cout << ".... " ;
        }
        std::cout << '\n';
    }
}

void csr::convert_coo_to_csr(const int *I_, const int *J_, double *AS_) {

    //allocate memory for I, J, AS
    IRP = new int[M+1];
    JA = new int[nz];
    AS = new double[nz];

    int last_row = 0;
    IRP[0]=0;
    int count=1;
    for (int i = 0; i < nz; i++){
        JA[i]=J_[i];
        AS[i]=AS_[i];
        if (I_[i] > last_row){
            last_row = I_[i];
            IRP[count]=i;
            count++;
        }
    }
    IRP[count]=nz;
}


void csr::sort_coo(int *I_, int *J_, double *AS_) const {

//    // create a vector of tuples to store I_temp, J_temp, and AS_temp
//    std::vector<std::tuple<int, int, double>> data;
//    data.reserve(nz);
//
//    for (int i = 0; i < nz; ++i) {
//        //decrement J_temp by 1
//        J_[i] = J_[i] - 1;
//        I_[i] = I_[i] - 1;
//        data.emplace_back(I_[i], J_[i], AS_[i]);
//    }
//
//    // sort the data vector based on I_temp
//    std::sort(data.begin(), data.end(), [](std::tuple<int, int, double> a, std::tuple<int, int, double> b) {
//        return std::get<0>(a) < std::get<0>(b) ||
//               (std::get<0>(a) == std::get<0>(b)) && std::get<1>(a) < std::get<1>(b);
//    });
//
//    // update I, J, and AS based on the sorted data
//    for (int i = 0; i < nz; ++i) {
//        I_[i] = std::get<0>(data[i]);
//        J_[i] = std::get<1>(data[i]);
//        AS_[i] = std::get<2>(data[i]);
//    }
//
//    //free memory
//    data.clear();
//    data.shrink_to_fit();
// create an index array to keep track of the original indices
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

void csr::createPNG(char *filename) const {

    //write ppm image P1
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s for writing \n", filename);
        exit(1);
    }

    fprintf(fp, "P1 \n");
    fprintf(fp, "%d %d \n", M, N);

    //print full matrix
    int k = 0;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if(j == JA[k]){
                fprintf(fp, "1");
                k++;
            }
            else
                fprintf(fp, "0");
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

csr::~csr() {
    delete[] IRP;
    delete[] JA;
    delete[] AS;

    IRP = nullptr;
    JA = nullptr;
    AS = nullptr;
}





