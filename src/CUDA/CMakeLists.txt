#/***********************************************************************
#*  This code is part of the Small Scale Parallel Programming Assignment
#*
#* SS Assignment
#* Author:  Simeon FEREZ S392371
#* Date:    February-2023
#***********************************************************************/

# NAME CONVENTION

# E = ELLPACK
# C = CSR

# mv = matrix-vector multiplication

# 1DB = 1D BLOCK
# 2DB = 2D BLOCK

# 1DG = 1D GRID
# 2DG = 2D GRID

# _2H = UNROLLING LEVEL 2 HORIZONTAL
# _4H = UNROLLING LEVEL 4 HORIZONTAL
#
# LOAD CUDA ENVIRONMENT
set (CC 35)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

# ELLPACK CUDA
cuda_add_executable(E_mv1DB_1DG E_mv1DB_1DG.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 1D BLOCK 1D GRID
cuda_add_executable(E_mv1DB_1DG_2H E_mv1DB_1DG_2H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp)
cuda_add_executable(E_mv1DB_1DG_4H E_mv1DB_1DG_4H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp)

cuda_add_executable(E_mv2DB_1DG E_mv2DB_1DG.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 2D BLOCK 1D GRID
cuda_add_executable(E_mv2DB_1DG_2H E_mv2DB_1DG_2H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp)
cuda_add_executable(E_mv2DB_1DG_4H E_mv2DB_1DG_4H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp)

cuda_add_executable(E_mv1Thread E_mv1Thread.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 1 THREAD
cuda_add_executable(E_mv1ThreadT E_mv1ThreadT.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 1 THREAD Transposed
cuda_add_executable(E_mv1ThreadT_2H E_mv1ThreadT_2H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 1 THREAD UNROLLING LEVEL 2 HORIZONTAL
cuda_add_executable(E_mv1ThreadT_4H E_mv1ThreadT_4H.cu ../ellpack.h ../ellpack.cpp ../matrix.h ../matrix.cpp) # 1 THREAD UNROLLING LEVEL 4 HORIZONTAL

# CSR CUDA
cuda_add_executable(C_mv1DB_1DG C_mv1DB_1DG.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp) # 1D BLOCK 1D GRID
cuda_add_executable(C_mv1DB_1DG_2H C_mv1DB_1DG_2H.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp)
cuda_add_executable(C_mv1DB_1DG_4H C_mv1DB_1DG_4H.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp)

cuda_add_executable(C_mv2DB_1DG C_mv2DB_1DG.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp) # 2D BLOCK 1D GRID
cuda_add_executable(C_mv2DB_1DG_2H C_mv2DB_1DG_2H.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp)
cuda_add_executable(C_mv2DB_1DG_4H C_mv2DB_1DG_4H.cu ../csr.h ../csr.cpp ../matrix.h ../matrix.cpp)
