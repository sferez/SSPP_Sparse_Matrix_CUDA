#/***********************************************************************
#*  This code is part of the Small Scale Parallel Programming Assignment
#*
#* SS Assignment
#* Author:  Simeon FEREZ S392371
#* Date:    February-2023
#***********************************************************************/

# NAME CONVENTION

# C: CSR
# E: ELLPACK

# Serial: Native Implementation

# Unroll2H: Unroll 2 Horizontal
# Unroll4H: Unroll 4 Horizontal
# Unroll8H: Unroll 8 Horizontal
# Unroll16H: Unroll 16 Horizontal

# Unroll2V: Unroll 2 Vertical
# Unroll4V: Unroll 4 Vertical
# Unroll8V: Unroll 8 Vertical

# CSR matrix-vector multiplication
add_executable(C_Serial C_mvSerial.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll2H C_mvUnroll2H.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll2V C_mvUnroll2V.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll8H C_mvUnroll8H.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll16H C_mvUnroll16H.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll4V C_mvUnroll4V.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll8V C_mvUnroll8V.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(C_Unroll4H C_mvUnroll4H.cpp ../csr.h ../csr.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)

# ELLPACK matrix-vector multiplication
add_executable(E_Serial E_mvSerial.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll2H E_mvUnroll2H.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll2V E_mvUnroll2V.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll8H E_mvUnroll8H.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll16H E_mvUnroll16H.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll4V E_mvUnroll4V.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll8V E_mvUnroll8V.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)
add_executable(E_Unroll4H E_mvUnroll4H.cpp ../ellpack.h ../ellpack.cpp ../matrix.cpp ../matrix.h ../wtime/wtime.cpp ../wtime/wtime.h)