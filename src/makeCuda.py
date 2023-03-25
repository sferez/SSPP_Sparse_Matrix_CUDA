# /***********************************************************************
# *  This code is part of the Small Scale Parallel Programming Assignment
# *
# * SS Assignment
# * Author:  Simeon FEREZ S392371
# * Date:    February-2023
# ***********************************************************************/

import os
import sys

# ELLPACK
os.system("make E_mv1DB_1DG") # 1D block ELLPACK
os.system("make E_mv1DB_1DG_2H")
os.system("make E_mv1DB_1DG_4H")

os.system("make E_mv2DB_1DG") # 2D block ELLPACK
os.system("make E_mv2DB_1DG_2H")
os.system("make E_mv2DB_1DG_4H")

# CSR
os.system("make C_mv1DB_1DG") # 1D block CSR
os.system("make C_mv1DB_1DG_2H")
os.system("make C_mv1DB_1DG_4H")

os.system("make C_mv2DB_1DG") # 2D block CSR
os.system("make C_mv2DB_1DG_2H")
os.system("make C_mv2DB_1DG_4H")



