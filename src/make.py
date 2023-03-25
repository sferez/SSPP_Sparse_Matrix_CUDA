# /***********************************************************************
# *  This code is part of the Small Scale Parallel Programming Assignment
# *
# * SS Assignment
# * Author:  Simeon FEREZ S392371
# * Date:    February-2023
# ***********************************************************************/

import os
import sys


os.system("make C_Serial") # CSR
os.system("make C_Unroll2V")
os.system("make C_Unroll4V")
os.system("make C_Unroll8V")
os.system("make C_Unroll2H")
os.system("make C_Unroll4H")
os.system("make C_Unroll8H")
os.system("make C_Unroll16H")

os.system("make E_Serial") # ELLPACK
os.system("make E_Unroll2V")
os.system("make E_Unroll4V")
os.system("make E_Unroll8V")
os.system("make E_Unroll2H")
os.system("make E_Unroll4H")
os.system("make E_Unroll8H")
os.system("make E_Unroll16H")



