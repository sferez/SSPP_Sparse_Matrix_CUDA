# /***********************************************************************
# *  This code is part of the Small Scale Parallel Programming Assignment
# *
# * SS Assignment
# * Author:  Simeon FEREZ S392371
# * Date:    February-2023
# ***********************************************************************/

import os
import sys

for direct in os.listdir("./input"):
    if direct == ".DS_Store" or direct=="test.mtx":
        continue
    for filename in os.listdir("./input/" + direct):
        if filename.endswith(".mtx"):
            try :
                print("Running " + filename,file=sys.stdout,flush=True)
                os.system("./src/OMP/C_Serial input/" + direct + "/" + filename) # CSR
                os.system("./src/OMP/C_Unroll2V input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll4V input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll8V input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll2H input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll4H input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll8H input/" + direct + "/" + filename)
                os.system("./src/OMP/C_Unroll16H input/" + direct + "/" + filename)
                print("",file=sys.stdout,flush=True)
                os.system("./src/OMP/E_Serial input/" + direct + "/" + filename) # ELLPACK
                os.system("./src/OMP/E_Unroll2V input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll4V input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll8V input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll2H input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll4H input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll8H input/" + direct + "/" + filename)
                os.system("./src/OMP/E_Unroll16H input/" + direct + "/" + filename)
                print("\n",file=sys.stdout,flush=True)
            except:
                continue
