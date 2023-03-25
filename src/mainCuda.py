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
                os.system("./src/CUDA/E_mv1DB_1DG input/" + direct + "/" + filename) # 1D block ELLPACK
                os.system("./src/CUDA/E_mv1DB_1DG_2H input/" + direct + "/" + filename)
                os.system("./src/CUDA/E_mv1DB_1DG_4H input/" + direct + "/" + filename)

                os.system("./src/CUDA/E_mv2DB_1DG input/" + direct + "/" + filename) # 2D block E
                os.system("./src/CUDA/E_mv2DB_1DG_2H input/" + direct + "/" + filename)
                os.system("./src/CUDA/E_mv2DB_1DG_4H input/" + direct + "/" + filename)

                os.system("./src/CUDA/C_mv1DB_1DG input/" + direct + "/" + filename) # 1D block CSR
                os.system("./src/CUDA/C_mv1DB_1DG_2H input/" + direct + "/" + filename)
                os.system("./src/CUDA/C_mv1DB_1DG_4H input/" + direct + "/" + filename)

                os.system("./src/CUDA/C_mv2DB_1DG input/" + direct + "/" + filename) # 2D block CSR
                os.system("./src/CUDA/C_mv2DB_1DG_2H input/" + direct + "/" + filename)
                os.system("./src/CUDA/C_mv2DB_1DG_4H input/" + direct + "/" + filename)

                print("\n",file=sys.stdout,flush=True)
            except:
                continue
