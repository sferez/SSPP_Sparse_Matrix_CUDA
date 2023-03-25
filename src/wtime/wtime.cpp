/***********************************************************************
 *  This code is part of the Small Scale Parallel Programming Assignment
 *
 * SS Assignment
 * Author:  Simeon FEREZ S392371
 * Date:    February-2023
 ***********************************************************************/

#include "wtime.h"

double wtime()
{
#if 1
    struct timeval tt;
    struct timezone tz;
    double temp;
    if (gettimeofday(&tt,&tz) != 0) {
        //fprintf(stderr,"Fatal error for gettimeofday ??? \n");
        exit(-1);
    }
    temp = ((double)tt.tv_sec) + ((double)tt.tv_usec)*1.0e-6;
    return(temp);
#else
    clock_t t=clock();
  return(   ((double) t/CLOCKS_PER_SEC) );
#endif
}
