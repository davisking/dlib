/*
 * cblas_dsdot.c
 *
 * The program is a C interface to dsdot.
 * It calls fthe fortran wrapper before calling dsdot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
double  cblas_dsdot( const int N, const float *X,
                      const int incX, const float *Y, const int incY)
{
   double dot;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_dsdot_sub( &F77_N, X, &F77_incX, Y, &F77_incY, &dot);
   return dot;
}   
