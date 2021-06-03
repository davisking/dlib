/*
 * cblas_dznrm2.c
 *
 * The program is a C interface to dznrm2.
 * It calls the fortran wrapper before calling dznrm2.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
double cblas_dznrm2( const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX) 
{
   double nrm2;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_dznrm2_sub( &F77_N, X, &F77_incX, &nrm2);
   return nrm2;
}
