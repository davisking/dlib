/*
 * cblas_dscal.c
 *
 * The program is a C interface to dscal.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_dscal( const CBLAS_INT_TYPE N, const double alpha, double *X, 
                       const CBLAS_INT_TYPE incX)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_dscal( &F77_N, &alpha, X, &F77_incX);
}
