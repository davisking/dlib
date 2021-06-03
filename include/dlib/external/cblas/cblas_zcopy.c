/*
 * cblas_zcopy.c
 *
 * The program is a C interface to zcopy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zcopy( const CBLAS_INT_TYPE N, const void *X,
                      const CBLAS_INT_TYPE incX, void *Y, const CBLAS_INT_TYPE incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_zcopy( &F77_N, X, &F77_incX, Y, &F77_incY);
}
