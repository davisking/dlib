/*
 * cblas_cdotu_sub.f
 *
 * The program is a C interface to cdotu.
 * It calls the forteran wrapper before calling cdotu.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_cdotu_sub( const CBLAS_INT_TYPE N, const void *X,
                     const CBLAS_INT_TYPE incX, const void *Y, const CBLAS_INT_TYPE incY,void *dotu)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_cdotu_sub( &F77_N, X, &F77_incX, Y, &F77_incY, dotu);
}
