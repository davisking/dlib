/*
 * cblas_saxpy.c
 *
 * The program is a C interface to saxpy.
 * It calls the fortran wrapper before calling saxpy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_saxpy( const int N, const float alpha, const float *X,
                       const int incX, float *Y, const int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_saxpy( &F77_N, &alpha, X, &F77_incX, Y, &F77_incY);
} 
