#include "cblas.h"
#include "cblas_f77.h"
void cblas_drotm( const CBLAS_INT_TYPE N, double *X, const CBLAS_INT_TYPE incX, double *Y, 
                       const CBLAS_INT_TYPE incY, const double *P)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_drotm( &F77_N, X, &F77_incX, Y, &F77_incY, P);
}   
