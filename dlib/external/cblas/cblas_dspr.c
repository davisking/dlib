/*
 *
 * cblas_dspr.c
 * This program is a C interface to dspr.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const double *X,
                const CBLAS_INT_TYPE incX, double *Ap)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else
   #define F77_N N
   #define F77_incX incX
#endif
   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dspr","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_dspr(F77_UL, &F77_N, &alpha, X, &F77_incX, Ap);

   }  else if (order == CblasRowMajor) 
   {
      if (Uplo == CblasLower) UL = 'U';
      else if (Uplo == CblasUpper) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dspr","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif  
      F77_dspr(F77_UL, &F77_N, &alpha, X, &F77_incX, Ap); 
   } else cblas_xerbla(1, "cblas_dspr", "Illegal Order setting, %d\n", order);
   return;
}
