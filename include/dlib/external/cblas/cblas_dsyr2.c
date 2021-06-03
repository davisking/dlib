/*
 *
 * cblas_dsyr2.c
 * This program is a C interface to dsyr2.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double  alpha, const double  *X,
                const CBLAS_INT_TYPE incX, const double  *Y, const CBLAS_INT_TYPE incY, double  *A,
                const CBLAS_INT_TYPE lda)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY, F77_lda=lda;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
   #define F77_lda  lda
#endif

   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dsyr2","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_dsyr2(F77_UL, &F77_N, &alpha, X, &F77_incX, Y, &F77_incY, A, 
                    &F77_lda);

   }  else if (order == CblasRowMajor) 
   {
      if (Uplo == CblasLower) UL = 'U';
      else if (Uplo == CblasUpper) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dsyr2","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif  
      F77_dsyr2(F77_UL, &F77_N, &alpha, X, &F77_incX, Y, &F77_incY,  A, 
                    &F77_lda); 
   } else cblas_xerbla(1, "cblas_dsyr2", "Illegal Order setting, %d\n", order);
   return;
}
