/*
 *
 * cblas_sspr2.c
 * This program is a C interface to sspr2.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float  alpha, const float  *X,
                const CBLAS_INT_TYPE incX, const float  *Y, const CBLAS_INT_TYPE incY, float  *A)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif

   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_sspr2","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_sspr2(F77_UL, &F77_N, &alpha, X, &F77_incX, Y, &F77_incY, A);

   }  else if (order == CblasRowMajor) 
   {
      if (Uplo == CblasLower) UL = 'U';
      else if (Uplo == CblasUpper) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_sspr2","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif  
      F77_sspr2(F77_UL, &F77_N, &alpha, X, &F77_incX, Y, &F77_incY,  A); 
   } else cblas_xerbla(1, "cblas_sspr2", "Illegal Order setting, %d\n", order);
}
