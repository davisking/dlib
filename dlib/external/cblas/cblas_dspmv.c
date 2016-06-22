/*
 *
 * cblas_dspmv.c
 * This program is a C interface to dspmv.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */
 

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dspmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_UPLO Uplo, const int N,
                 const double alpha, const double  *AP,
                 const double  *X, const int incX, const double beta,
                 double  *Y, const int incY)
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
      if (Uplo == CblasUpper) UL = 'U';
      else if (Uplo == CblasLower) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dspmv","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_dspmv(F77_UL, &F77_N, &alpha, AP, X,  
                     &F77_incX, &beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dspmv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_dspmv(F77_UL, &F77_N, &alpha, 
                     AP, X,&F77_incX, &beta, Y, &F77_incY);
   }
   else cblas_xerbla(1, "cblas_dspmv", "Illegal Order setting, %d\n", order);
   return;
}
