/*
 *
 * cblas_chemm.c
 * This program is a C interface to chemm.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_chemm(const enum CBLAS_ORDER Order, const  enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                 void *C, const CBLAS_INT_TYPE ldc)
{
   char SD, UL;   
#ifdef F77_CHAR
   F77_CHAR F77_SD, F77_UL;
#else
   #define F77_SD &SD  
   #define F77_UL &UL  
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif


   if( Order == CblasColMajor )
   {
      if( Side == CblasRight) SD='R';
      else if ( Side == CblasLeft ) SD='L';
      else 
      {
         cblas_xerbla(2, "cblas_chemm", "Illegal Side setting, %d\n", Side);
         return;
      }

      if( Uplo == CblasUpper) UL='U';
      else if ( Uplo == CblasLower ) UL='L';
      else 
      {
         cblas_xerbla(3, "cblas_chemm", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_SD = C2F_CHAR(&SD);
      #endif

      F77_chemm(F77_SD, F77_UL, &F77_M, &F77_N, alpha, A, &F77_lda, 
                     B, &F77_ldb, beta, C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      if( Side == CblasRight) SD='L';
      else if ( Side == CblasLeft ) SD='R';
      else 
      {
         cblas_xerbla(2, "cblas_chemm", "Illegal Side setting, %d\n", Side);
         return;
      }

      if( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else 
      {
         cblas_xerbla(3, "cblas_chemm", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_SD = C2F_CHAR(&SD);
      #endif

      F77_chemm(F77_SD, F77_UL, &F77_N, &F77_M, alpha, A,
                 &F77_lda, B, &F77_ldb, beta, C, &F77_ldc);
   } 
   else  cblas_xerbla(1, "cblas_chemm", "Illegal Order setting, %d\n", Order);
   return;
}
