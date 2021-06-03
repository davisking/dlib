/*
 *
 * cblas_zher2k.c
 * This program is a C interface to zher2k.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                  const void *B, const CBLAS_INT_TYPE ldb, const double beta,
                  void *C, const CBLAS_INT_TYPE ldc)
{
   char UL, TR;   
#ifdef F77_CHAR
   F77_CHAR F77_TR, F77_UL;
#else
   #define F77_TR &TR  
   #define F77_UL &UL  
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif

   double ALPHA[2]; 
   const double *alp=(double *)alpha;


   if( Order == CblasColMajor )
   {

      if( Uplo == CblasUpper) UL='U';
      else if ( Uplo == CblasLower ) UL='L';
      else 
      {
         cblas_xerbla(2, "cblas_zher2k", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      if( Trans == CblasTrans) TR ='T';
      else if ( Trans == CblasConjTrans ) TR='C';
      else if ( Trans == CblasNoTrans )   TR='N';
      else 
      {
         cblas_xerbla(3, "cblas_zher2k", "Illegal Trans setting, %d\n", Trans);
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TR = C2F_CHAR(&TR);
      #endif

      F77_zher2k(F77_UL, F77_TR, &F77_N, &F77_K, alpha, A, &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      
      if( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else 
      {
         cblas_xerbla(2, "cblas_zher2k", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      if( Trans == CblasTrans) TR ='N';
      else if ( Trans == CblasConjTrans ) TR='N';
      else if ( Trans == CblasNoTrans )   TR='C';
      else 
      {
         cblas_xerbla(3, "cblas_zher2k", "Illegal Trans setting, %d\n", Trans);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TR = C2F_CHAR(&TR);
      #endif

      ALPHA[0]= *alp;
      ALPHA[1]= -alp[1];
      F77_zher2k(F77_UL,F77_TR, &F77_N, &F77_K, ALPHA, A, &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
   } else  cblas_xerbla(1, "cblas_zher2k", "Illegal Order setting, %d\n", Order);
   return;
}
