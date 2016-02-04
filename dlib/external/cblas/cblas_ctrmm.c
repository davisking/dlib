/*
 *
 * cblas_ctrmm.c
 * This program is a C interface to ctrmm.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const  enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const void *alpha, const void  *A, const int lda,
                 void  *B, const int ldb)
{
   char UL, TA, SD, DI;   
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_UL, F77_SD, F77_DI;
#else
   #define F77_TA &TA  
   #define F77_UL &UL  
   #define F77_SD &SD
   #define F77_DI &DI
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_ldb=ldb;
#else
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
   #define F77_ldb ldb
#endif


   if( Order == CblasColMajor )
   {
      if( Side == CblasRight ) SD='R';
      else if ( Side == CblasLeft ) SD='L';
      else 
      {
         cblas_xerbla(2, "cblas_ctrmm", "Illegal Side setting, %d\n", Side);
         return;
      }
      if( Uplo == CblasUpper ) UL='U';
      else if ( Uplo == CblasLower ) UL='L';
      else 
      {
         cblas_xerbla(3, "cblas_ctrmm", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      if( TransA == CblasTrans ) TA ='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(4, "cblas_ctrmm", "Illegal Trans setting, %d\n", TransA);
         return;
      }

      if( Diag == CblasUnit ) DI='U';
      else if ( Diag == CblasNonUnit ) DI='N';
      else cblas_xerbla(5, "cblas_ctrmm", 
                       "Illegal Diag setting, %d\n", Diag);

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_SD = C2F_CHAR(&SD);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_ctrmm(F77_SD, F77_UL, F77_TA, F77_DI, &F77_M, &F77_N, alpha, A, &F77_lda, B, &F77_ldb);
   } else if (Order == CblasRowMajor)
   {
      if( Side == CblasRight ) SD='L';
      else if ( Side == CblasLeft ) SD='R';
      else 
      {
         cblas_xerbla(2, "cblas_ctrmm", "Illegal Side setting, %d\n", Side);
         return;
      }

      if( Uplo == CblasUpper ) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else 
      {
         cblas_xerbla(3, "cblas_ctrmm", "Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      if( TransA == CblasTrans ) TA ='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(4, "cblas_ctrmm", "Illegal Trans setting, %d\n", TransA);
         return;
      }

      if( Diag == CblasUnit ) DI='U';
      else if ( Diag == CblasNonUnit ) DI='N';
      else 
      {
         cblas_xerbla(5, "cblas_ctrmm", "Illegal Diag setting, %d\n", Diag);
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_SD = C2F_CHAR(&SD);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_ctrmm(F77_SD, F77_UL, F77_TA, F77_DI, &F77_N, &F77_M, alpha, A, &F77_lda, B, &F77_ldb);
   } 
   else  cblas_xerbla(1, "cblas_ctrmm", "Illegal Order setting, %d\n", Order);
   return;
}
