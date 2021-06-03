/*
 *
 * cblas_cgemm.c
 * This program is a C interface to cgemm.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE K, const void *alpha, const void  *A,
                 const CBLAS_INT_TYPE lda, const void  *B, const CBLAS_INT_TYPE ldb,
                 const void *beta, void  *C, const CBLAS_INT_TYPE ldc)
{
   char TA, TB;   
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_TB;
#else
   #define F77_TA &TA  
   #define F77_TB &TB  
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_M M
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif


   if( Order == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else 
      {
         cblas_xerbla(3, "cblas_cgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      F77_cgemm(F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, alpha, A,
                     &F77_lda, B, &F77_ldb, beta, C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      F77_cgemm(F77_TA, F77_TB, &F77_N, &F77_M, &F77_K, alpha, B,
                  &F77_ldb, A, &F77_lda, beta, C, &F77_ldc);
   } 
   else cblas_xerbla(1, "cblas_cgemm", "Illegal Order setting, %d\n", Order);
   return;
}
