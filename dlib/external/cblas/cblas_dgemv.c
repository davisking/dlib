/*
 *
 * cblas_dgemv.c
 * This program is a C interface to dgemv.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double  *A, const int lda,
                 const double  *X, const int incX, const double beta,
                 double  *Y, const int incY)
{
   char TA;
#ifdef F77_CHAR
   F77_CHAR F77_TA;
#else
   #define F77_TA &TA   
#endif
#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_incX=incX, F77_incY=incY;
#else
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
   #define F77_incX incX
   #define F77_incY incY
#endif

   if (order == CblasColMajor)
   {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(2, "cblas_dgemv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      F77_dgemv(F77_TA, &F77_M, &F77_N, &alpha, A, &F77_lda, X, &F77_incX, 
                &beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans) TA = 'N';
      else 
      {
         cblas_xerbla(2, "cblas_dgemv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      F77_dgemv(F77_TA, &F77_N, &F77_M, &alpha, A, &F77_lda, X,
                &F77_incX, &beta, Y, &F77_incY);
   }
   else cblas_xerbla(1, "cblas_dgemv", "Illegal Order setting, %d\n", order);
   return;
}
