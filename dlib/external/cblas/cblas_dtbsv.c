/*
 * cblas_dtbsv.c
 * The program is a C interface to dtbsv.
 *
 * Keita Teranishi  5/20/98
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double  *A, const int lda,
                 double  *X, const int incX)
{
   char TA;
   char UL;
   char DI;
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_UL, F77_DI;
#else
   #define F77_TA &TA
   #define F77_UL &UL
   #define F77_DI &DI   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_lda=lda, F77_K=K, F77_incX=incX;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_incX incX
#endif

   if (order == CblasColMajor)
   {
      if (Uplo == CblasUpper) UL = 'U';
      else if (Uplo == CblasLower) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dtbsv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(3, "cblas_dtbsv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_dtbsv","Illegal Diag setting, %d\n", Diag);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif
      F77_dtbsv( F77_UL, F77_TA, F77_DI, &F77_N, &F77_K, A, &F77_lda, X,
                      &F77_incX);
   }
   else if (order == CblasRowMajor)
   {
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dtbsv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }

      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans) TA = 'N';
      else 
      {
         cblas_xerbla(3, "cblas_dtbsv","Illegal TransA setting, %d\n", TransA);
         return;
      }

      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_dtbsv","Illegal Diag setting, %d\n", Diag);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_dtbsv( F77_UL, F77_TA, F77_DI, &F77_N, &F77_K, A, &F77_lda, X,
                      &F77_incX);
   }
   else cblas_xerbla(1, "cblas_dtbsv", "Illegal Order setting, %d\n", order);
   return;
}
