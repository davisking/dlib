/*
 * cblas_chemv.c
 * The program is a C interface to chemv
 * 
 * Keita Teranishi  5/18/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_chemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_UPLO Uplo, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void  *Y, const int incY)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_lda=lda, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_lda lda
   #define F77_incX incx
   #define F77_incY incY
#endif
   int n=0, i=0, incx=incX;
   const float *xx= (float *)X, *alp= (float *)alpha, *bet = (float *)beta;
   float ALPHA[2],BETA[2];
   int tincY, tincx;
   float *x=(float *)X, *y=(float *)Y, *st=0, *tx;


   if (order == CblasColMajor)
   {
      if (Uplo == CblasUpper) UL = 'U';
      else if (Uplo == CblasLower) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_chemv","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_chemv(F77_UL, &F77_N, alpha, A, &F77_lda, X, &F77_incX, 
                beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      ALPHA[0]= *alp;
      ALPHA[1]= -alp[1];
      BETA[0]= *bet;
      BETA[1]= -bet[1];

      if (N > 0)
      {
         n = N << 1;
         x = malloc(n*sizeof(float));
 
         tx = x;
         if( incX > 0 ) {
           i = incX << 1 ;
           tincx = 2;
           st= x+n;
         } else {
           i = incX *(-2);
           tincx = -2;
           st = x-2;
           x +=(n-2);
         }

         do
         {
           *x = *xx;
           x[1] = -xx[1];
           x += tincx ;
           xx += i;
         }
         while (x != st);
         x=tx;


         #ifdef F77_INT
            F77_incX = 1;
         #else
            incx = 1;
         #endif
 
         if(incY > 0)
           tincY = incY;
         else
           tincY = -incY;
         y++;

         i = tincY << 1;
         n = i * N ;
         st = y + n;
         do {
            *y = -(*y);
            y += i;
         } while(y != st);
         y -= n;
      }  else
         x = (float *) X;

          
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_chemv","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_chemv(F77_UL, &F77_N, ALPHA, A, &F77_lda, x, &F77_incX, 
                BETA, Y, &F77_incY);
   }
   else 
   {
      cblas_xerbla(1, "cblas_chemv","Illegal Order setting, %d\n", order);
      return;
   }
   if ( order == CblasRowMajor )
   {
      if ( X != x )
         free(x);
      if (N > 0)
      {
         do
         {
            *y = -(*y);
            y += i;
         }
         while (y != st);
     }
   }
   return;
}
