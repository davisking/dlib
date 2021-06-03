/*
 * cblas_cgemv.c
 * The program is a C interface of cgemv
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_cgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void  *A, const CBLAS_INT_TYPE lda,
                 const void  *X, const CBLAS_INT_TYPE incX, const void *beta,
                 void  *Y, const CBLAS_INT_TYPE incY)
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
   #define F77_incX incx
   #define F77_incY incY
#endif

   CBLAS_INT_TYPE n=0, i=0, incx=incX;
   const float *xx= (const float *)X;
   float ALPHA[2],BETA[2];
   CBLAS_INT_TYPE tincY, tincx;
   float *x=(float *)X, *y=(float *)Y, *st=0, *tx=0;
   const float *stx = x;


   if (order == CblasColMajor)
   {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(2, "cblas_cgemv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      F77_cgemv(F77_TA, &F77_M, &F77_N, alpha, A, &F77_lda, X, &F77_incX, 
                beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
         
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans)
      {
         ALPHA[0]=    *( (const float *)  alpha    );
         ALPHA[1]= -( *( (const float *)  alpha+1) );
         BETA[0]=     *( (const float *)  beta     );
         BETA[1]= -(  *( (const float *)  beta+1 ) );
         TA = 'N';
         if (M > 0)
         {
            n = M << 1;
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

            F77_incX = 1;

            if(incY > 0)
               tincY = incY; 
            else
               tincY = -incY; 

            y++;

            if (N > 0)
            {
               i = tincY << 1;
               n = i * N ;
               st = y + n;
               do {
                  *y = -(*y);
                  y += i;
               } while(y != st); 
               y -= n;
            }
            stx = x;
         }
         else stx = (const float *)X;
      }
      else 
      {
         cblas_xerbla(2, "cblas_cgemv","Illegal TransA setting, %d\n", TransA);
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      if (TransA == CblasConjTrans)
         F77_cgemv(F77_TA, &F77_N, &F77_M, ALPHA, A, &F77_lda, stx,
                &F77_incX, BETA, Y, &F77_incY);
      else
         F77_cgemv(F77_TA, &F77_N, &F77_M, alpha, A, &F77_lda, x,
                &F77_incX, beta, Y, &F77_incY);

      if (TransA == CblasConjTrans)
      {
         if (x != (const float *)X) free(x);
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
   }
   else cblas_xerbla(1, "cblas_cgemv", "Illegal Order setting, %d\n", order);
   return;
}
