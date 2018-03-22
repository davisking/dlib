/*
 * cblas_zhpmv.c
 * The program is a C interface of zhpmv
 * 
 * Keita Teranishi  5/18/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zhpmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_UPLO Uplo,const CBLAS_INT_TYPE N,
                 const void *alpha, const void  *AP,
                 const void  *X, const CBLAS_INT_TYPE incX, const void *beta,
                 void  *Y, const CBLAS_INT_TYPE incY)
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
   #define F77_incX incx
   #define F77_incY incY
#endif
   CBLAS_INT_TYPE n, i=0, incx=incX;
   const double *xx= (double *)X, *alp= (double *)alpha, *bet = (double *)beta;
   double ALPHA[2],BETA[2];
   CBLAS_INT_TYPE tincY, tincx;
   double *x=(double *)X, *y=(double *)Y, *st=0, *tx;

   if (order == CblasColMajor)
   { 
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_zhpmv","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_zhpmv(F77_UL, &F77_N, alpha, AP, X,  
                     &F77_incX, beta, Y, &F77_incY);
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
         x = malloc(n*sizeof(double));
 
         tx = x;
         if( incX > 0 ) {
           i = incX << 1;
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
         x = (double *) X;


      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_zhpmv","Illegal Uplo setting, %d\n", Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_zhpmv(F77_UL, &F77_N, ALPHA, 
                     AP, x, &F77_incX, BETA, Y, &F77_incY);
   }
   else 
   {
      cblas_xerbla(1, "cblas_zhpmv","Illegal Order setting, %d\n", order);
      return;
   }
   if ( order == CblasRowMajor ) 
   {
      if(X!=x)
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
