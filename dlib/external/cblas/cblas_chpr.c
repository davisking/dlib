/*
 * cblas_chpr.c
 * The program is a C interface to chpr.
 * 
 * Keita Teranishi  3/23/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const void *X,
                const int incX, void *A)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else
   #define F77_N N
   #define F77_incX incx
#endif
   int n, i, tincx, incx=incX;
   float *x=(float *)X, *xx=(float *)X, *tx, *st;

 
   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_chpr","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_chpr(F77_UL, &F77_N, &alpha, X, &F77_incX, A);

   }  else if (order == CblasRowMajor)
   {
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_chpr","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      if (N > 0)
      {
         n = N << 1;
         x = malloc(n*sizeof(float));
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
      }
      else x = (float *) X;

      F77_chpr(F77_UL, &F77_N, &alpha, x, &F77_incX, A);

   } else 
   {
      cblas_xerbla(1, "cblas_chpr","Illegal Order setting, %d\n", order);
      return;
   }
   if(X!=x)
     free(x);
   return;
}
