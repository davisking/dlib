/*
 * cblas_chpr2.c
 * The program is a C interface to chpr2.
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                      const CBLAS_INT_TYPE N,const void *alpha, const void *X, 
                      const CBLAS_INT_TYPE incX,const void *Y, const CBLAS_INT_TYPE incY, void *Ap)

{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N,  F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incx
   #define F77_incY incy
#endif
   CBLAS_INT_TYPE n, i, j, tincx, tincy, incx=incX, incy=incY;
   float *x=(float *)X, *xx=(float *)X, *y=(float *)Y,
         *yy=(float *)Y, *tx, *ty, *stx, *sty;

 
   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_chpr2","Illegal Uplo setting, %d\n",Uplo );
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_chpr2(F77_UL, &F77_N, alpha, X, &F77_incX, Y, &F77_incY, Ap);

   }  else if (order == CblasRowMajor)
   {
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_chpr2","Illegal Uplo setting, %d\n", Uplo);
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      if (N > 0)
      {
         n = N << 1;
         x = malloc(n*sizeof(float));
         y = malloc(n*sizeof(float));
         tx = x;
         ty = y;
         if( incX > 0 ) {
            i = incX << 1 ;
            tincx = 2;
            stx= x+n;
         } else {
            i = incX *(-2);
            tincx = -2;
            stx = x-2;
            x +=(n-2);
         }
 
         if( incY > 0 ) {
            j = incY << 1;
            tincy = 2;
            sty= y+n;
         } else {
            j = incY *(-2);
            tincy = -2;
            sty = y-2;
            y +=(n-2);
         }
 
         do
         {
            *x = *xx;
            x[1] = -xx[1];
            x += tincx ;
            xx += i;
         }
         while (x != stx);
         do
         {
            *y = *yy;
            y[1] = -yy[1];
            y += tincy ;
            yy += j;
         }
         while (y != sty);
 
         x=tx;
         y=ty;
 
         #ifdef F77_INT
            F77_incX = 1;
            F77_incY = 1;
         #else
            incx = 1;
            incy = 1;
         #endif

      }  else 
      {
         x = (float *) X;
         y = (void  *) Y;
      }
      F77_chpr2(F77_UL, &F77_N, alpha, y, &F77_incY, x, &F77_incX, Ap);
   } else 
   {
      cblas_xerbla(1, "cblas_chpr2","Illegal Order setting, %d\n", order);
      return;
   }
   if(X!=x)
      free(x);
   if(Y!=y)
      free(y);
   return;
}
