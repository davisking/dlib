// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LAPACk_BDC_Hh_
#define DLIB_LAPACk_BDC_Hh_

#include "fortran_id.h"
#include "../matrix.h"
namespace dlib
{
    namespace lapack
    {
        namespace binding
        {
            extern "C"
            {
                void DLIB_FORTRAN_ID(dpbtrf) (const char *uplo, const integer *n, const integer *kd,
                                              double *ab, const integer *ldab, integer *info);

                void DLIB_FORTRAN_ID(spbtrf) (const char *uplo, const integer *n, const integer *kd,
                                              float *ab, const integer *ldab, integer *info);

            }

            inline integer pbtrf (const char uplo, const integer n, const integer kd,
                                  double* ab, const integer ldab)
            {
                integer info = 0;
                DLIB_FORTRAN_ID(dpbtrf)(&uplo, &n, &kd, ab, &ldab, &info);
                return info;
            }

            inline integer pbtrf (const char uplo, const integer n, const integer kd,
                                  float* ab, const integer ldab)
            {
                integer info = 0;
                DLIB_FORTRAN_ID(spbtrf)(&uplo, &n, &kd, ab, &ldab, &info);
                return info;
            }
        }

    // ------------------------------------------------------------------------------------
/*  DPBTRF(l)		LAPACK routine (version	1.1)		    DPBTRF(l)

NAME
  DPBTRF - compute the Cholesky	factorization of a real	symmetric positive
  definite band	matrix A

SYNOPSIS

  SUBROUTINE DPBTRF( UPLO, N, KD, AB, LDAB, INFO )

      CHARACTER	     UPLO

      INTEGER	     INFO, KD, LDAB, N

      DOUBLE	     PRECISION AB( LDAB, * )

PURPOSE
  DPBTRF computes the Cholesky factorization of	a real symmetric positive
  definite band	matrix A.

  The factorization has	the form
     A = U**T *	U,  if UPLO = 'U', or
     A = L  * L**T,  if	UPLO = 'L',
  where	U is an	upper triangular matrix	and L is lower triangular.

ARGUMENTS

  UPLO	  (input) CHARACTER*1
	  = 'U':  Upper	triangle of A is stored;
	  = 'L':  Lower	triangle of A is stored.

  N	  (input) INTEGER
	  The order of the matrix A.  N	>= 0.

  KD	  (input) INTEGER
	  The number of	superdiagonals of the matrix A if UPLO = 'U', or the
	  number of subdiagonals if UPLO = 'L'.	 KD >= 0.

  AB	  (input/output) DOUBLE	PRECISION array, dimension (LDAB,N)
	  On entry, the	upper or lower triangle	of the symmetric band matrix
	  A, stored in the first KD+1 rows of the array.  The j-th column of
	  A is stored in the j-th column of the	array AB as follows: if	UPLO
	  = 'U', AB(kd+1+i-j,j)	= A(i,j) for max(1,j-kd)<=i<=j;	if UPLO	=
	  'L', AB(1+i-j,j)    =	A(i,j) for j<=i<=min(n,j+kd).

	  On exit, if INFO = 0,	the triangular factor U	or L from the Chole-
	  sky factorization A =	U**T*U or A = L*L**T of	the band matrix	A, in
	  the same storage format as A.

  LDAB	  (input) INTEGER
	  The leading dimension	of the array AB.  LDAB >= KD+1.

  INFO	  (output) INTEGER
	  = 0:	successful exit
	  < 0:	if INFO	= -i, the i-th argument	had an illegal value
	  > 0:	if INFO	= i, the leading minor of order	i is not positive
	  definite, and	the factorization could	not be completed.

FURTHER	DETAILS
  The band storage scheme is illustrated by the	following example, when	N =
  6, KD	= 2, and UPLO =	'U':

  On entry:			  On exit:

      *	   *   a13  a24	 a35  a46      *    *	u13  u24  u35  u46
      *	  a12  a23  a34	 a45  a56      *   u12	u23  u34  u45  u56
     a11  a22  a33  a44	 a55  a66     u11  u22	u33  u44  u55  u66

  Similarly, if	UPLO = 'L' the format of A is as follows:

  On entry:			  On exit:

     a11  a22  a33  a44	 a55  a66     l11  l22	l33  l44  l55  l66
     a21  a32  a43  a54	 a65   *      l21  l32	l43  l54  l65	*
     a31  a42  a53  a64	  *    *      l31  l42	l53  l64   *	*

  Array	elements marked	* are not used by the routine.

  Contributed by
  Peter	Mayes and Giuseppe Radicati, IBM ECSEC,	Rome, March 23,	1989 */

    // ------------------------------------------------------------------------------------

        template <
            typename T, 
            long NR1, long NC1,
            typename MM
            >
        int pbtrf (
            char uplo, matrix<T,NR1,NC1,MM,column_major_layout>& ab
        )
        {
            const long ldab = ab.nr();
            const long n = ab.nc();
            const long kd = ldab - 1; // assume fully packed 

            int info = binding::pbtrf(uplo, n, kd, &ab(0,0), ldab);

            return info;
        }

    // ------------------------------------------------------------------------------------


        template <
            typename T, 
            long NR1, long NC1,
            typename MM
            >
        int pbtrf (
            char uplo, matrix<T,NR1,NC1,MM,row_major_layout>& ab
        )
        {
            const long ldab = ab.nr();
            const long n = ab.nc();
            const long kd = ldab - 1; // assume fully packed 

            matrix<T,NC1,NR1,MM,row_major_layout> abt = trans(ab);

            int info = binding::pbtrf(uplo, n, kd, &abt(0,0), ldab);

            ab = trans(abt);

            return info;
        }

    // ------------------------------------------------------------------------------------

    }

}

// ----------------------------------------------------------------------------------------

#endif // DLIB_LAPACk_BDC_Hh_


