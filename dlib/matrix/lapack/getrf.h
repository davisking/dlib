#ifndef DLIB_LAPACk_GETRF_H__
#define DLIB_LAPACk_GETRF_H__

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
                void DLIB_FORTRAN_ID(dgetrf) (integer* m, integer *n, double *a, 
                                             integer* lda, integer *ipiv, integer *info);

                void DLIB_FORTRAN_ID(sgetrf) (integer* m, integer *n, float *a, 
                                             integer* lda, integer *ipiv, integer *info);

            }

            inline int getrf (integer m, integer n, double *a, 
                              integer lda, integer *ipiv)
            {
                integer info = 0;
                DLIB_FORTRAN_ID(dgetrf)(&m, &n, a, &lda, ipiv, &info);
                return info;
            }

            inline int getrf (integer m, integer n, float *a, 
                              integer lda, integer *ipiv)
            {
                integer info = 0;
                DLIB_FORTRAN_ID(sgetrf)(&m, &n, a, &lda, ipiv, &info);
                return info;
            }


        }

    // ------------------------------------------------------------------------------------


/*  -- LAPACK routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGETRF computes an LU factorization of a general M-by-N matrix A */
/*  using partial pivoting with row interchanges. */

/*  The factorization has the form */
/*     A = P * L * U */
/*  where P is a permutation matrix, L is lower triangular with unit */
/*  diagonal elements (lower trapezoidal if m > n), and U is upper */
/*  triangular (upper trapezoidal if m < n). */

/*  This is the right-looking Level 3 BLAS version of the algorithm. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix to be factored. */
/*          On exit, the factors L and U from the factorization */
/*          A = P*L*U; the unit diagonal elements of L are not stored. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  IPIV    (output) INTEGER array, dimension (min(M,N)) */
/*          The pivot indices; for 1 <= i <= min(M,N), row i of the */
/*          matrix was interchanged with row IPIV(i). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization */
/*                has been completed, but the factor U is exactly */
/*                singular, and division by zero will occur if it is used */
/*                to solve a system of equations. */


    // ------------------------------------------------------------------------------------

        template <
            typename T, 
            long NR1, long NR2,
            long NC1, long NC2, 
            typename MM,
            typename layout
            >
        int getrf (
            matrix<T,NR1,NC1,MM,column_major_layout>& a,
            matrix<long,NR2,NC2,MM,layout>& ipiv 
        )
        {
            const long m = a.nr();
            const long n = a.nc();

            matrix<integer,NR2,NC2,MM,column_major_layout> ipiv_temp(std::min(m,n), 1);

            // compute the actual decomposition 
            int info = binding::getrf(m, n, &a(0,0), a.nr(), &ipiv_temp(0,0));

            // Turn the P vector into a more useful form.  This way we will have the identity
            // a == rowm(L*U, ipiv).  The permutation vector that comes out of LAPACK is somewhat
            // different.
            ipiv = trans(range(0, a.nr()-1));
            for (long i = ipiv_temp.size()-1; i >= 0; --i)
            {
                // -1 because FORTRAN is indexed starting with 1 instead of 0
                std::swap(ipiv(i), ipiv(ipiv_temp(i)-1));
            }

            return info;
        }

    // ------------------------------------------------------------------------------------

    }

}

// ----------------------------------------------------------------------------------------

#endif // DLIB_LAPACk_GETRF_H__

