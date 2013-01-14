// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CCA_h__
#define DLIB_CCA_h__

#include "cca_abstract.h"
#include "../algs.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    matrix<T,0,1> compute_correlations (
        const matrix<T>& L,
        const matrix<T>& R
    )
    {
        matrix<T> A, B, C;
        A = diag(trans(R)*L);
        B = sqrt(diag(trans(L)*L));
        C = sqrt(diag(trans(R)*R));
        A = pointwise_multiply(A , reciprocal(pointwise_multiply(B,C)));
        return A;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type, 
        typename T
        >
    matrix<T,0,1> impl_cca (
        const matrix_type& L,
        const matrix_type& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank,
        unsigned long q,
        unsigned long num_output_correlations
    )
    {
        matrix<T> Ul, Vl;
        matrix<T> Ur, Vr;
        matrix<T> U, V;
        matrix<T,0,1> Dr, Dl, D;


        // Note that we add a few more singular vectors in because it helps improve the
        // final results if we run this part with a little higher rank than the final SVD.
        svd_fast(L, Ul, Dl, Vl, num_correlations+extra_rank, q);
        svd_fast(R, Ur, Dr, Vr, num_correlations+extra_rank, q);

        // This matrix is really small so we can do a normal full SVD on it.
        svd3(trans(Ul)*Ur, U, D, V);
        // now throw away extra columns of the transformations.  We do this in a way
        // that keeps the directions that have the highest correlations.
        matrix<T,0,1> temp = D;
        rsort_columns(U, temp);
        rsort_columns(V, D);
        U = colm(U, range(0, num_output_correlations-1));
        V = colm(V, range(0, num_output_correlations-1));
        D = rowm(D, range(0, num_output_correlations-1));

        Ltrans = Vl*inv(diagm(Dl))*U;
        Rtrans = Vr*inv(diagm(Dr))*V;

        // Note that the D matrix contains the correlation values for the transformed
        // vectors.  However, when the L and R matrices have rank higher than
        // num_correlations+extra_rank then the values in D become only approximate.
        return D; 
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    matrix<T,0,1> cca (
        const matrix<T>& L,
        const matrix<T>& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank = 5,
        unsigned long q = 2
    )
    {
        using std::min;
        const unsigned long n = min(num_correlations, (unsigned long)min(R.nr(),min(L.nc(), R.nc())));
        return impl_cca(L,R,Ltrans, Rtrans, num_correlations, extra_rank, q, n); 
    }

// ----------------------------------------------------------------------------------------

    template <typename sparse_vector_type, typename T>
    matrix<T,0,1> cca (
        const std::vector<sparse_vector_type>& L,
        const std::vector<sparse_vector_type>& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank = 5,
        unsigned long q = 2
    )
    {
        using std::min;
        const unsigned long n = min(max_index_plus_one(L), max_index_plus_one(R));
        const unsigned long num_output_correlations = min(num_correlations, min(R.size(),n));
        return impl_cca(L,R,Ltrans, Rtrans, num_correlations, extra_rank, q, num_output_correlations); 
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CCA_h__


