// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CCA_AbSTRACT_H__
#ifdef DLIB_CCA_AbSTRACT_H__

#include "../matrix/matrix_la_abstract.h"
#include "random_subset_selector_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    matrix<T,0,1> compute_correlations (
        const matrix<T>& L,
        const matrix<T>& R
    );
    /*!
        requires
            - L.size() > 0 
            - R.size() > 0 
            - L.nr() == R.nr()
        ensures
            - This function treats L and R as sequences of paired row vectors.  It
              then computes the correlation values between the elements of these 
              row vectors.  In particular, we return a vector COR such that:
                - COR.size() == L.nc()
                - for all valid i:
                    - COR(i) == the correlation coefficient between the following 
                      sequence of paired numbers: (L(k,i), R(k,i)) for k: 0 <= k < L.nr().
                      Therefore, COR(i) is a value between -1 and 1 inclusive where 1
                      indicates perfect correlation and -1 perfect anti-correlation.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    matrix<T,0,1> cca (
        const matrix<T>& L,
        const matrix<T>& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank = 5,
        unsigned long q = 2
    );
    /*!
        requires
            - num_correlations > 0
            - L.size() > 0 
            - R.size() > 0 
            - L.nr() == R.nr()
        ensures
            - This function performs a canonical correlation analysis between the row
              vectors in L and R.  That is, it finds two transformation matrices, Ltrans
              and Rtrans, such that row vectors in the transformed matrices L*Ltrans and
              R*Rtrans are as correlated as possible.  That is, we try to find two transforms
              such that the correlation values returned by compute_correlations(L*Ltrans, R*Rtrans)
              would be maximized.
            - Let N == min(num_correlations, min(R.nr(),min(L.nc(),R.nc())))
              (This is the actual number of elements in the transformed vectors.
              Therefore, note that you can't get more outputs than there are rows or
              columns in the input matrices.)
            - #Ltrans.nr() == L.nc()
            - #Ltrans.nc() == N 
            - #Rtrans.nr() == R.nc()
            - #Rtrans.nc() == N 
            - No centering is applied to the L and R matrices.  Therefore, if you want a
              CCA relative to the centered vectors then you must apply centering yourself
              before calling cca().
            - This function works with reduced rank approximations of the L and R matrices.
              This makes it fast when working with large matrices.  In particular, we use
              the svd_fast() routine to find reduced rank representations of the input
              matrices by calling it as follows: svd_fast(L, U,D,V, num_correlations+extra_rank, q) 
              and similarly for R.  This means that you can use the extra_rank and q
              arguments to cca() to influence the accuracy of the reduced rank
              approximation.  However, the default values should work fine for most
              problems.
            - returns an estimate of compute_correlations(L*#Ltrans, R*#Rtrans).  The
              returned vector should exactly match the output of compute_correlations()
              when the reduced rank approximation to L and R is accurate.  However, when L
              and/or R are higher rank than num_correlations+extra_rank the return value of
              this function will deviate from compute_correlations(L*#Ltrans, R*#Rtrans).
              This deviation can be used to check if the reduced rank approximation is
              working or you need to increase extra_rank.
            - A good discussion of CCA can be found in the paper "Canonical Correlation
              Analysis" by David Weenink.  In particular, this function is implemented
              using equations 29 and 30 from his paper.  We also use the idea of doing CCA
              on a reduced rank approximation of L and R as suggested by Paramveer S.
              Dhillon in his paper "Two Step CCA: A new spectral method for estimating
              vector models of words".
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename sparse_vector_type, 
        typename T
        >
    matrix<T,0,1> cca (
        const std::vector<sparse_vector_type>& L,
        const std::vector<sparse_vector_type>& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank = 5,
        unsigned long q = 2
    );
    /*!
        requires
            - num_correlations > 0
            - L.size() == R.size()
            - max_index_plus_one(L) > 0 && max_index_plus_one(R) > 0
              (i.e. L and R can't represent empty matrices)
            - L and R must contain sparse vectors (see the top of dlib/svm/sparse_vector_abstract.h
              for a definition of sparse vector)
        ensures
            - This is just an overload of the cca() function defined above.  Except in this
              case we take a sparse representation of the input L and R matrices rather than
              dense matrices.  Therefore, in this case, we interpret L and R as matrices
              with L.size() rows, where each row is defined by a sparse vector.  So this 
              function does exactly the same thing as the above cca().
            - Note that you can apply the output transforms to a sparse vector with the
              following code:
                sparse_matrix_vector_multiply(trans(Ltrans), your_sparse_vector)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename sparse_vector_type, 
        typename Rand_type, 
        typename T
        >
    matrix<T,0,1> cca (
        const random_subset_selector<sparse_vector_type,Rand_type>& L,
        const random_subset_selector<sparse_vector_type,Rand_type>& R,
        matrix<T>& Ltrans,
        matrix<T>& Rtrans,
        unsigned long num_correlations,
        unsigned long extra_rank = 5,
        unsigned long q = 2
    );
    /*!
        requires
            - num_correlations > 0
            - L.size() == R.size()
            - max_index_plus_one(L) > 0 && max_index_plus_one(R) > 0
              (i.e. L and R can't represent empty matrices)
            - L and R must contain sparse vectors (see the top of dlib/svm/sparse_vector_abstract.h
              for a definition of sparse vector)
        ensures
            - returns cca(L.to_std_vector(), R.to_std_vector(), Ltrans, Rtrans, num_correlations, extra_rank, q)
              (i.e. this is just a convenience function for calling the cca() routine when
              your sparse vectors are contained inside a random_subset_selector rather than
              a std::vector)
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CCA_AbSTRACT_H__


