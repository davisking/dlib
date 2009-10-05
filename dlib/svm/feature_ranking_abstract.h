// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_
#ifdef DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_

#include <vector>
#include <limits>

#include "svm_abstract.h"
#include "kcentroid_abstract.h"
#include "../is_kind.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename sample_matrix_type,
        typename label_matrix_type
        >
    matrix<typename kernel_type::scalar_type> rank_features (
        const kcentroid<kernel_type>& kc,
        const sample_matrix_type& samples,
        const label_matrix_type& labels,
        const long num_features = samples(0).nr()
    );
    /*!
        requires
            - sample_matrix_type == a matrix or something convertible to a matrix via vector_to_matrix()
            - label_matrix_type  == a matrix or something convertible to a matrix via vector_to_matrix()
            - is_binary_classification_problem(samples, labels) == true
            - kc.train(samples(0)) must be a valid expression.  This means that
              kc must use a kernel type that is capable of operating on the
              contents of the samples matrix
            - 0 < num_features <= samples(0).nr()
        ensures
            - Let Class1 denote the centroid of all the samples with labels that are < 0
            - Let Class2 denote the centroid of all the samples with labels that are > 0
            - finds a ranking of the features where the best features come first.  This 
              function does this by computing the distance between the centroid of the Class1 
              samples and the Class2 samples in kernel defined feature space.
              Good features are then ones that result in the biggest separation between
              the two centroids of Class1 and Class2.
            - Uses the kc object to compute the centroids of the two classes
            - returns a ranking matrix R where:
                - R.nr() == num_features
                - r.nc() == 2
                - R(i,0) == the index of the ith best feature according to our ranking.
                  (e.g. samples(n)(R(0,0)) is the best feature from sample(n) and
                   samples(n)(R(1,0)) is the second best, samples(n)(R(2,0)) the
                   third best and so on)
                - R(i,1) == a number that indicates how much separation exists between 
                  the two centroids when features 0 through i are used.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename sample_matrix_type,
        typename label_matrix_type
        >
    matrix<double,0,2> rank_features_rbf (
        const sample_matrix_type& samples,
        const label_matrix_type& labels,
        unsigned long num_sv = 40
    );
    /*!
        requires
            - num_sv > 0
            - is_binary_classification_problem(samples, labels) == true
        ensures
            - This function just calls the above rank_features() function but uses the 
              radial_basis_kernel and automatically picks a gamma parameter for you.  
              It also sets the kcentroid up to use num_sv dictionary vectors.  Finally, it 
              tells rank_features() to rank all the features.
            - The return value from this function is the matrix returned by rank_features()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename sample_matrix_type,
        typename label_matrix_type
        >
    matrix<double,0,2> verbose_rank_features_rbf (
        const sample_matrix_type& samples,
        const label_matrix_type& labels,
        unsigned long num_sv = 40
    );
    /*!
        requires
            - num_sv > 0
            - is_binary_classification_problem(samples, labels) == true
        ensures
            - This function just calls the above rank_features() function but uses the 
              radial_basis_kernel and automatically picks a gamma parameter for you.  
              It also sets the kcentroid up to use num_sv dictionary vectors.  Finally, it 
              tells rank_features() to rank all the features.
            - The return value from this function is the matrix returned by rank_features()
            - This function is verbose in the sense that it will print status messages to
              standard out during its processing.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_



