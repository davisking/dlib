// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_
#ifdef DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_

#include <vector>
#include <limits>

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
        const long num_features
    );
    /*!
        requires
            - is_matrix<sample_matrix_type>::value == true
              (i.e. samples must be a dlib matrix object)
            - is_matrix<sample_matrix_type::type>::value == true
              (i.e. samples must also contain dlib matrix objects)
            - is_matrix<label_matrix_type>::value == true
              (i.e. labels must be a dlib matrix object)
            - samples.nc() == 1 && labels.nc() == 1
              (i.e. samples and labels must be column vectors)
            - samples.size() == labels.size()
            - samples.size() > 0
            - for all i < samples.size()
                - 0 < num_features <= samples(i).nr()
                - samples(i).nc() = 1
                - i.e. samples must contain column vectors of equal length
                  and num_features must be less than the size of these column vectors
            - kc.train(samples(0)) must be a valid expression.  This means that
              kc must use a kernel type that is capable of operating on the
              contents of the samples matrix
        ensures
            - Let Class1 denote the centroid of all the samples with labels that are < 0
            - Let Class2 denote the centroid of all the samples with labels that are > 0
            - finds a ranking of the top num_features best features.  This function 
              does this by computing the distance between the centroid of the Class1 
              samples and the Class2 samples in kernel defined feature space.
              Good features are then ones that result in the biggest separation between
              the two centroids of Class1 and Class2
            - Uses the kc object to compute the centroids of the two classes
            - returns a ranking matrix R where:
                - R.nr() == num_features
                - r.nc() == 2
                - R(i,0) == the index of the ith best feature according to our ranking.
                  (e.g. samples(n)(R(0,0)) is the best feature from sample(n) and
                   samples(n)(R(1,0)) is the second best, samples(n)(R(2,0)) the
                   third best and so on)
                - R(i,1) == a number that indicates how much the feature R(i,0) contributes
                  to the separation of the Class1 and Class2 centroids when it 
                  is added into the feature set defined by R(0,0), R(1,0), R(2,0), up to 
                  R(i-1,0).  
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_KERNEL_FEATURE_RANKINg_ABSTRACT_H_



