// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_ABSTRACT_H__
#ifdef DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_ABSTRACT_H__

#include "cross_validate_sequence_labeler_abstract.h"
#include <vector>
#include "../matrix.h"
#include "svm.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sequence_labeler_type,
        typename sample_type
        >
    const matrix<double> test_sequence_labeler (
        const sequence_labeler_type& labeler,
        const std::vector<std::vector<sample_type> >& samples,
        const std::vector<std::vector<unsigned long> >& labels
    );
    /*!
        requires
            - is_sequence_labeling_problem(samples, labels)
        ensures
            - Tests labeler against the given samples and labels and returns a confusion 
              matrix summarizing the results.
            - The confusion matrix C returned by this function has the following properties.
                - C.nc() == labeler.num_labels()
                - C.nr() == max(labeler.num_labels(), max value in labels + 1)
                - C(T,P) == the number of times a sample with label T was predicted
                  to have a label of P.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename sample_type
        >
    const matrix<double> cross_validate_sequence_labeler (
        const trainer_type& trainer,
        const std::vector<std::vector<sample_type> >& samples,
        const std::vector<std::vector<unsigned long> >& labels,
        const long folds
    );
    /*!
        requires
            - is_sequence_labeling_problem(samples, labels)
            - 1 < folds <= samples.size()
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_ABSTRACT_H__



