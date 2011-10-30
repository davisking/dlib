// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_ABSTRACT_H__
#ifdef DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_ABSTRACT_H__


#include "structural_svm_sequence_labeling_problem_abstract.h"
#include "../matrix.h"
#include "sequence_labeler.h"
#include <vector>
#include "structural_svm_problem_threaded.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{

    template <
        typename feature_extractor
        >
    class structural_svm_sequence_labeling_problem : 
        public structural_svm_problem_threaded<matrix<double,0,1>, 
                                               std::vector<std::pair<unsigned long,double> > >
    {
    public:
        typedef matrix<double,0,1> matrix_type;
        typedef std::vector<std::pair<unsigned long, double> > feature_vector_type;

        typedef typename feature_extractor::sample_type sample_type;

        structural_svm_sequence_labeling_problem(
            const std::vector<std::vector<sample_type> >& samples_,
            const std::vector<std::vector<unsigned long> >& labels_,
            const feature_extractor& fe_        
        ) :
            structural_svm_problem_threaded<matrix_type,feature_vector_type>(4),
            samples(samples_),
            labels(labels_),
            fe(fe_)
        {
        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_ABSTRACT_H__


