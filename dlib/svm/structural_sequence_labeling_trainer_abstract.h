// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_ABSTRACT_H__
#ifdef DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_ABSTRACT_H__

#include "../algs.h"
#include "../optimization.h"
#include "structural_svm_sequence_labeling_problem_abstract.h"
#include "sequence_labeler_abstract.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class structural_sequence_labeling_trainer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/
    public:
        typedef typename feature_extractor::sample_type sample_type;
        typedef std::vector<sample_type> sample_sequence_type;
        typedef std::vector<unsigned long> labeled_sequence_type;

        typedef sequence_labeler<feature_extractor> trained_function_type;

        explicit structural_sequence_labeling_trainer (
            const feature_extractor& fe_
        ) : fe(fe_)
        {}

        structural_sequence_labeling_trainer (
        ) {}

        const sequence_labeler<feature_extractor> train(
            const std::vector<sample_sequence_type>& x,
            const std::vector<labeled_sequence_type>& y
        ) const;

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_ABSTRACT_H__




