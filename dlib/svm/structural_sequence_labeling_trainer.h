// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_H__
#define DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_H__

#include "structural_sequence_labeling_trainer_abstract.h"
#include "../algs.h"
#include "../optimization.h"
#include "structural_svm_sequence_labeling_problem.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class structural_sequence_labeling_trainer
    {
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
        ) const
        {
            structural_svm_sequence_labeling_problem<feature_extractor> prob(x, y, fe);
            oca solver;
            matrix<double,0,1> weights; 
            prob.be_verbose();
            prob.set_epsilon(0.5);
            prob.set_c(100);
            solver(prob, weights);

            return sequence_labeler<feature_extractor>(fe,weights);
        }

    private:

        feature_extractor fe;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_H__



