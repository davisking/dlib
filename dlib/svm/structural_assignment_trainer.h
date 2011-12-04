// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_ASSiGNMENT_TRAINER_H__
#define DLIB_STRUCTURAL_ASSiGNMENT_TRAINER_H__

#include "structural_assignment_trainer_abstract.h"
#include "../algs.h"
#include "../optimization.h"
#include "structural_svm_assignment_problem.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class structural_assignment_trainer
    {
    public:
        typedef typename feature_extractor::lhs_element lhs_element;
        typedef typename feature_extractor::rhs_element rhs_element;

        typedef std::pair<std::vector<lhs_element>, std::vector<rhs_element> > sample_type;

        typedef std::vector<long> label_type;

        typedef assignment_function<feature_extractor> trained_function_type;

        bool forces_assignment(
        ) const { return false; } // TODO

        const assignment_function<feature_extractor> train (  
            const std::vector<sample_type>& x,
            const std::vector<label_type>& y
        ) const
        /*!
            requires
                - is_assignment_problem(x,y) == true
                - if (force assignment) then
                    - is_forced_assignment_problem(x,y) == true
        !*/
        {
            DLIB_CASSERT(is_assignment_problem(x,y), "");

            feature_extractor fe;

            bool force_assignment = false;
            unsigned long num_threads = 1;
            structural_svm_assignment_problem<feature_extractor> prob(x,y, fe, force_assignment, num_threads);

            prob.be_verbose();
            prob.set_c(50);
            prob.set_epsilon(1e-10);
            oca solver;

            matrix<double,0,1> weights; 

            solver(prob, weights);
            std::cout << "weights: "<<  trans(weights) << std::endl;

            return assignment_function<feature_extractor>(fe,weights,force_assignment);

        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_ASSiGNMENT_TRAINER_H__




