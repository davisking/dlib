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

        const feature_extractor& get_feature_extractor (
        ) const { return fe; }
        /*!
            ensures
                - returns the feature extractor used by this object
        !*/

        unsigned long num_labels (
        ) const { return fe.num_labels(); }
        /*!
            ensures
                - returns get_feature_extractor().num_labels()
                  (i.e. returns the number of possible output labels for each 
                  element of a sequence)
        !*/

        const sequence_labeler<feature_extractor> train(
            const std::vector<sample_sequence_type>& x,
            const std::vector<labeled_sequence_type>& y
        ) const;
        /*!
            requires
                - is_sequence_labeling_problem(x, y)
                - for all valid i and j: y[i][j] < num_labels()
            ensures
                - Uses the structural_svm_sequence_labeling_problem to train a 
                  sequence_labeler on the given x/y training pairs.  The idea is 
                  to learn to predict a y given an input x.
                - returns a function F with the following properties:
                    - F(new_x) == A sequence of predicted labels for the elements of new_x.  
                    - F(new_x).size() == new_x.size()
                    - for all valid i:
                        - F(new_x)[i] == the predicted label of new_x[i]
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SEQUENCE_LABELING_TRAiNER_ABSTRACT_H__




