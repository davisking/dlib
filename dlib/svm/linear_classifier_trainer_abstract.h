// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LINEAR_CLASSIFIER_TRAINER_ABSTRACT_Hh_
#ifdef DLIB_LINEAR_CLASSIFIER_TRAINER_ABSTRACT_Hh_

#include "svm_c_linear_dcd_trainer_abstract.h"
#include "function_abstract.h"
#include "../matrix.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type_
        >
    class linear_classifier_trainer
    {
        /*!
            REQUIREMENTS ON sample_type_
                sample_type_ is a dlib::matrix column vector, or a sparse vector as
                defined in dlib/svm/sparse_vector_abstract.h.

            INITIAL VALUE
                - get_solver() == solver_type()
                - get_c_candidates() == {1}
                - get_num_folds() == 5
                - This object is not verbose.

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for training a binary linear classifier that
                outputs the probability that a sample belongs to the +1 class.  It trains
                a linear SVM with svm_c_linear_dcd_trainer, and:
                    - Gives each class a C inversely proportional to its number of
                      samples, so both classes weigh the same in the SVM's objective,
                      however unbalanced they are.
                    - Chooses C among a list of candidates by cross-validation.
                    - Turns the SVM's output into a probability by fitting a sigmoid to
                      the scores of the cross-validation folds, i.e. Platt scaling.

                It splits the samples into cross-validation folds itself, with the same
                share of each class in every fold.  The samples are shuffled with a fixed
                seed before being dealt to the folds, so the folds do not follow the order
                of the samples, and the same inputs always give the same result.

                The trained function outputs a probability, so a sample is predicted to
                be +1 when its output is > 0.5.  Tools that judge a classifier by the
                sign of its output, such as cross_validate_trainer() and
                test_binary_decision_function(), give meaningless results with it.
        !*/

    public:
        using sample_type = sample_type_;
        using kernel_type = linear_kernel<sample_type> if sample_type is a dlib::matrix,
                            sparse_linear_kernel<sample_type> otherwise;
        using scalar_type = typename kernel_type::scalar_type;
        using mem_manager_type = typename kernel_type::mem_manager_type;
        using trained_function_type = probabilistic_decision_function<kernel_type>;
        using solver_type = svm_c_linear_dcd_trainer<kernel_type>;

        linear_classifier_trainer (
        );
        /*!
            ensures
                - This object has its initial value.
        !*/

        void set_solver (
            const solver_type& solver
        );
        /*!
            ensures
                - #get_solver() == solver
        !*/

        const solver_type& get_solver (
        ) const;
        /*!
            ensures
                - Returns the solver train() uses.  Its settings, such as its loss,
                  epsilon or whether it includes a bias, are the ones train() uses, except
                  for its C values, which train() sets for each class.
        !*/

        void set_c (
            scalar_type c
        );
        /*!
            requires
                - c > 0
            ensures
                - #get_c_candidates() == {c}
        !*/

        void set_c_candidates (
            const std::vector<scalar_type>& candidates
        );
        /*!
            requires
                - candidates.size() > 0
                - Every element of candidates is > 0.
            ensures
                - #get_c_candidates() == candidates
        !*/

        const std::vector<scalar_type>& get_c_candidates (
        ) const;
        /*!
            ensures
                - Returns the values of C that train() chooses from.
        !*/

        void set_num_folds (
            long folds
        );
        /*!
            requires
                - folds > 1
            ensures
                - #get_num_folds() == folds
        !*/

        long get_num_folds (
        ) const;
        /*!
            ensures
                - Returns the number of cross-validation folds train() uses.
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - When it cross-validates, train() prints the accuracy of each C candidate
                  to standard out.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - train() prints nothing.
        !*/

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const trained_function_type train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const;
        /*!
            requires
                - x is a vector of sample_type, and y a vector of labels, each either a
                  dlib::matrix or something convertible to one via mat(), such as a
                  std::vector or a view of selected elements made with rowm().
                - is_learning_problem(x,y) == true
                - Every element of y is +1 or -1.
            ensures
                - Trains a class-balanced linear SVM on x and y, choosing C from
                  get_c_candidates() by cross-validation, and returns a function that
                  outputs the probability that a sample's label is +1.
                - Let N be y.size(), N_pos the number of +1 labels, N_neg the number of -1
                  labels and F == get_num_folds().
                - If std::min(N_pos, N_neg) < F, the samples are too few to
                  cross-validate.  Then returns a function that gives N_pos/N for every
                  sample.
                - Otherwise:
                    - Splits the samples into F folds.  The samples of each class are
                      shuffled, and dealt across the folds in turn.  The shuffle is the
                      same on every call, so the same x and y give the same result.
                    - For each C in get_c_candidates(), and each fold, trains the solver
                      on the other folds, with a C for the +1 class of C*M/(2*M_pos) and
                      a C for the -1 class of C*M/(2*M_neg), where M, M_pos and M_neg are
                      the numbers of samples, +1 samples and -1 samples it is trained on.
                    - The accuracy of a candidate is the fraction of held out +1 samples
                      with a score >= 0, and the fraction of held out -1 samples with a
                      score < 0.  The chosen C is the candidate with the largest mean of
                      the two.  When several have the same, it is the first of them.
                    - Returns a probabilistic_decision_function whose decision_funct is
                      the solver trained on all the samples with the chosen C, scaled per
                      class as above, and whose sigmoid is fit by learn_platt_scaling() on
                      the held out scores of the chosen C.
                    - If this object is verbose, prints the accuracy of each candidate.
            throws
                - any exception thrown by the solver's train()
        !*/

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const trained_function_type train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& c_used
        ) const;
        /*!
            requires
                - The requirements of train(x,y) hold.
            ensures
                - Returns train(x,y).
                - #c_used == the C that train(x,y) chose, or 0 if it returned a constant
                  probability without training an SVM.
                - If an exception is thrown, c_used is unchanged.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LINEAR_CLASSIFIER_TRAINER_ABSTRACT_Hh_
