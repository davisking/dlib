// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_ABSTRACT_H__
#ifdef DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_ABSTRACT_H__

#include <vector>
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename reg_funct_type,
        typename sample_type,
        typename label_type
        >
    label_type
    test_regression_function (
        const reg_funct_type& reg_funct,
        const std::vector<sample_type>& x_test,
        const std::vector<label_type>& y_test
    );
    /*!
        requires
            - is_learning_problem(x_test, y_test)
            - reg_funct_type == some kind of regression function object 
              (e.g. a decision_function created by the svr_trainer )
        ensures
            - Tests reg_funct against the given samples in x_test and target values in 
              y_test and returns the mean squared error.  Specifically, the MSE is given
              by:
                sum over i: pow(reg_funct(x_test[i]) - y_test[i], 2.0)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename sample_type,
        typename label_type 
        >
    label_type 
    cross_validate_regression_trainer (
        const trainer_type& trainer,
        const std::vector<sample_type>& x,
        const std::vector<label_type>& y,
        const long folds
    );
    /*!
        requires
            - is_learning_problem(x,y)
            - 1 < folds <= x.size()
            - trainer_type == some kind of regression trainer object (e.g. svr_trainer)
        ensures
            - performs k-fold cross validation by using the given trainer to solve the
              given regression problem for the given number of folds.  Each fold is tested using 
              the output of the trainer and the mean squared error is computed and returned.
            - The total MSE is computed by running test_binary_decision_function()
              on each fold and averaging its output.
    !*/

}

// ----------------------------------------------------------------------------------------

#endif // DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_ABSTRACT_H__



