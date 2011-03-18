// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_H__
#define DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_H__

#include <vector>
#include "../matrix.h"
#include "../statistics.h"

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
    )
    {
        typedef typename reg_funct_type::mem_manager_type mem_manager_type;

        // make sure requires clause is not broken
        DLIB_ASSERT( is_learning_problem(x_test,y_test) == true,
                    "\tmatrix test_regression_function()"
                    << "\n\t invalid inputs were given to this function"
                    << "\n\t is_learning_problem(x_test,y_test): " 
                    << is_learning_problem(x_test,y_test));

        running_stats<label_type> rs;

        for (unsigned long i = 0; i < x_test.size(); ++i)
        {
            // compute error
            label_type temp = reg_funct(x_test[i]) - y_test[i];

            rs.add(temp*temp);
        }

        return rs.mean();
    }

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
    )
    {
        typedef typename trainer_type::mem_manager_type mem_manager_type;

        // make sure requires clause is not broken
        DLIB_ASSERT(is_learning_problem(x,y) == true &&
                    1 < folds && folds <= static_cast<long>(x.size()),
            "\tmatrix cross_validate_regression_trainer()"
            << "\n\t invalid inputs were given to this function"
            << "\n\t x.size(): " << x.size() 
            << "\n\t folds:  " << folds 
            << "\n\t is_learning_problem(x,y): " << is_learning_problem(x,y)
            );



        const long num_in_test = x.size()/folds;
        const long num_in_train = x.size() - num_in_test;


        std::vector<sample_type> x_test, x_train;
        std::vector<label_type> y_test, y_train;

        running_stats<label_type> rs;

        long next_test_idx = 0;


        for (long i = 0; i < folds; ++i)
        {
            x_test.clear();
            y_test.clear();
            x_train.clear();
            y_train.clear();

            // load up the test samples
            for (long cnt = 0; cnt < num_in_test; ++cnt)
            {
                x_test.push_back(x[next_test_idx]);
                y_test.push_back(y[next_test_idx]);
                next_test_idx = (next_test_idx + 1)%x.size();
            }

            // load up the training samples
            long next = next_test_idx;
            for (long cnt = 0; cnt < num_in_train; ++cnt)
            {
                x_train.push_back(x[next]);
                y_train.push_back(y[next]);
                next = (next + 1)%x.size();
            }


            try
            {
                // do the training and testing
                rs.add(test_regression_function(trainer.train(x_train,y_train),x_test,y_test));
            }
            catch (invalid_nu_error&)
            {
                // just ignore cases which result in an invalid nu
            }

        } // for (long i = 0; i < folds; ++i)

        return rs.mean();
    }

}

// ----------------------------------------------------------------------------------------

#endif // DLIB_CROSS_VALIDATE_REGRESSION_TRaINER_H__


