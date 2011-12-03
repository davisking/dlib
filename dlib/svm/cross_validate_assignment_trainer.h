// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CROSS_VALIDATE_ASSiGNEMNT_TRAINER_H__
#define DLIB_CROSS_VALIDATE_ASSiGNEMNT_TRAINER_H__

#include "cross_validate_assignment_trainer_abstract.h"
#include <vector>
#include "../matrix.h"
#include "svm.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename assignment_function
        >
    double test_assignment_function (
        const assignment_function& assigner,
        const std::vector<typename assignment_function::sample_type>& samples,
        const std::vector<typename assignment_function::label_type>& labels
    )
    {
        double total_right = 0;
        double total = 0;
        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            const std::vector<long>& out = assigner(samples[i]);
            for (unsigned long j = 0; j < out.size(); ++j)
            {
                if (out[j] == labels[i][j])
                    ++total_right;

                ++total;
            }
        }

        if (total != 0)
            return total_right/total;
        else
            return 1;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type
        >
    double cross_validate_assignment_trainer (
        const trainer_type& trainer,
        const std::vector<typename trainer_type::sample_type>& samples,
        const std::vector<typename trainer_type::label_type>& labels,
        const long folds
    )
    {
        typedef typename trainer_type::sample_type sample_type;
        typedef typename trainer_type::label_type label_type;

        const long num_in_test  = samples.size()/folds;
        const long num_in_train = samples.size() - num_in_test;

        running_stats<double> rs;

        std::vector<sample_type> samples_test, samples_train;
        std::vector<label_type> labels_test, labels_train;


        long next_test_idx = 0;


        for (long i = 0; i < folds; ++i)
        {
            samples_test.clear();
            labels_test.clear();
            samples_train.clear();
            labels_train.clear();

            // load up the test samples
            for (long cnt = 0; cnt < num_in_test; ++cnt)
            {
                samples_test.push_back(samples[next_test_idx]);
                labels_test.push_back(labels[next_test_idx]);
                next_test_idx = (next_test_idx + 1)%samples.size();
            }

            // load up the training samples
            long next = next_test_idx;
            for (long cnt = 0; cnt < num_in_train; ++cnt)
            {
                samples_train.push_back(samples[next]);
                labels_train.push_back(labels[next]);
                next = (next + 1)%samples.size();
            }


            rs.add(test_assignment_function(trainer.train(samples_train,labels_train),
                                            samples_test,
                                            labels_test));

        } // for (long i = 0; i < folds; ++i)

        return rs.mean();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CROSS_VALIDATE_ASSiGNEMNT_TRAINER_H__

