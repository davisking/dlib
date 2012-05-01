// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CROSS_VALIDATE_GRAPh_LABELING_TRAINER_H__
#define DLIB_CROSS_VALIDATE_GRAPh_LABELING_TRAINER_H__

#include "../array.h"
#include "../graph_cuts/min_cut.h"
#include "svm.h"
#include "cross_validate_graph_labeling_trainer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename graph_labeler,
        typename graph_type
        >
    matrix<double,1,2> test_graph_labeling_function (
        const graph_labeler& labeler,
        const dlib::array<graph_type>& samples,
        const std::vector<std::vector<node_label> >& labels
    )
    {
        DLIB_ASSERT(is_graph_labeling_problem(samples, labels) ,
            "\t matrix test_graph_labeling_function()"
            << "\n\t invalid inputs were given to this function"
            << "\n\t samples.size(): " << samples.size() 
            << "\n\t is_graph_labeling_problem(samples,labels): " << is_graph_labeling_problem(samples,labels)
            << "\n\t is_learning_problem(samples,labels):       " << is_learning_problem(samples,labels)
            );

        std::vector<node_label> temp;
        unsigned long num_pos_correct = 0;
        unsigned long num_pos = 0;
        unsigned long num_neg_correct = 0;
        unsigned long num_neg = 0;

        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            labeler(samples[i], temp);

            for (unsigned long j = 0; j < labels[i].size(); ++j)
            {
                if (labels[i][j])
                {
                    ++num_pos;
                    if (temp[j])
                        ++num_pos_correct;
                }
                else
                {
                    ++num_neg;
                    if (!temp[j])
                        ++num_neg_correct;
                }
            }
        }

        matrix<double, 1, 2> res;
        if (num_pos != 0)
            res(0) = (double)num_pos_correct/(double)(num_pos); 
        else
            res(0) = 1;
        if (num_neg != 0)
            res(1) = (double)num_neg_correct/(double)(num_neg); 
        else
            res(1) = 1;
        return res;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename graph_type
        >
    matrix<double,1,2> cross_validate_graph_labeling_trainer (
        const trainer_type& trainer,
        const dlib::array<graph_type>& samples,
        const std::vector<std::vector<node_label> >& labels,
        const long folds
    )
    {
        DLIB_ASSERT(is_graph_labeling_problem(samples, labels) &&
                    1 < folds && folds <= static_cast<long>(samples.size()),
            "\t matrix cross_validate_graph_labeling_trainer()"
            << "\n\t invalid inputs were given to this function"
            << "\n\t samples.size(): " << samples.size() 
            << "\n\t folds:  " << folds 
            << "\n\t is_graph_labeling_problem(samples,labels): " << is_graph_labeling_problem(samples,labels)
            << "\n\t is_learning_problem(samples,labels):       " << is_learning_problem(samples,labels)
            );

        typedef std::vector<node_label> label_type;

        const long num_in_test  = samples.size()/folds;
        const long num_in_train = samples.size() - num_in_test;


        dlib::array<graph_type> samples_test, samples_train;
        std::vector<label_type> labels_test, labels_train;


        long next_test_idx = 0;

        std::vector<node_label> temp;
        unsigned long num_pos_correct = 0;
        unsigned long num_pos = 0;
        unsigned long num_neg_correct = 0;
        unsigned long num_neg = 0;

        graph_type gtemp;

        for (long i = 0; i < folds; ++i)
        {
            samples_test.clear();
            labels_test.clear();
            samples_train.clear();
            labels_train.clear();

            // load up the test samples
            for (long cnt = 0; cnt < num_in_test; ++cnt)
            {
                copy_graph(samples[next_test_idx], gtemp);
                samples_test.push_back(gtemp);
                labels_test.push_back(labels[next_test_idx]);
                next_test_idx = (next_test_idx + 1)%samples.size();
            }

            // load up the training samples
            long next = next_test_idx;
            for (long cnt = 0; cnt < num_in_train; ++cnt)
            {
                copy_graph(samples[next], gtemp);
                samples_train.push_back(gtemp);
                labels_train.push_back(labels[next]);
                next = (next + 1)%samples.size();
            }


            const typename trainer_type::trained_function_type& labeler = trainer.train(samples_train,labels_train);

            // check how good labeler is on the test data
            for (unsigned long i = 0; i < samples_test.size(); ++i)
            {
                labeler(samples_test[i], temp);
                for (unsigned long j = 0; j < labels_test[i].size(); ++j)
                {
                    if (labels_test[i][j])
                    {
                        ++num_pos;
                        if (temp[j])
                            ++num_pos_correct;
                    }
                    else
                    {
                        ++num_neg;
                        if (!temp[j])
                            ++num_neg_correct;
                    }
                }
            }

        } // for (long i = 0; i < folds; ++i)


        matrix<double, 1, 2> res;
        if (num_pos != 0)
            res(0) = (double)num_pos_correct/(double)(num_pos); 
        else
            res(0) = 1;
        if (num_neg != 0)
            res(1) = (double)num_neg_correct/(double)(num_neg); 
        else
            res(1) = 1;
        return res;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CROSS_VALIDATE_GRAPh_LABELING_TRAINER_H__


