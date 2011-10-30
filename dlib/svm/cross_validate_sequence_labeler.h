// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_H__
#define DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_H__

#include "cross_validate_sequence_labeler_abstract.h"
#include <vector>
#include "../matrix.h"
#include "svm.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sequence_labeler_type,
        typename sample_type
        >
    const matrix<double> test_sequence_labeler (
        const sequence_labeler_type& labeler,
        const std::vector<std::vector<sample_type> >& samples,
        const std::vector<std::vector<unsigned long> >& labels
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT( is_sequence_labeling_problem(samples, labels) == true,
                    "\tmatrix test_sequence_labeler()"
                    << "\n\t invalid inputs were given to this function"
                    << "\n\t is_sequence_labeling_problem(samples, labels): " 
                    << is_sequence_labeling_problem(samples, labels));

        matrix<double> res(labeler.num_labels(), labeler.num_labels());
        res = 0;

        std::vector<unsigned long> pred;
        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            labeler.label_sequence(samples[i], pred);

            for (unsigned long j = 0; j < pred.size(); ++j)
            {
                const unsigned long truth = labels[i][j];
                if (truth >= res.nr())
                {
                    // make res big enough for this unexpected label
                    res = join_cols(res, zeros_matrix<double>(truth-res.nr()+1, res.nc()));
                }

                res(truth, pred[j]) += 1;
            }
        }

        return res;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename sample_type
        >
    const matrix<double> cross_validate_sequence_labeler (
        const trainer_type& trainer,
        const std::vector<std::vector<sample_type> >& samples,
        const std::vector<std::vector<unsigned long> >& labels,
        const long folds
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(is_sequence_labeling_problem(samples,labels) == true &&
                    1 < folds && folds <= static_cast<long>(samples.size()),
            "\tmatrix cross_validate_sequence_labeler()"
            << "\n\t invalid inputs were given to this function"
            << "\n\t samples.size(): " << samples.size() 
            << "\n\t folds:  " << folds 
            << "\n\t is_sequence_labeling_problem(samples,labels): " << is_sequence_labeling_problem(samples,labels)
            );



        const long num_in_test = samples.size()/folds;
        const long num_in_train = samples.size() - num_in_test;

        std::vector<std::vector<sample_type> > x_test, x_train;
        std::vector<std::vector<unsigned long> > y_test, y_train;


        long next_test_idx = 0;

        matrix<double> res;


        for (long i = 0; i < folds; ++i)
        {
            x_test.clear();
            y_test.clear();
            x_train.clear();
            y_train.clear();

            // load up the test samples
            for (long cnt = 0; cnt < num_in_test; ++cnt)
            {
                x_test.push_back(samples[next_test_idx]);
                y_test.push_back(labels[next_test_idx]);
                next_test_idx = (next_test_idx + 1)%samples.size();
            }

            // load up the training samples
            long next = next_test_idx;
            for (long cnt = 0; cnt < num_in_train; ++cnt)
            {
                x_train.push_back(samples[next]);
                y_train.push_back(labels[next]);
                next = (next + 1)%samples.size();
            }


            matrix<double> temp = test_sequence_labeler(trainer.train(x_train,y_train), x_test, y_test);

            // Make sure res is always at least as big as temp.  This might not be the case
            // because temp is sized differently depending on how many different kinds of labels 
            // test_sequence_labeler() sees.
            if (get_rect(res).contains(get_rect(temp)) == false)
            {
                if (res.size() == 0)
                {
                    res.set_size(temp.nr(), temp.nc());
                    res = 0;
                }

                // Make res bigger by padding with zeros on the bottom or right if necessary.
                if (res.nr() < temp.nr())
                    res = join_cols(res, zeros_matrix<double>(temp.nr()-res.nc(), res.nc()));
                if (res.nc() < temp.nc())
                    res = join_rows(res, zeros_matrix<double>(res.nr(), temp.nc()-res.nc()));

            }

            // add temp to res
            for (long r = 0; r < temp.nr(); ++r)
            {
                for (long c = 0; c < temp.nc(); ++c)
                {
                    res(r,c) += temp(r,c);
                }
            }


        } // for (long i = 0; i < folds; ++i)

        return res;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CROSS_VALIDATE_SEQUENCE_LABeLER_H__


