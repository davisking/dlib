// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SVM_PRObLEM_THREADED_H__
#define DLIB_STRUCTURAL_SVM_PRObLEM_THREADED_H__

#include "structural_svm_problem_threaded_abstract.h"
#include "../algs.h"
#include <vector>
#include "structural_svm_problem.h"
#include "../matrix.h"
#include "sparse_vector.h"
#include <iostream>
#include "../threads.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type,
        typename feature_vector_type_ = matrix_type
        >
    class structural_svm_problem_threaded : public structural_svm_problem<matrix_type,feature_vector_type_> 
    {
    public:

        typedef typename matrix_type::type scalar_type;
        typedef feature_vector_type_ feature_vector_type;

        explicit structural_svm_problem_threaded (
            unsigned long num_threads
        ) :
            tp(num_threads)
        {}

        unsigned long get_num_threads (
        ) const { return tp.num_threads_in_pool(); }

    private:

        struct binder
        {
            binder (
                const structural_svm_problem_threaded& self_,
                matrix_type& w_,
                matrix_type& subgradient_,
                scalar_type& total_loss_
            ) : self(self_), w(w_), subgradient(subgradient_), total_loss(total_loss_) {}

            void call_oracle (
                long i
            ) 
            {
                scalar_type loss;
                feature_vector_type ftemp;
                self.separation_oracle_cached(i, w, loss, ftemp);

                auto_mutex lock(self.accum_mutex);
                total_loss += loss;
                sparse_vector::add_to(subgradient, ftemp);
            }

            const structural_svm_problem_threaded& self;
            matrix_type& w;
            matrix_type& subgradient;
            scalar_type& total_loss;
        };


        virtual void call_separation_oracle_on_all_samples (
            matrix_type& w,
            matrix_type& subgradient,
            scalar_type& total_loss
        ) const
        {
            const long num = this->get_num_samples();
            binder b(*this, w, subgradient, total_loss);
            for (long i = 0; i < num; ++i)
            {
                tp.add_task(b, &binder::call_oracle, i);
            }
            tp.wait_for_all_tasks();
        }

        mutable thread_pool tp;
        mutable mutex accum_mutex;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_PRObLEM_THREADED_H__


