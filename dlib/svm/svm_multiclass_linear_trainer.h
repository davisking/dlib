// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVm_MULTICLASS_LINEAR_TRAINER_H__ 
#define DLIB_SVm_MULTICLASS_LINEAR_TRAINER_H__

#include "svm_multiclass_linear_trainer_abstract.h"
#include <vector>
#include "../optimization/optimization_oca.h"
#include "../matrix.h"
#include "sparse_vector.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type,
        typename sample_type,
        typename label_type
        >
    class multiclass_svm_problem : public structural_svm_problem<matrix_type,
                                                                 std::vector<std::pair<unsigned long,typename matrix_type::type> > > 
    {
    public:
        typedef typename matrix_type::type scalar_type;
        typedef std::vector<std::pair<unsigned long,scalar_type> > feature_vector_type;

        multiclass_svm_problem (
            const std::vector<sample_type>& samples_,
            const std::vector<label_type>& labels_
        ) :
            samples(samples_),
            labels(labels_),
            distinct_labels(select_all_distinct_labels(labels_)),
            dims(sparse_vector::max_index_plus_one(samples_)+1) // +1 for the bias
        {}

        virtual long get_num_dimensions (
        ) const
        {
            return dims*distinct_labels.size();
        }

        virtual long get_num_samples (
        ) const 
        {
            return static_cast<long>(samples.size());
        }

        virtual void get_truth_joint_feature_vector (
            long idx,
            feature_vector_type& psi
        ) const 
        {
            sparse_vector::assign(psi, samples[idx]);
            // Add a constant -1 to account for the bias term.
            psi.push_back(std::make_pair(dims-1,-1));

            // Find which distinct label goes with this psi.
            const long label_idx = index_of_max(vector_to_matrix(distinct_labels) == labels[idx]);

            offset_feature_vector(psi, dims*label_idx);
        }

        virtual void separation_oracle (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const 
        {
            scalar_type best_val = -std::numeric_limits<scalar_type>::infinity();
            unsigned long best_idx = 0;

            // figure out which label is the best
            for (unsigned long i = 0; i < distinct_labels.size(); ++i)
            {
                using sparse_vector::dot;
                // perform: temp == dot(relevant part of current solution, samples[idx]) - current_bias
                scalar_type temp = dot(rowm(current_solution, range(i*dims, (i+1)*dims-2)), samples[idx]) - current_solution((i+1)*dims-1);

                if (labels[idx] != distinct_labels[i])
                    temp += 1;

                if (temp > best_val)
                {
                    best_val = temp;
                    best_idx = i;
                }
            }

            sparse_vector::assign(psi, samples[idx]);
            // add a constant -1 to account for the bias term
            psi.push_back(std::make_pair(dims-1,-1));

            offset_feature_vector(psi, dims*best_idx);

            if (distinct_labels[best_idx] == labels[idx])
                loss = 0;
            else
                loss = 1;
        }

    private:

        void offset_feature_vector (
            feature_vector_type& sample,
            const unsigned long val
        ) const
        {
            if (val != 0)
            {
                for (typename feature_vector_type::iterator i = sample.begin(); i != sample.end(); ++i)
                {
                    i->first += val;
                }
            }
        }


        const std::vector<sample_type>& samples;
        const std::vector<label_type>& labels;
        const std::vector<label_type> distinct_labels;
        const long dims;
    };


// ----------------------------------------------------------------------------------------

    template <
        typename K,
        typename label_type_ = typename K::scalar_type 
        >
    class svm_multiclass_linear_trainer
    {
    public:
        typedef label_type_ label_type;
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;

        typedef multiclass_linear_decision_function<kernel_type, label_type> trained_function_type;


        trained_function_type train (
            const std::vector<sample_type>& all_samples,
            const std::vector<label_type>& all_labels
        ) const
        {
            oca solver;
            typedef matrix<scalar_type,0,1> w_type;
            w_type weights;
            multiclass_svm_problem<w_type, sample_type, label_type> problem(all_samples, all_labels);
            problem.be_verbose();
            problem.set_max_cache_size(0);
            problem.set_c(100);

            solver(problem, weights);

            trained_function_type df;

            const long dims = sparse_vector::max_index_plus_one(all_samples);
            df.labels  = select_all_distinct_labels(all_labels);
            df.weights = colm(reshape(weights, df.labels.size(), dims+1), range(0,dims-1));
            df.b       = colm(reshape(weights, df.labels.size(), dims+1), dims);
            return df;
        }
    };

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_SVm_MULTICLASS_LINEAR_TRAINER_H__

