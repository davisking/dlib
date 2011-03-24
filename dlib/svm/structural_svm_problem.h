// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SVM_PRObLEM_H__
#define DLIB_STRUCTURAL_SVM_PRObLEM_H__

#include "structural_svm_problem_abstract.h"
#include "../algs.h"
#include <vector>
#include "../optimization/optimization_oca.h"
#include "../matrix.h"
#include "sparse_vector.h"
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type,
        typename feature_vector_type_ 
        >
    class structural_svm_problem : public oca_problem<matrix_type> 
    {
    public:
        /*!
            CONVENTION
                - C == get_c()
                - eps == get_epsilon()
                - if (skip_cache) then
                    - we won't use the oracle cache when we need to evaluate the separation
                      oracle. Instead, we will directly call the user supplied separation_oracle().

                - get_max_cache_size() == max_cache_size

                - if (cache.size() != 0) then
                    - cache.size() == get_num_samples()
                    - for all i: cache[i] == the cached results of calls to separation_oracle()
                      for the i-th sample.
        !*/

        typedef typename matrix_type::type scalar_type;
        typedef feature_vector_type_ feature_vector_type;

        structural_svm_problem (
        ) :
            eps(0.001),
            verbose(false),
            skip_cache(true),
            max_cache_size(10),
            C(1)
        {}

        void set_epsilon (
            scalar_type eps_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\t void structural_svm_problem::set_epsilon()"
                << "\n\t eps_ must be greater than 0"
                << "\n\t eps_: " << eps_ 
                << "\n\t this: " << this
                );

            eps = eps_;
        }

        const scalar_type get_epsilon (
        ) const { return eps; }

        void set_max_cache_size (
            unsigned long max_size
        )
        {
            max_cache_size = max_size;
        }

        unsigned long get_max_cache_size (
        ) const { return max_cache_size; }

        void be_verbose (
        ) 
        {
            verbose = true;
        }

        void be_quiet(
        )
        {
            verbose = false;
        }

        scalar_type get_c (
        ) const { return C; }

        void set_c (
            scalar_type C_
        ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(C_ > 0,
                "\t void structural_svm_problem::set_c()"
                << "\n\t C_ must be greater than 0"
                << "\n\t C_:    " << C_ 
                << "\n\t this: " << this
                );

            C = C_; 
        }

        virtual long get_num_dimensions (
        ) const = 0;

        virtual long get_num_samples (
        ) const = 0;

        virtual void get_truth_joint_feature_vector (
            long idx,
            feature_vector_type& psi 
        ) const = 0;

        virtual void separation_oracle (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const = 0;

    private:

        virtual bool risk_has_lower_bound (
            scalar_type& lower_bound
        ) const 
        { 
            lower_bound = 0;
            return true; 
        }

        virtual bool optimization_status (
            scalar_type current_objective_value,
            scalar_type current_error_gap,
            unsigned long num_cutting_planes,
            unsigned long num_iterations
        ) const 
        {
            if (verbose)
            {
                using namespace std;
                cout << "svm objective: " << current_objective_value << endl;
                cout << "gap: " << current_error_gap << endl;
                cout << "num planes: " << num_cutting_planes << endl;
                cout << "iter: " << num_iterations << endl;
                cout << endl;
            }

            cur_gap = current_error_gap/get_c();

            bool should_stop = false;

            if (current_objective_value == 0)
                should_stop = true;
            else if (current_error_gap/current_objective_value < eps)
                should_stop = true;

            if (should_stop && !skip_cache)
            {
                // Instead of stopping we shouldn't use the cache on the next iteration.  This way
                // we can be sure to have the best solution rather than assuming the cache is up-to-date
                // enough.
                should_stop = false;
                skip_cache = true;
            }
            else
            {
                skip_cache = false;
            }


            return should_stop;
        }

        virtual void get_risk (
            matrix_type& w,
            scalar_type& risk,
            matrix_type& subgradient
        ) const 
        {
            feature_vector_type ftemp;
            const unsigned long num = get_num_samples();

            // initialize psi_true if we haven't done so already.  
            if (psi_true.size() == 0)
            {
                psi_true.set_size(w.size(),1);
                psi_true = 0;
                for (unsigned long i = 0; i < num; ++i)
                {
                    get_truth_joint_feature_vector(i, ftemp);
                    sparse_vector::subtract_from(psi_true, ftemp);
                }
            }

            subgradient = psi_true;
            scalar_type total_loss = 0;
            for (unsigned long i = 0; i < num; ++i)
            {
                scalar_type loss;
                separation_oracle_cached(i, w, loss, ftemp);
                total_loss += loss;
                sparse_vector::add_to(subgradient, ftemp);
            }

            subgradient /= num;
            total_loss /= num;
            risk = total_loss + dot(subgradient,w);
            cur_risk = risk;
        }

        void separation_oracle_cached (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const 
        {

            if (cache.size() == 0 && max_cache_size != 0)
                cache.resize(get_num_samples());

            if (!skip_cache && max_cache_size != 0)
            {
                scalar_type best_val = -std::numeric_limits<scalar_type>::infinity();
                unsigned long best_idx = 0;

                cache_record& rec = cache[idx];

                // figure out which element in the cache is the best
                for (unsigned long i = 0; i < rec.loss.size(); ++i)
                {
                    using sparse_vector::dot;
                    const scalar_type temp = rec.loss[i] + dot(rec.psi[i], current_solution);
                    if (temp > best_val)
                    {
                        best_val = temp;
                        loss = rec.loss[i];
                        best_idx = i;
                    }
                }

                if (best_val > cur_risk-cur_gap)
                {
                    psi = rec.psi[best_idx];
                    rec.lru_count[best_idx] += 1;
                    return;
                }
            }


            separation_oracle(idx, current_solution, loss, psi);

            if (cache.size() != 0)
            {
                if (cache[idx].loss.size() < max_cache_size)
                {
                    cache[idx].loss.push_back(loss);
                    cache[idx].psi.push_back(psi);
                    cache[idx].lru_count.push_back(cache[idx].lru_count.size());
                }
                else
                {
                    // find least recently used cache entry for idx-th sample
                    const long i       = index_of_min(vector_to_matrix(cache[idx].lru_count));

                    // save our new data in the cache
                    cache[idx].loss[i] = loss;
                    cache[idx].psi[i]  = psi;

                    const long max_use = max(vector_to_matrix(cache[idx].lru_count));
                    // Make sure this new cache entry has the best lru count since we have used
                    // it most recently.
                    cache[idx].lru_count[i] = max_use + 1;
                }
            }
        }

        struct cache_record
        {
            std::vector<scalar_type> loss;
            std::vector<feature_vector_type> psi;
            std::vector<long> lru_count;
        };


        mutable scalar_type cur_risk;
        mutable scalar_type cur_gap;
        mutable matrix_type psi_true;
        scalar_type eps;
        mutable bool verbose;

        mutable std::vector<cache_record> cache;
        mutable bool skip_cache;
        unsigned long max_cache_size;

        scalar_type C;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_PRObLEM_H__

