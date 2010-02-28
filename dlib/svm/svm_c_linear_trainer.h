// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVM_C_LiNEAR_TRAINER_H__
#define DLIB_SVM_C_LiNEAR_TRAINER_H__

#include "svm_c_linear_trainer_abstract.h"
#include "../algs.h"
#include "../optimization.h"
#include "../matrix.h"
#include "function.h"
#include "kernel.h"
#include <iostream>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename matrix_type, 
        typename in_sample_vector_type,
        typename in_scalar_vector_type
        >
    class oca_problem_c_svm : public oca_problem<matrix_type >
    {
    public:
        /*
            This class is used as part of the implementation of the svm_c_linear_trainer
            defined towards the end of this file.


            The bias parameter is dealt with by imagining that each sample vector has -1
            as its last element.
        */

        typedef typename matrix_type::type scalar_type;

        oca_problem_c_svm(
            const scalar_type C_pos,
            const scalar_type C_neg,
            const in_sample_vector_type& samples_,
            const in_scalar_vector_type& labels_,
            bool be_verbose_
        ) :
            samples(samples_),
            labels(labels_),
            Cpos(C_pos),
            Cneg(C_neg),
            be_verbose(be_verbose_)
        {
            dot_prods.resize(samples.size());
            is_first_call = true;
        }

        virtual scalar_type get_c (
        ) const 
        {
            return 1;
        }

        virtual long get_num_dimensions (
        ) const 
        {
            // plus 1 for the bias term
            return num_dimensions_in_samples(samples) + 1;
        }

        virtual bool optimization_status (
            scalar_type current_objective_value,
            scalar_type current_error_gap,
            unsigned long num_cutting_planes,
            unsigned long num_iterations
        ) const 
        {
            if (be_verbose)
            {
                using namespace std;
                cout << "svm objective: " << current_objective_value << endl;
                cout << "gap: " << current_error_gap << endl;
                cout << "num planes: " << num_cutting_planes << endl;
                cout << "iter: " << num_iterations << endl;
                cout << endl;
            }

            if (current_error_gap/current_objective_value < 0.001)
                return true;

            if (num_iterations > 10000)
                return true;

            return false;
        }

        virtual bool r_has_lower_bound (
            scalar_type& lower_bound
        ) const 
        { 
            lower_bound = 0;
            return true; 
        }

        virtual void get_risk (
            matrix_type& w,
            scalar_type& risk,
            matrix_type& subgradient
        ) const 
        {
            line_search(w);

            subgradient.set_size(w.size(),1);
            subgradient = 0;
            risk = 0;


            // loop over all the samples and compute the risk and its subgradient at the current solution point w
            for (long i = 0; i < samples.size(); ++i)
            {
                // multiply current SVM output for the ith sample by its label
                const scalar_type df_val = labels(i)*dot_prods[i];

                if (labels(i) > 0)
                    risk += Cpos*std::max<scalar_type>(0.0,1 - df_val);
                else
                    risk += Cneg*std::max<scalar_type>(0.0,1 - df_val);

                if (df_val < 1)
                {
                    if (labels(i) > 0)
                    {
                        subtract_from(subgradient, samples(i), Cpos);

                        subgradient(subgradient.size()-1) += Cpos;
                    }
                    else
                    {
                        add_to(subgradient, samples(i), Cneg);

                        subgradient(subgradient.size()-1) -= Cneg;
                    }
                }
            }

            scalar_type scale = 1.0/samples.size();

            risk *= scale;
            subgradient = scale*subgradient;
        }

    private:

    // -----------------------------------------------------
    // -----------------------------------------------------

        // The next few functions are overloads to handle both dense and sparse vectors
        template <typename EXP>
        inline void add_to (
            matrix_type& subgradient,
            const matrix_exp<EXP>& sample,
            const scalar_type& C
        ) const
        {
            for (long r = 0; r < sample.size(); ++r)
                subgradient(r) += C*sample(r);
        }

        template <typename T>
        inline typename disable_if<is_matrix<T> >::type add_to (
            matrix_type& subgradient,
            const T& sample,
            const scalar_type& C
        ) const
        {
            for (unsigned long i = 0; i < sample.size(); ++i)
                subgradient(sample[i].first) += C*sample[i].second;
        }

        template <typename EXP>
        inline void subtract_from (
            matrix_type& subgradient,
            const matrix_exp<EXP>& sample,
            const scalar_type& C
        ) const
        {
            for (long r = 0; r < sample.size(); ++r)
                subgradient(r) -= C*sample(r);
        }

        template <typename T>
        inline typename disable_if<is_matrix<T> >::type subtract_from (
            matrix_type& subgradient,
            const T& sample,
            const scalar_type& C
        ) const
        {
            for (unsigned long i = 0; i < sample.size(); ++i)
                subgradient(sample[i].first) -= C*sample[i].second;
        }

        template <typename EXP>
        scalar_type dot_helper (
            const matrix_type& w,
            const matrix_exp<EXP>& sample
        ) const
        {
            return dot(colm(w,0,w.size()-1), sample);
        }

        template <typename T>
        typename disable_if<is_matrix<T>,scalar_type >::type dot_helper (
            const matrix_type& w,
            const T& sample
        ) const
        {
            // compute a dot product between a dense column vector and a sparse vector
            scalar_type temp = 0;
            for (unsigned long i = 0; i < sample.size(); ++i)
                temp += w(sample[i].first) * sample[i].second;
        }

        template <typename T>
        typename enable_if<is_matrix<typename T::type>,unsigned long>::type num_dimensions_in_samples (
            const T& samples
        ) const
        {
            if (samples.size() > 0)
                return samples(0).size();
            else
                return 0;
        }

        template <typename T>
        typename disable_if<is_matrix<typename T::type>,unsigned long>::type num_dimensions_in_samples (
            const T& samples
        ) const
        /*!
            T must be a sparse vector with an integral key type
        !*/
        {
            // these should be sparse samples so look over all them to find the max dimension.
            unsigned long max_dim = 0;
            for (long i = 0; i < samples.size(); ++i)
            {
                if (samples(i).size() > 0)
                    max_dim = std::max<unsigned long>(max_dim, samples(i).back().first + 1);
            }

            return max_dim;
        }
        
    // -----------------------------------------------------
    // -----------------------------------------------------

        void line_search (
            matrix_type& w
        ) const
        /*!
            ensures
                - does a line search to find a better w
                - for all i: #dot_prods[i] == dot(colm(#w,0,w.size()-1), samples(i)) - #w(w.size()-1)
        !*/
        {
            const scalar_type mu = 0.1;

            for (long i = 0; i < samples.size(); ++i)
                dot_prods[i] = dot_helper(w,samples(i)) - w(w.size()-1);


            if (is_first_call)
            {
                is_first_call = false;
                best_so_far = w;
                dot_prods_best = dot_prods;
            }
            else
            {
                // do line search going from best_so_far to w.  Store results in w.  
                // Here we use the line search algorithm presented in section 3.1.1 of Franc and Sonnenburg.

                const scalar_type A0 = length_squared(best_so_far - w);
                const scalar_type B0 = dot(best_so_far, w - best_so_far);

                const scalar_type scale_pos = (get_c()*Cpos)/samples.size();
                const scalar_type scale_neg = (get_c()*Cneg)/samples.size();

                ks.clear();
                ks.reserve(samples.size());

                scalar_type f0 = B0;
                for (long i = 0; i < samples.size(); ++i)
                {
                    const scalar_type& scale = (labels(i)>0) ? scale_pos : scale_neg;

                    const scalar_type B = scale*labels(i) * ( dot_prods_best[i] - dot_prods[i]);
                    const scalar_type C = scale*(1 - labels(i)* dot_prods_best[i]);
                    // Note that if B is 0 then it doesn't matter what k is set to.  So 0 is fine.
                    scalar_type k = 0;
                    if (B != 0)
                        k = -C/B;

                    if (k > 0)
                        ks.push_back(helper(k, std::abs(B)));

                    if ( (B < 0 && k > 0) || (B > 0 && k <= 0) )
                        f0 += B;
                }

                // ks.size() == 0 shouldn't happen but check anyway
                if (f0 >= 0 || ks.size() == 0)
                {
                    // getting here means that we aren't searching in a descent direction.  So don't
                    // move the best_so_far position.
                }
                else
                {
                    std::sort(ks.begin(), ks.end());

                    // figure out where f0 goes positive.
                    scalar_type opt_k = 1;
                    for (unsigned long i = 0; i < ks.size(); ++i)
                    {
                        f0 += ks[i].B;
                        if (f0 + A0*ks[i].k >= 0)
                        {
                            opt_k = ks[i].k;
                            break;
                        }
                    }

                    // take the step suggested by the line search
                    best_so_far = (1-opt_k)*best_so_far + opt_k*w;

                    // update best_so_far dot products
                    for (unsigned long i = 0; i < dot_prods_best.size(); ++i)
                        dot_prods_best[i] = (1-opt_k)*dot_prods_best[i] + opt_k*dot_prods[i];
                }

                // Put the best_so_far point into w but also take a little bit of w as well.  We do
                // this since it is possible that some steps won't advance the best_so_far point. 
                // So this ensures we always make some progress each iteration.
                w = (1-mu)*best_so_far + mu*w;

                // update dot products
                for (unsigned long i = 0; i < dot_prods.size(); ++i)
                    dot_prods[i] = (1-mu)*dot_prods_best[i] + mu*dot_prods[i];
            }
        }

        struct helper
        {
            helper(scalar_type k_, scalar_type B_) : k(k_), B(B_) {}
            scalar_type k;
            scalar_type B;

            bool operator< (const helper& item) const { return k < item.k; }
        };

        mutable std::vector<helper> ks;

        mutable bool is_first_call;
        mutable std::vector<scalar_type> dot_prods;

        mutable matrix_type best_so_far;  // best w seen so far
        mutable std::vector<scalar_type> dot_prods_best; // dot products between best_so_far and samples


        const in_sample_vector_type& samples;
        const in_scalar_vector_type& labels;
        const scalar_type Cpos;
        const scalar_type Cneg;

        bool be_verbose;
    };


    template <
        typename matrix_type, 
        typename in_sample_vector_type,
        typename in_scalar_vector_type,
        typename scalar_type
        >
    oca_problem_c_svm<matrix_type, in_sample_vector_type, in_scalar_vector_type> make_oca_problem_c_svm (
        const scalar_type C_pos,
        const scalar_type C_neg,
        const in_sample_vector_type& samples,
        const in_scalar_vector_type& labels,
        bool be_verbose
    )
    {
        return oca_problem_c_svm<matrix_type, in_sample_vector_type, in_scalar_vector_type>(C_pos, C_neg, samples, labels, be_verbose);
    }
// ----------------------------------------------------------------------------------------

    template <
        typename K 
        >
    class svm_c_linear_trainer
    {
        /*!
            REQUIREMENTS ON K 
                is either linear_kernel or sparse_linear_kernel

            WHAT THIS OBJECT REPRESENTS
        !*/

    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        svm_c_linear_trainer (
        )
        /*!
            ensures
                - This object is properly initialized and ready to be used
                  to train a support vector machine.
                - #get_oca() == oca() (i.e. an instance of oca with default parameters) 
                - #get_c_class1() == 1
                - #get_c_class2() == 1
        !*/
        {
            Cpos = 1;
            Cneg = 1;
        }

        explicit svm_c_linear_trainer (
            const scalar_type& C 
        )
        /*!
            requires
                - C > 0
            ensures
                - This object is properly initialized and ready to be used
                  to train a support vector machine.
                - #get_oca() == oca() (i.e. an instance of oca with default parameters) 
                - #get_c_class1() == C
                - #get_c_class2() == C
        !*/
        {
            Cpos = C;
            Cneg = C;
        }

        void set_oca (
            const oca& item
        )
        /*!
            ensures
                - #get_oca() == item 
        !*/
        {
            solver = item;
        }

        const oca get_oca (
        ) const
        /*!
            ensures
                - returns a copy of the optimizer used to solve the SVM problem.  
        !*/
        {
            return solver;
        }

        const kernel_type get_kernel (
        ) const
        /*!
            ensures
                - returns a copy of the kernel function in use by this object
        !*/
        {
            return kernel_type();
        }

        void set_c (
            scalar_type C 
        )
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class1() == C 
                - #get_c_class2() == C 
        !*/
        {
            Cpos = C;
            Cneg = C;
        }

        const scalar_type get_c_class1 (
        ) const
        /*!
            ensures
                - returns the SVM regularization parameter for the +1 class.  
                  It is the parameter that determines the trade off between
                  trying to fit the +1 training data exactly or allowing more errors 
                  but hopefully improving the generalization ability of the 
                  resulting classifier.  Larger values encourage exact fitting 
                  while smaller values of C may encourage better generalization. 
        !*/
        {
            return Cpos;
        }

        const scalar_type get_c_class2 (
        ) const
        /*!
            ensures
                - returns the SVM regularization parameter for the -1 class.  
                  It is the parameter that determines the trade off between
                  trying to fit the -1 training data exactly or allowing more errors 
                  but hopefully improving the generalization ability of the 
                  resulting classifier.  Larger values encourage exact fitting 
                  while smaller values of C may encourage better generalization. 
        !*/
        {
            return Cneg;
        }

        void set_c_class1 (
            scalar_type C
        )
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class1() == C
        !*/
        {
            Cpos = C;
        }

        void set_c_class2 (
            scalar_type C
        )
        /*!
            requires
                - C > 0
            ensures
                - #get_c_class2() == C
        !*/
        {
            Cneg = C;
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        /*!
            requires
                - is_binary_classification_problem(x,y) == true
                - x == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, x should contain sample_type objects.
                - y == a matrix or something convertible to a matrix via vector_to_matrix().
                  Also, y should contain scalar_type objects.
            ensures
                - trains a C support vector classifier given the training samples in x and 
                  labels in y.  
                - returns a decision function F with the following properties:
                    - if (new_x is a sample predicted have +1 label) then
                        - F(new_x) >= 0
                    - else
                        - F(new_x) < 0
        !*/
        {
            typedef matrix<scalar_type,0,1> w_type;
            w_type w;

            scalar_type obj = solver(make_oca_problem_c_svm<w_type>(Cpos, Cneg, vector_to_matrix(x), vector_to_matrix(y), true), w);

            std::cout << "final obj: "<< obj << std::endl;

            // put the solution into a decision function and then return it
            decision_function<kernel_type> df;
            df.b = static_cast<scalar_type>(w(w.size()-1));
            df.basis_vectors.set_size(1);
            df.basis_vectors(0) = matrix_cast<scalar_type>(colm(w, 0, w.size()-1));
            df.alpha.set_size(1);
            df.alpha(0) = 1;

            return df;
        }

    private:
        
        scalar_type Cpos;
        scalar_type Cneg;
        oca solver;
    }; 

// ----------------------------------------------------------------------------------------

}

// ----------------------------------------------------------------------------------------


#endif // DLIB_OCA_PROBLeM_SVM_C_H__

