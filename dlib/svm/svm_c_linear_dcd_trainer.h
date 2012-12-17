// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVm_C_LINEAR_DCD_TRAINER_H__ 
#define DLIB_SVm_C_LINEAR_DCD_TRAINER_H__

#include "svm_c_linear_dcd_trainer_abstract.h"
#include <cmath>
#include <limits>
#include "../matrix.h"
#include "../algs.h"
#include "../rand.h"

#include "function.h"
#include "kernel.h"

namespace dlib 
{

// ----------------------------------------------------------------------------------------

    template <
        typename K 
        >
    class svm_c_linear_dcd_trainer
    {
    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;
        typedef typename decision_function<K>::sample_vector_type sample_vector_type;
        typedef typename decision_function<K>::scalar_vector_type scalar_vector_type;

        // You are getting a compiler error on this line because you supplied a non-linear
        // kernel to the svm_c_linear_dcd_trainer object.  You have to use one of the
        // linear kernels with this trainer.
        COMPILE_TIME_ASSERT((is_same_type<K, linear_kernel<sample_type> >::value ||
                             is_same_type<K, sparse_linear_kernel<sample_type> >::value ));

        svm_c_linear_dcd_trainer (
        ) :
            Cpos(1),
            Cneg(1),
            eps(0.1),
            max_iterations(10000),
            verbose(false),
            have_bias(true),
            last_weight_1(false),
            do_shrinking(true)
        {
        }

        explicit svm_c_linear_dcd_trainer (
            const scalar_type& C_
        ) :
            Cpos(C_),
            Cneg(C_),
            eps(0.1),
            max_iterations(10000),
            verbose(false),
            do_shrinking(true)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(0 < C_,
                "\tsvm_c_trainer::svm_c_linear_dcd_trainer(kernel,C)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t C_: " << C_
                );
        }

        bool includes_bias (
        ) const 
        { 
            return have_bias; 
        }

        void include_bias (
            bool should_have_bias
        ) 
        { 
            have_bias = should_have_bias; 
        }

        bool forces_last_weight_to_1 (
        ) const
        {
            return last_weight_1;
        }

        void force_last_weight_to_1 (
            bool should_last_weight_be_1
        )
        {
            last_weight_1 = should_last_weight_be_1;
        }

        bool shrinking_enabled (
        ) const { return do_shrinking; }

        void enable_shrinking (
            bool enabled
        ) { do_shrinking = enabled; }

        void be_verbose (
        )
        {
            verbose = true;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }

        void set_epsilon (
            scalar_type eps_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\tvoid svm_c_linear_dcd_trainer::set_epsilon(eps_)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t eps_: " << eps_ 
                );
            eps = eps_;
        }

        const scalar_type get_epsilon (
        ) const
        { 
            return eps;
        }

        const kernel_type& get_kernel (
        ) const
        {
            return kernel_type();
        }

        unsigned long get_max_iterations (
        ) const { return max_iterations; }

        void set_max_iterations (
            unsigned long max_iter
        ) 
        {
            max_iterations = max_iter;
        }

        void set_c (
            scalar_type C 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_linear_dcd_trainer::set_c()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

            Cpos = C;
            Cneg = C;
        }

        const scalar_type get_c_class1 (
        ) const
        {
            return Cpos;
        }

        const scalar_type get_c_class2 (
        ) const
        {
            return Cneg;
        }

        void set_c_class1 (
            scalar_type C
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_linear_dcd_trainer::set_c_class1()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

            Cpos = C;
        }

        void set_c_class2 (
            scalar_type C
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(C > 0,
                "\t void svm_c_linear_dcd_trainer::set_c_class2()"
                << "\n\t C must be greater than 0"
                << "\n\t C:    " << C 
                << "\n\t this: " << this
                );

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
        {
            scalar_vector_type alpha(x.size());
            alpha = 0;
            return do_train(vector_to_matrix(x), vector_to_matrix(y), alpha);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_vector_type& alpha
        ) const
        {
            DLIB_CASSERT (static_cast<long>(x.size()) >= alpha.size(), 
                "\t decision_function svm_c_linear_dcd_trainer::train(x,y,alpha)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t x.size():     " << x.size() 
                << "\n\t alpha.size(): " << alpha.size() 
                );

            if (static_cast<long>(x.size()) > alpha.size())
            {
                // Make sure alpha has the same length as x.  So pad with extra zeros if
                // necessary to make this happen.
                alpha = join_cols(alpha, zeros_matrix<scalar_type>(1,x.size()-alpha.size()));
            }

            return do_train(vector_to_matrix(x), vector_to_matrix(y), alpha);
        }

    private:

    // ------------------------------------------------------------------------------------

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_vector_type& alpha
        ) const
        {
            // TODO, requires labels are all +1 or -1.  But we don't have to see both
            // types.

            // make sure requires clause is not broken
            DLIB_ASSERT(is_learning_problem(x,y) == true,
                "\t decision_function svm_c_linear_dcd_trainer::train(x,y)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t x.size(): " << x.size() 
                << "\n\t y.size(): " << y.size() 
                << "\n\t is_learning_problem(x,y): " << is_learning_problem(x,y)
                );

            const long dims = max_index_plus_one(x);

            // TODO, return an opaque object instead of alpha.  Also, the object
            // needs to verify that the trainer has the same settings from one
            // call to the next.

            std::vector<long> index(x.size());
            scalar_vector_type Q(x.size());

            scalar_vector_type w;
            if (have_bias)
                w.set_size(dims+1);
            else
                w.set_size(dims);

            w = 0;
            if (last_weight_1)
                w(dims-1) = 1;

            long ii = 0;
            for (long i = 0; i < alpha.size(); ++i)
            {
                index[ii] = i;
                Q(ii) = dlib::dot(x(i),x(i));

                if (have_bias)
                {
                    Q(ii) += 1;
                    ++ii;
                }
                else if (Q(ii) != 0) 
                {
                    ++ii;
                }
            }

            // What we are doing here is ignoring x elements that have 0 norm.  We
            // Do this because they are impossible to classify and this also avoids
            // a division by zero problem later on in the code.
            const long max_possible_active = ii;

            dlib::rand rnd;
            long active_size = max_possible_active;

            scalar_type PG_max_prev = std::numeric_limits<scalar_type>::infinity();
            scalar_type PG_min_prev = -std::numeric_limits<scalar_type>::infinity();

            // main loop
            for (unsigned long iter = 0; iter < max_iterations; ++iter)
            {
                scalar_type PG_max = -std::numeric_limits<scalar_type>::infinity();
                scalar_type PG_min = std::numeric_limits<scalar_type>::infinity();

                // randomly shuffle the indices
                for (long i = 0; i < active_size; ++i)
                {
                    // pick a random index >= i
                    const long j = i + rnd.get_random_32bit_number()%(active_size-i);
                    std::swap(index[i], index[j]);
                }
                
                // for all the active training samples
                for (long ii = 0; ii < active_size; ++ii)
                {
                    const long i = index[ii];

                    const scalar_type G = y(i)*dot(w, x(i)) - 1;
                    const scalar_type C = (y(i) > 0) ? Cpos : Cneg;

                    scalar_type PG = 0;
                    if (alpha(i) == 0)
                    {
                        if (G > PG_max_prev)
                        {
                            // shrink the active set of training examples
                            --active_size;
                            std::swap(index[ii], index[active_size]);
                            --ii;
                            continue;
                        }

                        if (G < 0)
                            PG = G;
                    }
                    else if (alpha(i) == C)
                    {
                        if (G < PG_min_prev)
                        {
                            // shrink the active set of training examples
                            --active_size;
                            std::swap(index[ii], index[active_size]);
                            --ii;
                            continue;
                        }

                        if (G > 0)
                            PG = G;
                    }
                    else
                    {
                        PG = G;
                    }

                    if (PG > PG_max) 
                        PG_max = PG;
                    if (PG < PG_min) 
                        PG_min = PG;

                    // if PG != 0
                    if (std::abs(PG) > 1e-12)
                    {
                        const scalar_type alpha_old = alpha(i);
                        alpha(i) = std::min(std::max(alpha(i) - G/Q(i), (scalar_type)0.0), C);
                        const scalar_type delta = (alpha(i)-alpha_old)*y(i);
                        add_to(w, x(i), delta);
                        if (have_bias)
                            w(w.size()-1) -= delta;

                        if (last_weight_1)
                            w(dims-1) = 1;
                    }

                }

                if (verbose)
                {
                    using namespace std;
                    cout << "gap:         " << PG_max - PG_min << endl;
                    cout << "active_size: " << active_size << endl;
                    cout << "iter:        " << iter << endl;
                    cout << endl;
                }

                if (PG_max - PG_min <= eps)
                {
                    // stop if we are within eps tolerance and the last iteration
                    // was over all the samples
                    if (active_size == max_possible_active)
                        break;

                    // Turn of shrinking on the next iteration.  We will stop if the
                    // tolerance is still <= eps when shrinking is off.
                    active_size = max_possible_active;
                    PG_max_prev = std::numeric_limits<scalar_type>::infinity();
                    PG_min_prev = -std::numeric_limits<scalar_type>::infinity();
                }
                else if (do_shrinking)
                {
                    PG_max_prev = PG_max;
                    PG_min_prev = PG_min;
                    if (PG_max_prev <= 0)
                        PG_max_prev = std::numeric_limits<scalar_type>::infinity();
                    if (PG_min_prev >= 0)
                        PG_min_prev = -std::numeric_limits<scalar_type>::infinity();
                }
            }

            // put the solution into a decision function and then return it
            decision_function<kernel_type> df;
            if (have_bias)
                df.b = w(w.size()-1);
            else
                df.b = 0;

            df.basis_vectors.set_size(1);
            // Copy the plane normal into the output basis vector.  The output vector might be a
            // sparse vector container so we need to use this special kind of copy to handle that case.
            // As an aside, the reason for using max_index_plus_one() and not just w.size()-1 is because
            // doing it this way avoids an inane warning from gcc that can occur in some cases.
            assign(df.basis_vectors(0), colm(w, 0, dims));
            df.alpha.set_size(1);
            df.alpha(0) = 1;

            return df;
        }

        scalar_type dot (
            const scalar_vector_type& w,
            const sample_type& sample
        ) const
        {
            if (have_bias)
            {
                const long w_size_m1 = w.size()-1;
                return dlib::dot(colm(w,0,w_size_m1), sample) - w(w_size_m1);
            }
            else
            {
                return dlib::dot(w, sample);
            }
        }

    // ------------------------------------------------------------------------------------

        scalar_type Cpos;
        scalar_type Cneg;
        scalar_type eps;
        unsigned long max_iterations;
        bool verbose;
        bool have_bias; // having a bias means we pretend all x vectors have an extra element which is always -1.
        bool last_weight_1;
        bool do_shrinking;

    }; // end of class svm_c_linear_dcd_trainer

// ----------------------------------------------------------------------------------------


}

#endif // DLIB_SVm_C_LINEAR_DCD_TRAINER_H__


