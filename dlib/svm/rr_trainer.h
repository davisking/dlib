// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KRR_TRAInER_H__
#define DLIB_KRR_TRAInER_H__

#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "empirical_kernel_map.h"
#include "linearly_independent_subset_finder.h"
#include "../statistics.h"
#include "krr_trainer_abstract.h"
#include <vector>
#include <iostream>

namespace dlib
{
    template <
        typename K 
        >
    class krr_trainer
    {

    public:
        typedef K kernel_type;
        typedef typename kernel_type::scalar_type scalar_type;
        typedef typename kernel_type::sample_type sample_type;
        typedef typename kernel_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type> trained_function_type;

        krr_trainer (
        ) :
            verbose(false),
            use_regression_loss(true),
            lambda(0),
            max_basis_size(400),
            ekm_stale(true)
        {
            // default lambda search list
            lams = matrix_cast<scalar_type>(logspace(-9, 2, 50)); 
        }

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

        void use_regression_loss_for_loo_cv (
        )
        {
            use_regression_loss = true;
        }

        void use_classification_loss_for_loo_cv (
        )
        {
            use_regression_loss = false;
        }

        bool will_use_regression_loss_for_loo_cv (
        ) const
        {
            return use_regression_loss;
        }

        const kernel_type get_kernel (
        ) const
        {
            return kern;
        }

        void set_kernel (
            const kernel_type& k
        )
        {
            kern = k;
        }

        template <typename T>
        void set_basis (
            const T& basis_samples
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(basis_samples.size() > 0 && is_vector(vector_to_matrix(basis_samples)),
                "\tvoid krr_trainer::set_basis(basis_samples)"
                << "\n\t You have to give a non-empty set of basis_samples and it must be a vector"
                << "\n\t basis_samples.size():                       " << basis_samples.size() 
                << "\n\t is_vector(vector_to_matrix(basis_samples)): " << is_vector(vector_to_matrix(basis_samples)) 
                << "\n\t this: " << this
                );

            basis = vector_to_matrix(basis_samples);
            ekm_stale = true;
        }

        bool basis_loaded (
        ) const
        {
            return (basis.size() != 0);
        }

        void clear_basis (
        )
        {
            basis.set_size(0);
            ekm.clear();
            ekm_stale = true;
        }

        unsigned long get_max_basis_size (
        ) const
        {
            return max_basis_size;
        }

        void set_max_basis_size (
            unsigned long max_basis_size_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(max_basis_size_ > 0,
                "\t void krr_trainer::set_max_basis_size()"
                << "\n\t max_basis_size_ must be greater than 0"
                << "\n\t max_basis_size_: " << max_basis_size_ 
                << "\n\t this:            " << this
                );

            max_basis_size = max_basis_size_;
        }

        void set_lambda (
            scalar_type lambda_ 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(lambda_ >= 0,
                "\t void krr_trainer::set_lambda()"
                << "\n\t lambda must be greater than or equal to 0"
                << "\n\t lambda: " << lambda 
                << "\n\t this:   " << this
                );

            lambda = lambda_;
        }

        const scalar_type get_lambda (
        ) const
        {
            return lambda;
        }

        template <typename EXP>
        void set_search_lambdas (
            const matrix_exp<EXP>& lambdas
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_vector(lambdas) && lambdas.size() > 0 && min(lambdas) > 0,
                "\t void krr_trainer::set_search_lambdas()"
                << "\n\t lambdas must be a non-empty vector of values"
                << "\n\t is_vector(lambdas): " << is_vector(lambdas) 
                << "\n\t lambdas.size():     " << lambdas.size()
                << "\n\t min(lambdas):       " << min(lambdas) 
                << "\n\t this:   " << this
                );


            lams = matrix_cast<scalar_type>(lambdas);
        }

        const matrix<scalar_type,0,0,mem_manager_type>& get_search_lambdas (
        ) const
        {
            return lams;
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
            scalar_type temp, temp2;
            return do_train(vector_to_matrix(x), vector_to_matrix(y), false, temp, temp2);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& looe
        ) const
        {
            scalar_type temp;
            return do_train(vector_to_matrix(x), vector_to_matrix(y), true, looe, temp);
        }

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            scalar_type& looe,
            scalar_type& lambda_used 
        ) const
        {
            return do_train(vector_to_matrix(x), vector_to_matrix(y), true, looe, lambda_used);
        }


    private:

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y,
            bool output_looe,
            scalar_type& best_looe,
            scalar_type& the_lambda
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(is_learning_problem(x,y),
                "\t decision_function krr_trainer::train(x,y)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t is_vector(x): " << is_vector(x)
                << "\n\t is_vector(y): " << is_vector(y)
                << "\n\t x.size():     " << x.size() 
                << "\n\t y.size():     " << y.size() 
                );

#ifdef ENABLE_ASSERTS
            if (get_lambda() == 0 && will_use_regression_loss_for_loo_cv() == false)
            {
                // make sure requires clause is not broken
                DLIB_ASSERT(is_binary_classification_problem(x,y),
                    "\t decision_function krr_trainer::train(x,y)"
                    << "\n\t invalid inputs were given to this function"
                    );
            }
#endif

            // The first thing we do is make sure we have an appropriate ekm ready for use below.
            if (basis_loaded())
            {
                if (ekm_stale)
                {
                    ekm.load(kern, basis);
                    ekm_stale = false;
                }
            }
            else
            {
                linearly_independent_subset_finder<kernel_type> lisf(kern, max_basis_size);
                fill_lisf(lisf, x);
                ekm.load(lisf);
            }

            if (verbose)
            {
                std::cout << "\nNumber of basis vectors used: " << ekm.out_vector_size() << std::endl;
            }

            typedef matrix<scalar_type,0,1,mem_manager_type> column_matrix_type;
            typedef matrix<scalar_type,0,0,mem_manager_type> general_matrix_type;

            running_stats<scalar_type> rs;

            // Now we project all the x samples into kernel space using our EKM 
            matrix<column_matrix_type,0,1,mem_manager_type > proj_x;
            proj_x.set_size(x.size());
            for (long i = 0; i < proj_x.size(); ++i)
            {
                scalar_type err;
                // Note that we also append a 1 to the end of the vectors because this is
                // a convenient way of dealing with the bias term later on.
                if (verbose == false)
                {
                    proj_x(i) = ekm.project(x(i));
                }
                else
                {
                    proj_x(i) = ekm.project(x(i),err);
                    rs.add(err);
                }
            }

            const long dims = ekm.out_vector_size();

            if (verbose)
            {
                std::cout << "Mean EKM projection error:                  " << rs.mean() << std::endl;
                std::cout << "Standard deviation of EKM projection error: " << rs.stddev() << std::endl;
            }

            /*
                Notes on the solution of KRR

                Let A = an proj_x.size() by ekm.out_vector_size() matrix which contains
                all the projected data samples.

                Let I = an identity matrix

                Let C = trans(A)*A
                Let L = trans(A)*y

                Then the optimal w is given by:
                    w = inv(C + lambda*I) * L 


                There is a trick to compute leave one out cross validation results for many different
                lambda values quickly.  The following paper has a detailed discussion of various
                approaches:

                    Notes on Regularized Least Squares by Ryan M. Rifkin and Ross A. Lippert.

                    In the implementation of the krr_trainer I'm only using two simple equations
                    from the above paper.


                    First note that inv(C + lambda*I) can be computed for many different lambda
                    values in an efficient way by using an eigen decomposition of C.  So we use
                    the fact that:
                        inv(C + lambda*I) == V*inv(D + lambda*I)*trans(V)
                        where V*D*trans(V) == C 

                    Also, via some simple linear algebra the above paper works out that the leave one out 
                    value for a sample x(i) is equal to the following (we refer to proj_x(i) as x(i) for brevity):
                        Let G = inv(C + lambda*I)
                        let val = trans(x(i))*G*x(i);

                        leave one out value for sample x(i):
                        LOOV = (trans(w)*x(i) - y(i)*val) / (1 - val)

                        leave one out error for sample x(i):
                        LOOE = loss(y(i), LOOV)


                Finally, note that we will pretend there was a 1 appended to the end of each
                vector in proj_x.  We won't actually do that though because we don't want to
                have to make a copy of all the samples.  So throughout the following code 
                I have explicitly dealt with this.
            */

            general_matrix_type C, tempm, G;
            column_matrix_type  L, tempv, w;

            // compute C and L
            for (long i = 0; i < proj_x.size(); ++i)
            {
                C += proj_x(i)*trans(proj_x(i));
                L += y(i)*proj_x(i);
                tempv += proj_x(i);
            }

            // Make C = [C      tempv
            //           tempv' proj_x.size()]
            C = join_cols(join_rows(C, tempv), 
                          join_rows(trans(tempv), uniform_matrix<scalar_type>(1,1, proj_x.size())));
            L = join_cols(L, uniform_matrix<scalar_type>(1,1, sum(y)));

            eigenvalue_decomposition<general_matrix_type> eig(make_symmetric(C));
            const general_matrix_type V = eig.get_pseudo_v();
            const column_matrix_type  D = eig.get_real_eigenvalues();

            // We can save some work by pre-multiplying the proj_x vectors by trans(V)
            // and saving the result so we don't have to recompute it over and over later.
            matrix<column_matrix_type,0,1,mem_manager_type > Vx;
            if (lambda == 0 || output_looe)
            {
                // Save the transpose of V into a temporary because the subsequent matrix
                // vector multiplies will be faster (because of better cache locality).
                const general_matrix_type transV( colm(trans(V),range(0,dims-1))  );
                // Remember the pretend 1 at the end of proj_x(*).  We want to multiply trans(V)*proj_x(*)
                // so to do this we pull the last column off trans(V) and store it separately.
                const column_matrix_type lastV = colm(trans(V), dims);
                Vx.set_size(proj_x.size());
                for (long i = 0; i < proj_x.size(); ++i)
                {
                    Vx(i) = transV*proj_x(i);
                    Vx(i) = squared(Vx(i) + lastV);
                }
            }

            the_lambda = lambda;

            // If we need to automatically select a lambda then do so using the LOOE trick described
            // above.
            if (lambda == 0)
            {
                best_looe = std::numeric_limits<scalar_type>::max();

                // Compute leave one out errors for a bunch of different lambdas and pick the best one.
                for (long idx = 0; idx < lams.size(); ++idx)
                {
                    // first compute G
                    tempv = 1.0/(D + lams(idx));
                    tempm = scale_columns(V,tempv);
                    G = tempm*trans(V);

                    // compute the solution w for the current lambda
                    w = G*L;

                    // make w have the same length as the x_proj vectors.
                    const scalar_type b = w(dims);
                    w = colm(w,0,dims);

                    scalar_type looe = 0;
                    for (long i = 0; i < proj_x.size(); ++i)
                    {
                        // perform equivalent of: val = trans(proj_x(i))*G*proj_x(i);
                        const scalar_type val = dot(tempv, Vx(i));
                        const scalar_type temp = (1 - val);
                        scalar_type loov;
                        if (temp != 0)
                            loov = (trans(w)*proj_x(i) + b - y(i)*val) / temp;
                        else
                            loov = 0;

                        looe += loss(loov, y(i));
                    }

                    // Keep track of the lambda which gave the lowest looe.  If two lambdas
                    // have the same looe then pick the biggest lambda.
                    if (looe < best_looe || (looe == best_looe && lams(idx) > the_lambda))
                    {
                        best_looe = looe;
                        the_lambda = lams(idx);
                    }
                }

                // mark that we saved the looe to best_looe already
                output_looe = false;
                best_looe /= proj_x.size();

                if (verbose)
                {
                    using namespace std;
                    cout << "Using lambda: " << the_lambda << endl;
                    cout << "LOO Error:    " << best_looe << endl;
                }
            }



            // Now perform the main training.  That is, find w.
            // first, compute G = inv(C + the_lambda*I)
            tempv = 1.0/(D + the_lambda);
            tempm = scale_columns(V,tempv);
            G = tempm*trans(V);
            w = G*L;
           
            // make w have the same length as the x_proj vectors.
            const scalar_type b = w(dims);
            w = colm(w,0,dims);


            // If we haven't done this already and we are supposed to then compute the LOO error rate for 
            // the current lambda and store the result in best_looe.
            if (output_looe)
            {
                best_looe = 0;
                for (long i = 0; i < proj_x.size(); ++i)
                {
                    // perform equivalent of: val = trans(proj_x(i))*G*proj_x(i);
                    const scalar_type val = dot(tempv, Vx(i));
                    const scalar_type temp = (1 - val);
                    scalar_type loov;
                    if (temp != 0)
                        loov = (trans(w)*proj_x(i) + b - y(i)*val) / temp;
                    else
                        loov = 0;

                    best_looe += loss(loov, y(i));
                }

                best_looe /= proj_x.size();

                if (verbose)
                {
                    using namespace std;
                    cout << "Using lambda: " << the_lambda << endl;
                    cout << "LOO Error:    " << best_looe << endl;
                }
            }


            // convert w into a proper decision function
            decision_function<kernel_type> df;
            df = ekm.convert_to_decision_function(w);
            df.b = -b; // don't forget about the bias we stuck onto all the vectors

            // If we used an automatically derived basis then there isn't any point in
            // keeping the ekm around.  So free its memory.
            if (basis_loaded() == false)
            {
                ekm.clear();
            }

            return df;
        }

        inline scalar_type loss (
            const scalar_type& a,
            const scalar_type& b
        ) const
        {
            if (use_regression_loss)
            {
                return (a-b)*(a-b);
            }
            else
            {
                // if a and b have the same sign then no loss
                if (a*b >= 0)
                    return 0;
                else
                    return 1;
            }
        }


        /*!
            CONVENTION
                - if (ekm_stale) then
                    - kern or basis have changed since the last time
                      they were loaded into the ekm

                - get_lambda() == lambda
                - get_kernel() == kern
                - get_max_basis_size() == max_basis_size
                - will_use_regression_loss_for_loo_cv() == use_regression_loss
                - get_search_lambdas() == lams

                - basis_loaded() == (basis.size() != 0)
        !*/

        bool verbose;
        bool use_regression_loss;

        scalar_type lambda;

        kernel_type kern;
        unsigned long max_basis_size;

        matrix<sample_type,0,1,mem_manager_type> basis;
        mutable empirical_kernel_map<kernel_type> ekm;
        mutable bool ekm_stale; 

        matrix<scalar_type,0,0,mem_manager_type> lams; 
    }; 

}

#endif // DLIB_KRR_TRAInER_H__


