// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_KRR_TRAInER_H__
#define DLIB_KRR_TRAInER_H__

#include "../algs.h"
#include "function.h"
#include "kernel.h"
#include "empirical_kernel_map.h"
#include "linearly_independent_subset_finder.h"
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
            lams = logspace(-9, 2, 40); 
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

        void estimate_lambda_for_regression (
        )
        {
            use_regression_loss = true;
        }

        void estimate_lambda_for_classification (
        )
        {
            use_regression_loss = false;
        }

        bool will_estimate_lambda_for_regression (
        )
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
            DLIB_ASSERT(is_vector(x) && is_vector(y) && x.size() == y.size() && x.size() > 0,
                "\t decision_function krr_trainer::train(x,y)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t is_vector(x): " << is_vector(x)
                << "\n\t is_vector(y): " << is_vector(y)
                << "\n\t x.size():     " << x.size() 
                << "\n\t y.size():     " << y.size() 
                );

#ifdef ENABLE_ASSERTS
            if (get_lambda() == 0 && will_estimate_lambda_for_regression() == false)
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
                std::cout << "Number of basis vectors used: " << ekm.out_vector_size() << std::endl;
            }

            typedef matrix<scalar_type,0,1,mem_manager_type> column_matrix_type;
            typedef matrix<scalar_type,0,0,mem_manager_type> general_matrix_type;

            // Now we project all the x samples into kernel space using our EKM 
            matrix<column_matrix_type,0,1,mem_manager_type > proj_x;
            proj_x.set_size(x.size());
            for (long i = 0; i < proj_x.size(); ++i)
            {
                // Note that we also append a 1 to the end of the vectors because this is
                // a convenient way of dealing with the bias term later on.
                proj_x(i) = join_cols(ekm.project(x(i)), ones_matrix<scalar_type>(1,1));
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
            */

            general_matrix_type C, tempm, G;
            column_matrix_type  L, tempv, w;

            // compute C and L
            for (long i = 0; i < proj_x.size(); ++i)
            {
                C += proj_x(i)*trans(proj_x(i));
                L += y(i)*proj_x(i);
            }

            eigenvalue_decomposition<general_matrix_type> eig(C);
            const general_matrix_type V = eig.get_pseudo_v();
            const column_matrix_type  D = eig.get_real_eigenvalues();


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
                    tempv = reciprocal(D + uniform_matrix<scalar_type>(D.nr(),D.nc(), lams(idx)));
                    tempm = scale_columns(V,tempv);
                    G = tempm*trans(V);

                    // compute the solution w for the current lambda
                    w = G*L;

                    scalar_type looe = 0;
                    for (long i = 0; i < proj_x.size(); ++i)
                    {
                        const scalar_type val = trans(proj_x(i))*G*proj_x(i);
                        const scalar_type temp = (1 - val);
                        scalar_type loov;
                        if (temp != 0)
                            loov = (trans(w)*proj_x(i) - y(i)*val) / temp;
                        else
                            loov = 0;

                        looe += loss(loov, y(i));
                    }

                    if (looe < best_looe)
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
            tempv = reciprocal(D + uniform_matrix<scalar_type>(D.nr(),D.nc(), the_lambda));
            tempm = scale_columns(V,tempv);
            G = tempm*trans(V);
            w = G*L;
           

            // If we haven't done this already and we are supposed to then compute the LOO error rate for 
            // the current lambda and store the result in best_looe.
            if (output_looe)
            {
                best_looe = 0;
                for (long i = 0; i < proj_x.size(); ++i)
                {
                    const scalar_type val = trans(proj_x(i))*G*proj_x(i);
                    const scalar_type temp = (1 - val);
                    scalar_type loov;
                    if (temp != 0)
                        loov = (trans(w)*proj_x(i) - y(i)*val) / temp;
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
            df = ekm.convert_to_decision_function(colm(w,0,w.size()-1));
            df.b = -w(w.size()-1); // don't forget about the bias we stuck onto all the vectors

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
                - will_estimate_lambda_for_regression() == use_regression_loss
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


