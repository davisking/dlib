// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RBf_NETWORK_
#define DLIB_RBf_NETWORK_

#include "../matrix.h"
#include "rbf_network_abstract.h"
#include "kernel.h"
#include "kcentroid.h"
#include "function.h"
#include "../algs.h"

namespace dlib
{

// ------------------------------------------------------------------------------

    template <
        typename sample_type_
        >
    class rbf_network_trainer 
    {
        /*!
            This is an implemenation of an RBF network trainer that follows
            the directions right off Wikipedia basically.  So nothing 
            particularly fancy.
        !*/

    public:
        typedef radial_basis_kernel<sample_type_>      kernel_type;
        typedef sample_type_                           sample_type;
        typedef typename sample_type::type             scalar_type;
        typedef typename sample_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type>         trained_function_type;

        rbf_network_trainer (
        ) :
            gamma(0.1),
            tolerance(0.01)
        {
        }

        void set_gamma (
            scalar_type gamma_ 
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(gamma_ > 0,
                "\tvoid rbf_network_trainer::set_gamma(gamma_)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t gamma: " << gamma_
                );
            gamma = gamma_;
        }

        const scalar_type get_gamma (
        ) const
        { 
            return gamma;
        }

        void set_tolerance (
            const scalar_type& tol 
        )
        {
            tolerance = tol;
        }

        const scalar_type& get_tolerance (
        ) const
        {
            return tolerance;
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
            return do_train(vector_to_matrix(x), vector_to_matrix(y));
        }

        void swap (
            rbf_network_trainer& item
        )
        {
            exchange(gamma, item.gamma);
            exchange(tolerance, item.tolerance);
        }

    private:

    // ------------------------------------------------------------------------------------

        template <
            typename in_sample_vector_type,
            typename in_scalar_vector_type
            >
        const decision_function<kernel_type> do_train (
            const in_sample_vector_type& x,
            const in_scalar_vector_type& y
        ) const
        {
            typedef typename decision_function<kernel_type>::scalar_vector_type scalar_vector_type;

            // make sure requires clause is not broken
            DLIB_ASSERT(x.nr() > 1 && x.nr() == y.nr() && x.nc() == 1 && y.nc() == 1,
                "\tdecision_function rbf_network_trainer::train(x,y)"
                << "\n\t invalid inputs were given to this function"
                << "\n\t x.nr(): " << x.nr() 
                << "\n\t y.nr(): " << y.nr() 
                << "\n\t x.nc(): " << x.nc() 
                << "\n\t y.nc(): " << y.nc() 
                );

            // first run all the sampes through a kcentroid object to find the rbf centers
            const kernel_type kernel(gamma);
            kcentroid<kernel_type> kc(kernel,tolerance);
            for (long i = 0; i < x.size(); ++i)
            {
                kc.train(x(i));
            }

            // now we have a trained kcentroid so lets just extract its results.  Note that
            // all we want out of the kcentroid is really just the set of support vectors
            // it contains so that we can use them as the RBF centers.
            distance_function<kernel_type> df(kc.get_distance_function());
            const long num_centers = df.support_vectors.nr();

            // fill the K matrix with the output of the kernel for all the center and sample point pairs
            matrix<scalar_type,0,0,mem_manager_type> K(x.nr(), num_centers+1);
            for (long r = 0; r < x.nr(); ++r)
            {
                for (long c = 0; c < num_centers; ++c)
                {
                    K(r,c) = kernel(x(r), df.support_vectors(c));
                }
                // This last column of the K matrix takes care of the bias term
                K(r,num_centers) = 1;
            }

            // compute the best weights by using the pseudo-inverse
            scalar_vector_type weights(pinv(K)*y);

            // now put everything into a decision_function object and return it
            return decision_function<kernel_type> (remove_row(weights,num_centers),
                                                   -weights(num_centers),
                                                   kernel,
                                                   df.support_vectors);

        }

        scalar_type gamma;
        scalar_type tolerance;

    }; // end of class rbf_network_trainer 

// ----------------------------------------------------------------------------------------

    template <typename sample_type>
    void swap (
        rbf_network_trainer<sample_type>& a,
        rbf_network_trainer<sample_type>& b
    ) { a.swap(b); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RBf_NETWORK_

