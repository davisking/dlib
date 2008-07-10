// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RBf_NETWORK_ABSTRACT_
#ifdef DLIB_RBf_NETWORK_ABSTRACT_

#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "function_abstract.h"
#include "kernel_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type_
        >
    class rbf_network_trainer 
    {
        /*!
            REQUIREMENTS ON sample_type_
                is a dlib::matrix type  

            INITIAL VALUE
                - get_gamma() == 0.1
                - get_tolerance() == 0.01

            WHAT THIS OBJECT REPRESENTS
                This object implements a trainer for an radial basis function network.

                The implementation of this algorithm follows the normal RBF training 
                process.  For more details see the code or the Wikipedia article
                about RBF networks.  
        !*/
    public:
        typedef radial_basis_kernel<sample_type_>      kernel_type;
        typedef          sample_type_                  sample_type;
        typedef typename sample_type::type             scalar_type;
        typedef typename sample_type::mem_manager_type mem_manager_type;
        typedef decision_function<kernel_type>         trained_function_type;

        rbf_network_trainer (
        ); 
        /*!
            ensures
                - this object is properly initialized
        !*/

        void set_gamma (
            scalar_type gamma
        );
        /*!
            requires
                - gamma > 0
            ensures
                - #get_gamma() == gamma
        !*/

        const scalar_type get_gamma (
        ) const
        /*!
            ensures
                - returns the gamma argument used in the radial_basis_kernel used
                  to represent each node in an RBF network.
        !*/

        void set_tolerance (
            const scalar_type& tol 
        );
        /*!
            ensures
                - #get_tolerance() == tol
        !*/

        const scalar_type& get_tolerance (
        ) const;
        /*!
            ensures
                - returns the tolerance parameter.  This parameter controls how many
                  RBF centers (a.k.a. support_vectors in the trained decision_function)
                  you get when you call the train function.  A smaller tolerance
                  results in more centers while a bigger number results in fewer.
        !*/

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
                - x == a matrix or something convertable to a matrix via vector_to_matrix().
                  Also, x should contain sample_type objects.
                - y == a matrix or something convertable to a matrix via vector_to_matrix().
                  Also, y should contain scalar_type objects.
                - x.nr() > 1
                - x.nr() == y.nr() && x.nc() == 1 && y.nc() == 1 
                  (i.e. x and y are both column vectors of the same length)
            ensures
                - trains a RBF network given the training samples in x and 
                  labels in y.  
                - returns a decision function F with the following properties:
                    - if (new_x is a sample predicted have +1 label) then
                        - F(new_x) >= 0
                    - else
                        - F(new_x) < 0
            throws
                - std::bad_alloc
        !*/

        void swap (
            rbf_network_trainer& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/

    }; 

// ----------------------------------------------------------------------------------------

    template <typename sample_type>
    void swap (
        rbf_network_trainer<sample_type>& a,
        rbf_network_trainer<sample_type>& b
    ) { a.swap(b); }
    /*!
        provides a global swap
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RBf_NETWORK_ABSTRACT_



