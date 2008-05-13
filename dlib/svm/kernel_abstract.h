// Copyright (C) 2007  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_KERNEL_ABSTRACT_
#ifdef DLIB_SVm_KERNEL_ABSTRACT_

#include <cmath>
#include <limits>
#include <sstream>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
/*!A                               Kernel_Function_Objects                               */
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    /*! 
        WHAT IS A KERNEL FUNCTION OBJECT?
            In the context of the dlib library documentation a kernel function object
            is an object with an interface with the following properties:
                - a public typedef named sample_type
                - a public typedef named scalar_type which should be a float, double, or 
                  long double type.
                - an overloaded operator() that operates on two items of sample_type 
                  and returns a scalar_type.  
                  (e.g. scalar_type val = kernel_function(sample(i),sample(j)); 
                   would be a valid expression)
                - a public typedef named mem_manager_type that is an implementation of 
                  dlib/memory_manager/memory_manager_kernel_abstract.h or
                  dlib/memory_manager_global/memory_manager_global_kernel_abstract.h or
                  dlib/memory_manager_stateless/memory_manager_stateless_kernel_abstract.h 

        For examples of kernel functions see the following objects
        (e.g. the radial_basis_kernel).
    !*/

    template <
        typename T
        >
    struct radial_basis_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a radial basis function kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        const scalar_type gamma;

        radial_basis_kernel(
        );
        /*!
            ensures
                - #gamma == 0.1 
        !*/

        radial_basis_kernel(
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma == k.gamma
        !*/

        radial_basis_kernel(
            const scalar_type g
        );
        /*!
            ensures
                - #gamma == g
        !*/

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns exp(-gamma * ||a-b||^2)
        !*/

        radial_basis_kernel& operator= (
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma = k.gamma
                - returns *this
        !*/

    };

    template <
        typename T
        >
    void serialize (
        const radial_basis_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for radial_basis_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        radial_basis_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for radial_basis_kernel
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct polynomial_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a polynomial kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        const scalar_type gamma;
        const scalar_type coef;
        const scalar_type degree;

        polynomial_kernel(
        );
        /*!
            ensures
                - #gamma == 1 
                - #coef == 0 
                - #degree == 1 
        !*/

        polynomial_kernel(
            const radial_basis_kernel& k
        );
        /*!
            ensures
                - #gamma == k.gamma
        !*/

        polynomial_kernel(
            const scalar_type g,
            const scalar_type c,
            const scalar_type d
        );
        /*!
            ensures
                - #gamma == g
                - #coef == c
                - #degree == d
        !*/

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns pow(gamma*trans(a)*b + coef, degree)
        !*/

        polynomial_kernel& operator= (
            const polynomial_kernel& k
        );
        /*!
            ensures
                - #gamma = k.gamma
                - #coef = k.coef
                - #degree = k.degree
                - returns *this
        !*/

    };

    template <
        typename T
        >
    void serialize (
        const polynomial_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for polynomial_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        polynomial_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for polynomial_kernel
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct linear_kernel
    {
        /*!
            REQUIREMENTS ON T
                T must be a dlib::matrix object 

            WHAT THIS OBJECT REPRESENTS
                This object represents a linear function kernel
        !*/

        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const;
        /*!
            requires
                - a.nc() == 1
                - b.nc() == 1
                - a.nr() == b.nr()
            ensures
                - returns trans(a)*b
        !*/
    };

    template <
        typename T
        >
    void serialize (
        const linear_kernel<T>& item,
        std::ostream& out
    );
    /*!
        provides serialization support for linear_kernel
    !*/

    template <
        typename K
        >
    void deserialize (
        linear_kernel<T>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support for linear_kernel 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_KERNEL_ABSTRACT_



