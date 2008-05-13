// Copyright (C) 2007  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SVm_KERNEL
#define DLIB_SVm_KERNEL

#include "kernel_abstract.h"
#include <cmath>
#include <limits>
#include <sstream>
#include "../matrix.h"
#include "../algs.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct radial_basis_kernel
    {
        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        radial_basis_kernel(const scalar_type g) : gamma(g) {}
        radial_basis_kernel() : gamma(0.1) {}
        radial_basis_kernel(
            const radial_basis_kernel& k
        ) : gamma(k.gamma) {}


        const scalar_type gamma;

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const
        { 
            const scalar_type d = trans(a-b)*(a-b);
            return std::exp(-gamma*d);
        }

        radial_basis_kernel& operator= (
            const radial_basis_kernel& k
        )
        {
            const_cast<scalar_type&>(gamma) = k.gamma;
            return *this;
        }
    };

    template <
        typename T
        >
    void serialize (
        const radial_basis_kernel<T>& item,
        std::ostream& out
    )
    {
        try
        {
            serialize(item.gamma, out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type radial_basis_kernel"); 
        }
    }

    template <
        typename T
        >
    void deserialize (
        radial_basis_kernel<T>& item,
        std::istream& in 
    )
    {
        typedef typename T::type scalar_type;
        try
        {
            deserialize(const_cast<scalar_type&>(item.gamma), in);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while deserializing object of type radial_basis_kernel"); 
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    struct polynomial_kernel
    {
        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        polynomial_kernel(const scalar_type g, const scalar_type c, const scalar_type d) : gamma(g), coef(c), degree(d) {}
        polynomial_kernel() : gamma(1), coef(0), degree(1) {}
        polynomial_kernel(
            const polynomial_kernel& k
        ) : gamma(k.gamma), coef(k.coef), degree(k.degree) {}

        typedef T type;
        const scalar_type gamma;
        const scalar_type coef;
        const scalar_type degree;

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const
        { 
            return std::pow(gamma*(trans(a)*b) + coef, degree);
        }

        polynomial_kernel& operator= (
            const polynomial_kernel& k
        )
        {
            const_cast<scalar_type&>(gamma) = k.gamma;
            const_cast<scalar_type&>(coef) = k.coef;
            const_cast<scalar_type&>(degree) = k.degree;
            return *this;
        }
    };

    template <
        typename T
        >
    void serialize (
        const polynomial_kernel<T>& item,
        std::ostream& out
    )
    {
        try
        {
            serialize(item.gamma, out);
            serialize(item.coef, out);
            serialize(item.degree, out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type polynomial_kernel"); 
        }
    }

    template <
        typename T
        >
    void deserialize (
        polynomial_kernel<T>& item,
        std::istream& in 
    )
    {
        typedef typename T::type scalar_type;
        try
        {
            deserialize(const_cast<scalar_type&>(item.gamma), in);
            deserialize(const_cast<scalar_type&>(item.coef), in);
            deserialize(const_cast<scalar_type&>(item.degree), in);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while deserializing object of type polynomial_kernel"); 
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct linear_kernel
    {
        typedef typename T::type scalar_type;
        typedef T sample_type;
        typedef typename T::mem_manager_type mem_manager_type;

        scalar_type operator() (
            const sample_type& a,
            const sample_type& b
        ) const
        { 
            return trans(a)*b;
        }
    };

    template <
        typename T
        >
    void serialize (
        const linear_kernel<T>& item,
        std::ostream& out
    ){}

    template <
        typename T
        >
    void deserialize (
        linear_kernel<T>& item,
        std::istream& in 
    ){}

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SVm_KERNEL


