// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SVm_KERNEL_MATRIX_ABSTRACT_
#ifdef DLIB_SVm_KERNEL_MATRIX_ABSTRACT_

#include <vector>
#include "kernel_abstract.h"
#include "../matrix/matrix_abstract.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                                  Symmetric Kernel Matrices
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kernel,
        const std::vector<typename kernel_type::sample_type,alloc>& m
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - R is a square matrix of m.size() rows by m.size() columns
                - for all valid r and c:
                    - R(r,c) == kernel(m[r], m[c])
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kernel,
        const matrix_exp& m
    )
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
            - is_vector(m) == true
            - the elements of m must be the type of element the given kernel operates on.  
              (e.g. kernel(m(0), m(1)) should be a legal expression)
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - R is a square matrix of m.size() rows by m.size() columns
                - for all valid r and c:
                    - R(r,c) == kernel(m(r), m(c))
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                              Column or Row Kernel Matrices
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type,alloc>& m,
        const typename kernel_type::sample_type& samp
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - is_col_vector(R) == true
                - R.size() == m.size()
                - for all valid i:
                    - R(i) == kernel(m[i], samp)
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kern,
        const matrix_exp& m,
        const typename kernel_type::sample_type& samp
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
            - is_vector(m) == true
            - the elements of m must be the type of element the given kernel operates on.  
              (e.g. kernel(m(0), m(1)) should be a legal expression)
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - is_col_vector(R) == true
                - R.size() == m.size()
                - for all valid i:
                    - R(i) == kernel(m(i), samp)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kern,
        const typename kernel_type::sample_type& samp,
        const std::vector<typename kernel_type::sample_type,alloc>& m
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - is_row_vector(R) == true
                - R.size() == m.size()
                - for all valid i:
                    - R(i) == kernel(samp,m[i])
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kern,
        const typename kernel_type::sample_type& samp,
        const matrix_exp& m
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
            - is_vector(m) == true
            - the elements of m must be the type of element the given kernel operates on.  
              (e.g. kernel(m(0), m(1)) should be a legal expression)
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - is_row_vector(R) == true
                - R.size() == m.size()
                - for all valid i:
                    - R(i) == kernel(samp, m(i))
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                              Rectangular Kernel Matrices
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type,
        typename alloc
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kern,
        const std::vector<typename kernel_type::sample_type,alloc>& lhs,
        const std::vector<typename kernel_type::sample_type,alloc>& rhs
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - R.nr() == lhs.size()
                - R.nc() == rhs.size()
                - for all valid r and c:
                    - R(r,c) == kernel(lhs[r], rhs[c])
    !*/
    
// ----------------------------------------------------------------------------------------

    template <
        typename kernel_type
        >
    const kernel_matrix_exp3<kernel_type,EXP1,EXP2> kernel_matrix (
        const kernel_type& kern,
        const matrix_exp& lhs,
        const matrix_exp& rhs
    );
    /*!
        requires
            - kernel == a kernel function object as defined by the file dlib/svm/kernel_abstract.h
            - is_vector(lhs) == true
            - is_vector(rhs) == true
            - the elements of lhs and rhs must be the type of element the given kernel operates on.  
              (e.g. kernel(lhs(0), rhs(0)) should be a legal expression)
        ensures
            - returns a matrix R such that:
                - R::type == kernel_type::scalar_type
                - R.nr() == lhs.size()
                - R.nc() == rhs.size()
                - for all valid r and c:
                    - R(r,c) == kernel(lhs(r), rhs(c))
    !*/

// ----------------------------------------------------------------------------------------

}
    
#endif // DLIB_SVm_KERNEL_MATRIX_ABSTRACT_

