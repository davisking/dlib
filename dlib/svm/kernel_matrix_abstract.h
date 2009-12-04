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

    template <
        typename kernel_type
        >
    const matrix_exp kernel_matrix (
        const kernel_type& kernel,
        const std::vector<typename kernel_type::sample_type>& m
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

}

#endif // DLIB_SVm_KERNEL_MATRIX_ABSTRACT_

