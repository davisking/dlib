// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DPCA_ABSTRaCT_
#ifdef DLIB_DPCA_ABSTRaCT_

#include <limits>
#include <cmath>
#include "../matrix/matrix_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename column_matrix
        >
    class discriminant_pca
    {
        /*!
            REQUIREMENTS ON column_matrix
                Must be some type of dlib::matrix capable of representing a column vector.

            INITIAL VALUE
                - in_vector_size() == 0
                - between_class_weight() == 1
                - within_class_weight() == 1

            WHAT THIS OBJECT REPRESENTS
                This object implements the Discriminant PCA technique described in the paper:
                    A New Discriminant Principal Component Analysis Method with Partial Supervision
                    by Dan Sun and Daoqiang Zhang

                TODO
        !*/

    public:

        struct discriminant_pca_error : public error;
        /*!
            This exception is thrown if there is some error that prevents us from creating
            a DPCA matrix.
        !*/

        typedef typename column_matrix::mem_manager_type mem_manager_type;
        typedef typename column_matrix::type scalar_type;
        typedef typename column_matrix::layout_type layout_type;
        const static long N = column_matrix::NR;
        typedef matrix<scalar_type,0,0,mem_manager_type,layout_type> general_matrix;

        discriminant_pca (
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        void clear(
        );
        /*!
            ensures
                - #*this has its initial value
        !*/

        long in_vector_size (
        ) const;
        /*!
            ensures
                - if (this object has been presented with any input vectors) then
                    - returns the dimension of the column vectors used with this object
                - else
                    - returns 0
        !*/

        void set_within_class_weight (
            scalar_type weight
        );
        /*!
            requires
                - weight >= 0
            ensures
                - #within_class_weight() == weight
        !*/

        scalar_type within_class_weight (
        ) const;
        /*!
        !*/

        void set_between_class_weight (
            scalar_type weight
        );
        /*!
            requires
                - weight >= 0
            ensures
                - #between_class_weight() == weight
        !*/

        scalar_type between_class_weight (
        ) const;
        /*!
        !*/

        void add_to_within_class_variance(
            const column_matrix& x,
            const column_matrix& y
        );
        /*!
            requires
                - is_col_vector(x) == true
                - is_col_vector(y) == true
                - x.size() == y.size()
                - if (in_vector_size() != 0) then
                    - x.size() == y.size() == in_vector_size()
            ensures
                - #in_vector_size() == x.size()
                - TODO
        !*/

        void add_to_between_class_variance(
            const column_matrix& x,
            const column_matrix& y
        );
        /*!
            requires
                - is_col_vector(x) == true
                - is_col_vector(y) == true
                - x.size() == y.size()
                - if (in_vector_size() != 0) then
                    - x.size() == y.size() == in_vector_size()
            ensures
                - #in_vector_size() == x.size()
                - TODO
        !*/

        void add_to_total_variance(
            const column_matrix& x
        );
        /*!
            requires
                - is_col_vector(x) == true
                - if (in_vector_size() != 0) then
                    - x.size() == in_vector_size()
            ensures
                - #in_vector_size() == x.size()
                - TODO
        !*/

        const general_matrix dpca_matrix (
            const double eps = 0.99
        ) const;
        /*!
            requires
                - 0 < eps <= 1
                - in_vector_size() != 0
                  (i.e. you have to have given this object some data)
            ensures
                - computes and returns the matrix M given by dpca_matrix(M,eigen,eps).  
                  That is, this function returns the dpca_matrix computed by the function
                  defined below.
            throws
                - discriminant_pca_error
                    This exception is thrown if we are unable to create the dpca_matrix for some 
                    reason.  For example, if only within class examples have been given or
                    within_class_weight() is very large then all eigenvalues will be negative and
                    that prevents this algorithm from working properly.
        !*/

        void dpca_matrix (
            general_matrix& dpca_mat,
            general_matrix& eigenvalues,
            const double eps = 0.99
        ) const;
        /*!
            requires
                - 0 < eps <= 1
                - in_vector_size() != 0
                  (i.e. you have to have given this object some data)
            ensures
                - #is_col_vector(eigenvalues) == true
                - #eigenvalues.size() == #dpca_mat.nr()
                - #dpca_mat.nc() == in_vector_size()
                - all values in #eigenvalues are > 0
                - TODO
            throws
                - discriminant_pca_error
                    This exception is thrown if we are unable to create the dpca_matrix for some 
                    reason.  For example, if only within class examples have been given or
                    within_class_weight() is very large then all eigenvalues will be negative and
                    that prevents this algorithm from working properly.
        !*/

        void swap (
            discriminant_pca& item
        );
        /*!
            ensures
                - swaps *this and item
        !*/

    };

    template <
        typename column_matrix
        >
    inline void swap (
        discriminant_pca<column_matrix>& a, 
        discriminant_pca<column_matrix>& b 
    ) { a.swap(b); }   
    /*!
        provides a global swap function
    !*/

    template <
        typename column_matrix,
        >
    void deserialize (
        discriminant_pca<column_matrix>& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/

    template <
        typename column_matrix,
        >
    void serialize (
        const discriminant_pca<column_matrix>& item, 
        std::ostream& out 
    );   
    /*!
        provides serialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DPCA_ABSTRaCT_

