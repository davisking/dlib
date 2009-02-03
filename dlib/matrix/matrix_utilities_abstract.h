// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MATRIx_UTILITIES_ABSTRACT_
#ifdef DLIB_MATRIx_UTILITIES_ABSTRACT_

#include "matrix_abstract.h"
#include <complex>
#include "../pixel.h"
#include "../geometry.h"
#inclue <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                                   Simple matrix utilities 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    const matrix_exp diag (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a column vector R that contains the elements from the diagonal 
              of m in the order R(0)==m(0,0), R(1)==m(1,1), R(2)==m(2,2) and so on.
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp diagm (
        const matrix_exp& m
    );
    /*!
        requires
            - m is a row or column matrix
        ensures
            - returns a square matrix M such that:
                - diag(M) == m
                - non diagonal elements of M are 0
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp trans (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns the transpose of the matrix m
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp lowerm (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a matrix M such that:
                - M::type == the same type that was in m
                - M has the same dimensions as m
                - M is the lower triangular part of m.  That is:
                    - if (r >= c) then
                        - M(r,c) == m(r,c)
                    - else
                        - M(r,c) == 0
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp lowerm (
        const matrix_exp& m,
        const matrix_exp::type scalar_value
    );
    /*!
        ensures
            - returns a matrix M such that:
                - M::type == the same type that was in m
                - M has the same dimensions as m
                - M is the lower triangular part of m except that the diagonal has
                  been set to scalar_value.  That is:
                    - if (r > c) then
                        - M(r,c) == m(r,c)
                    - else if (r == c) then
                        - M(r,c) == scalar_value 
                    - else
                        - M(r,c) == 0
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp upperm (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a matrix M such that:
                - M::type == the same type that was in m
                - M has the same dimensions as m
                - M is the upper triangular part of m.  That is:
                    - if (r <= c) then
                        - M(r,c) == m(r,c)
                    - else
                        - M(r,c) == 0
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp upperm (
        const matrix_exp& m,
        const matrix_exp::type scalar_value
    );
    /*!
        ensures
            - returns a matrix M such that:
                - M::type == the same type that was in m
                - M has the same dimensions as m
                - M is the upper triangular part of m except that the diagonal has
                  been set to scalar_value.  That is:
                    - if (r < c) then
                        - M(r,c) == m(r,c)
                    - else if (r == c) then
                        - M(r,c) == scalar_value 
                    - else
                        - M(r,c) == 0
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        long NR, 
        long NC, 
        T val
        >
    const matrix_exp uniform_matrix (
    );
    /*!
        requires
            - NR > 0 && NC > 0
        ensures
            - returns an NR by NC matrix with elements of type T and all set to val.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long NR, 
        long NC
        >
    const matrix_exp uniform_matrix (
        const T& val
    );
    /*!
        requires
            - NR > 0 && NC > 0
        ensures
            - returns an NR by NC matrix with elements of type T and all set to val.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_exp uniform_matrix (
        long nr,
        long nc,
        const T& val
    );
    /*!
        requires
            - nr > 0 && nc > 0
        ensures
            - returns an nr by nc matrix with elements of type T and all set to val.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_exp identity_matrix (
        long N
    );
    /*!
        requires
            - N > 0
        ensures
            - returns an N by N identity matrix with elements of type T.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        long N
        >
    const matrix_exp identity_matrix (
    );
    /*!
        requires
            - N > 0
        ensures
            - returns an N by N identity matrix with elements of type T.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long R,
        long C
        >
    const matrix_exp rotate (
        const matrix_exp& m
    );
    /*!
        requires
            - R < m.nr()
            - C < m.nc()
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m
                - R has the same dimensions as m
                - for all valid r and c:
                  R( (r+R)%m.nr() , (c+C)%m.nc() ) == m(r,c)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type
        >
    const matrix_exp vector_to_matrix (
        const vector_type& vector
    );
    /*!
        requires
            - vector_type is an implementation of array/array_kernel_abstract.h or
              std::vector or dlib::std_vector_c or dlib::matrix
        ensures
            - if (vector_type is a dlib::matrix) then
                - returns a reference to vector
            - else
                - returns a matrix R such that:
                    - R.nr() == vector.size() 
                    - R.nc() == 1 
                    - for all valid r:
                      R(r) == vector[r]
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename array_type
        >
    const matrix_exp array_to_matrix (
        const array_type& array
    );
    /*!
        requires
            - array_type is an implementation of array2d/array2d_kernel_abstract.h
              or dlib::matrix
        ensures
            - if (array_type is a dlib::matrix) then
                - returns a reference to array 
            - else
                - returns a matrix R such that:
                    - R.nr() == array.nr() 
                    - R.nc() == array.nc()
                    - for all valid r and c:
                      R(r, c) == array[r][c]
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long R,
        long C
        >
    const matrix_exp removerc (
        const matrix_exp& m
    );
    /*!
        requires
            - m.nr() > R
            - m.nc() > C
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() - 1
                - M.nc() == m.nc() - 1
                - M == m with its R row and C column removed
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp removerc (
        const matrix_exp& m,
        long R,
        long C
    );
    /*!
        requires
            - m.nr() > R
            - m.nc() > C
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() - 1
                - M.nc() == m.nc() - 1
                - M == m with its R row and C column removed
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long R
        >
    const matrix_exp remove_row (
        const matrix_exp& m
    );
    /*!
        requires
            - m.nr() > R
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() - 1
                - M.nc() == m.nc() 
                - M == m with its R row removed
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp remove_row (
        const matrix_exp& m,
        long R
    );
    /*!
        requires
            - m.nr() > R
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() - 1
                - M.nc() == m.nc() 
                - M == m with its R row removed
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long C
        >
    const matrix_exp remove_col (
        const matrix_exp& m
    );
    /*!
        requires
            - m.nc() > C
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() 
                - M.nc() == m.nc() - 1 
                - M == m with its C column removed
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp remove_col (
        const matrix_exp& m,
        long C
    );
    /*!
        requires
            - m.nc() > C
        ensures
            - returns a matrix M such that:
                - M.nr() == m.nr() 
                - M.nc() == m.nc() - 1 
                - M == m with its C column removed
    !*/

// ----------------------------------------------------------------------------------------

    template <
       typename target_type
       >
    const matrix_exp matrix_cast (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a matrix R where for all valid r and c:
              R(r,c) == static_cast<target_type>(m(r,c))
              also, R has the same dimensions as m.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long NR,
        long NC,
        typename MM,
        typename U
        >
    void set_all_elements (
        matrix<T,NR,NC,MM>& m,
        U value
    );
    /*!
        ensures
            - for all valid r and c:
              m(r,c) == value
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::matrix_type tmp (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a temporary matrix object that is a copy of m. 
              (This allows you to easily force a matrix_exp to fully evaluate)
    !*/

// ----------------------------------------------------------------------------------------

    bool equal (
        const matrix_exp& a,
        const matrix_exp& b,
        const matrix_exp::type epsilon = 100*std::numeric_limits<matrix_exp::type>::epsilon()
    );
    /*!
        ensures
            - if (a and b don't have the same dimensions) then
                - returns false
            - else if (there exists an r and c such that abs(a(r,c)-b(r,c)) > epsilon) then
                - returns false
            - else
                - returns true
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp pointwise_multiply (
        const matrix_exp& a,
        const matrix_exp& b 
    );
    /*!
        requires
            - a.nr() == b.nr()
            - a.nc() == b.nc()
            - a and b both contain the same type of element
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in a and b.
                - R has the same dimensions as a and b. 
                - for all valid r and c:
                  R(r,c) == a(r,c) * b(r,c)
    !*/

    const matrix_exp pointwise_multiply (
        const matrix_exp& a,
        const matrix_exp& b,
        const matrix_exp& c 
    );
    /*!
        performs pointwise_multiply(a,pointwise_multiply(b,c));
    !*/

    const matrix_exp pointwise_multiply (
        const matrix_exp& a,
        const matrix_exp& b,
        const matrix_exp& c,
        const matrix_exp& d 
    );
    /*!
        performs pointwise_multiply(pointwise_multiply(a,b),pointwise_multiply(c,d));
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp tensor_product (
        const matrix_exp& a,
        const matrix_exp& b 
    );
    /*!
        requires
            - a and b both contain the same type of element
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in a and b.
                - R.nr() == a.nr()*b.nr()  
                - R.nc() == a.nc()*b.nc()  
                - for all valid r and c:
                  R(r,c) == a(r/b.nr(), c/b.nc()) * b(r%b.nr(), c%b.nc())
                - I.e. R is the tensor product of matrix a with matrix b
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp scale_columns (
        const matrix_exp& m,
        const matrix_exp& v
    );
    /*!
        requires
            - v.nc() == 1 (i.e. v is a column vector)
            - v.nr() == m.nc()
            - m and v both contain the same type of element
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m and v.
                - R has the same dimensions as m. 
                - for all valid r and c:
                  R(r,c) == m(r,c) * v(c)
                - i.e. R is the result of multiplying each of m's columns by
                  the corresponding scalar in v.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    void sort_columns (
        matrix<T>& m,
        matrix<T>& v
    );
    /*!
        requires
            - v.nc() == 1 (i.e. v is a column vector)
            - v.nr() == m.nc()
            - m and v both contain the same type of element
        ensures
            - the dimensions for m and v are not changed
            - sorts the columns of m according to the values in v.
              i.e. 
                - #v == the contents of v but in sorted order according to
                  operator<.  So smaller elements come first.
                - Let #v(new(i)) == v(i) (i.e. new(i) is the index element i moved to)
                - colm(#m,new(i)) == colm(m,i) 
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T>
    void rsort_columns (
        matrix<T>& m,
        matrix<T>& v
    );
    /*!
        requires
            - v.nc() == 1 (i.e. v is a column vector)
            - v.nr() == m.nc()
            - m and v both contain the same type of element
        ensures
            - the dimensions for m and v are not changed
            - sorts the columns of m according to the values in v.
              i.e. 
                - #v == the contents of v but in sorted order according to
                  operator>.  So larger elements come first.
                - Let #v(new(i)) == v(i) (i.e. new(i) is the index element i moved to)
                - colm(#m,new(i)) == colm(m,i) 
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type length_squared (
        const matrix_exp& m
    );
    /*!
        requires
            - m.nr() == 1 || m.nc() == 1
              (i.e. m must be a vector)
        ensures
            - returns sum(squared(m))
              (i.e. returns the square of the length of the vector m)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type length (
        const matrix_exp& m
    );
    /*!
        requires
            - m.nr() == 1 || m.nc() == 1
              (i.e. m must be a vector)
        ensures
            - returns sqrt(sum(squared(m)))
              (i.e. returns the length of the vector m)
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                              Statistics
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    const matrix_exp::type min (
        const matrix_exp& m
    );
    /*!
        requires
            - m.size() > 0
        ensures
            - returns the value of the smallest element of m
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type max (
        const matrix_exp& m
    );
    /*!
        requires
            - m.size() > 0
        ensures
            - returns the value of the biggest element of m
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type sum (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns the sum of all elements in m
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type prod (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns the results of multiplying all elements of m together. 
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type mean (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns the mean of all elements in m. 
              (i.e. returns sum(m)/(m.nr()*m.nc()))
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp::type variance (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns the unbiased sample variance of all elements in m 
              (i.e. 1.0/(m.nr()*m.nc() - 1)*(sum of all pow(m(i,j) - mean(m),2)))
    !*/

// ----------------------------------------------------------------------------------------

    const matrix covariance (
        const matrix_exp& m
    );
    /*!
        requires
            - matrix_exp::type == a dlib::matrix object
            - m.nr() > 1
            - m.nc() == 1 (i.e. m is a column vector)
            - for all valid i, j:
                - m(i).nr() > 0
                - m(i).nc() == 1
                - m(i).nr() == m(j).nr() 
                - i.e. m contains only column vectors and all the column vectors
                  have the same non-zero length
        ensures
            - returns the unbiased sample covariance matrix for the set of samples
              in m.  
              (i.e. 1.0/(m.nr()-1)*(sum of all (m(i) - mean(m))*trans(m(i) - mean(m))))
            - the returned matrix will contain elements of type matrix_exp::type::type.
            - the returned matrix will have m(0).nr() rows and columns.
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                                 Pixel and Image Utilities
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename P
        >
    const matrix_exp pixel_to_vector (
        const P& pixel
    );
    /*!
        requires
            - pixel_traits<P>::has_alpha == false
        ensures
            - returns a matrix M such that:
                - M::type == T
                - M::NC == 1 
                - M::NR == pixel_traits<P>::num
                - if (pixel_traits<P>::grayscale) then
                    - M(0) == pixel 
                - if (pixel_traits<P>::rgb) then
                    - M(0) == pixel.red 
                    - M(1) == pixel.green 
                    - M(2) == pixel.blue 
                - if (pixel_traits<P>::hsi) then
                    - M(0) == pixel.h 
                    - M(1) == pixel.s 
                    - M(2) == pixel.i 
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename P
        >
    void vector_to_pixel (
        P& pixel,
        const matrix_exp& vector 
    );
    /*!
        requires
            - pixel_traits<P>::has_alpha == false
            - vector::NR == pixel_traits<P>::num
            - vector::NC == 1 
              (i.e. you have to use a statically dimensioned vector)
        ensures
            - if (pixel_traits<P>::grayscale) then
                - pixel == M(0) 
            - if (pixel_traits<P>::rgb) then
                - pixel.red   == M(0)  
                - pixel.green == M(1) 
                - pixel.blue  == M(2)  
            - if (pixel_traits<P>::hsi) then
                - pixel.h == M(0)
                - pixel.s == M(1)
                - pixel.i == M(2)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        long lower,
        long upper 
        >
    const matrix_exp clamp (
        const matrix_exp& m
    );
    /*!
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m
                - R has the same dimensions as m
                - for all valid r and c:
                    - if (m(r,c) > upper) then
                        - R(r,c) == upper
                    - else if (m(r,c) < lower) then
                        - R(r,c) == lower
                    - else
                        - R(r,c) == m(r,c)
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_UTILITIES_ABSTRACT_

