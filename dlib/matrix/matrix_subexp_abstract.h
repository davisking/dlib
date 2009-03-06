// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MATRIx_SUBEXP_ABSTRACT_
#ifdef DLIB_MATRIx_SUBEXP_ABSTRACT_

#include "matrix_abstract.h"
#include "../geometry.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    const rectangle get_rect (  
        const matrix_exp& m
    );
    /*!
        ensures
            - returns rectangle(0, 0, m.nc()-1, m.nr()-1)
              (i.e. returns a rectangle that has the same dimensions as
              the matrix m)
    !*/

// ----------------------------------------------------------------------------------------

    template <long start, long inc, long end>
    const matrix_exp range (
    );
    /*!
        requires
            - inc > 0
        ensures
            - returns a matrix R such that:
                - R::type == long
                - R.nr() == 1
                - R.nc() == abs(end - start)/inc + 1
                - if (start <= end) then
                    - R(i) == start + i*inc
                - else
                    - R(i) == start - i*inc
    !*/

    template <long start, long end>
    const matrix_exp range (
    ) { return range<start,1,end>(); }

    const matrix_exp range (
        long start,
        long inc,
        long end
    ); 
    /*!
        requires
            - inc > 0
        ensures
            - returns a matrix R such that:
                - R::type == long
                - R.nr() == 1
                - R.nc() == abs(end - start)/inc + 1
                - if (start <= end) then
                    - R(i) == start + i*inc
                - else
                    - R(i) == start - i*inc
    !*/

    const matrix_exp range (
        long start,
        long end
    ) { return range(start,1,end); }

// ----------------------------------------------------------------------------------------

    const matrix_exp subm (
        const matrix_exp& m,
        const matrix_exp& rows,
        const matrix_exp& cols,
    );
    /*!
        requires
            - rows and cols contain elements of type long
            - 0 <= min(rows) && max(rows) < m.nr() 
            - 0 <= min(cols) && max(cols) < m.nc()
            - rows.nr() == 1 || rows.nc() == 1
            - cols.nr() == 1 || cols.nc() == 1
              (i.e. rows and cols must be vectors)
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m
                - R.nr() == rows.size()
                - R.nc() == cols.size()
                - for all valid r and c:
                  R(r,c) == m(rows(r),cols(c))
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp subm (
        const matrix_exp& m,
        long row,
        long col,
        long nr,
        long nc
    );
    /*!
        requires
            - row >= 0
            - row + nr <= m.nr()
            - col >= 0
            - col + nc <= m.nc()
        ensures
            - returns a matrix R such that:
                - R.nr() == nr 
                - R.nc() == nc
                - for all valid r and c:
                  R(r, c) == m(r+row,c+col)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp subm (
        const matrix_exp& m,
        const rectangle& rect
    );
    /*!
        requires
            - get_rect(m).contains(rect) == true
              (i.e. rect is a region inside the matrix m)
        ensures
            - returns a matrix R such that:
                - R.nr() == rect.height()  
                - R.nc() == rect.width()
                - for all valid r and c:
                  R(r, c) == m(r+rect.top(), c+rect.left())
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp rowm (
        const matrix_exp& m,
        long row
    );
    /*!
        requires
            - 0 <= row < m.nr()
        ensures
            - returns a matrix R such that:
                - R.nr() == 1
                - R.nc() == m.nc()
                - for all valid i:
                  R(i) == m(row,i)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp rowm (
        const matrix_exp& m,
        long row,
        long length
    );
    /*!
        requires
            - 0 <= row < m.nr()
            - 0 <= length <= m.nc()
        ensures
            - returns a matrix R such that:
                - R.nr() == 1
                - R.nc() == length
                - for all valid i:
                  R(i) == m(row,i)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp rowm (
        const matrix_exp& m,
        const matrix_exp& rows
    );
    /*!
        requires
            - rows contains elements of type long
            - 0 <= min(rows) && max(rows) < m.nr() 
            - rows.nr() == 1 || rows.nc() == 1
              (i.e. rows must be a vector)
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m
                - R.nr() == rows.size()
                - R.nc() == m.nc() 
                - for all valid r and c:
                  R(r,c) == m(rows(r),c)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp colm (
        const matrix_exp& m,
        long col 
    );
    /*!
        requires
            - 0 <= col < m.nc()
        ensures
            - returns a matrix R such that:
                - R.nr() == m.nr() 
                - R.nc() == 1
                - for all valid i:
                  R(i) == m(i,col)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp colm (
        const matrix_exp& m,
        long col,
        long length
    );
    /*!
        requires
            - 0 <= col < m.nc()
            - 0 <= length <= m.nr()
        ensures
            - returns a matrix R such that:
                - R.nr() == length 
                - R.nc() == 1
                - for all valid i:
                  R(i) == m(i,col)
    !*/

// ----------------------------------------------------------------------------------------

    const matrix_exp colm (
        const matrix_exp& m,
        const matrix_exp& cols
    );
    /*!
        requires
            - cols contains elements of type long
            - 0 <= min(cols) && max(cols) < m.nc() 
            - cols.nr() == 1 || cols.nc() == 1
              (i.e. cols must be a vector)
        ensures
            - returns a matrix R such that:
                - R::type == the same type that was in m
                - R.nr() == m.nr()
                - R.nc() == cols.size()
                - for all valid r and c:
                  R(r,c) == m(r,cols(c))
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_subm (
        matrix& m,
        long row,
        long col,
        long nr,
        long nc
    );
    /*!
        requires
            - row >= 0
            - row + nr <= m.nr()
            - col >= 0
            - col + nc <= m.nc()
        ensures
            - statements of the following form:
                - set_subm(m,row,col,nr,nc) = some_matrix;
              result in it being the case that:
                - subm(m,row,col,nr,nc) == some_matrix.

            - statements of the following form:
                - set_subm(m,row,col,nr,nc) = scalar_value;
              result in it being the case that:
                - subm(m,row,col,nr,nc) == uniform_matrix<matrix::type>(nr,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_subm (
        matrix& m,
        const rectangle& rect
    );
    /*!
        requires
            - get_rect(m).contains(rect) == true
              (i.e. rect is a region inside the matrix m)
        ensures
            - statements of the following form:
                - set_subm(m,rect) = some_matrix;
              result in it being the case that:
                - subm(m,rect) == some_matrix.

            - statements of the following form:
                - set_subm(m,rect) = scalar_value;
              result in it being the case that:
                - subm(m,rect) == uniform_matrix<matrix::type>(nr,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_subm (
        matrix& m,
        const matrix_exp& rows,
        const matrix_exp& cols
    );
    /*!
        requires
            - rows and cols contain elements of type long
            - 0 <= min(rows) && max(rows) < m.nr() 
            - 0 <= min(cols) && max(cols) < m.nc()
            - rows.nr() == 1 || rows.nc() == 1
            - cols.nr() == 1 || cols.nc() == 1
              (i.e. rows and cols must be vectors)
        ensures
            - statements of the following form:
                - set_subm(m,rows,cols) = some_matrix;
              result in it being the case that:
                - subm(m,rows,cols) == some_matrix.

            - statements of the following form:
                - set_subm(m,rows,cols) = scalar_value;
              result in it being the case that:
                - subm(m,rows,cols) == uniform_matrix<matrix::type>(nr,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_rowm (
        matrix& m,
        long row
    );
    /*!
        requires
            - 0 <= row < m.nr()
        ensures
            - statements of the following form:
                - set_rowm(m,row) = some_matrix;
              result in it being the case that:
                - rowm(m,row) == some_matrix.

            - statements of the following form:
                - set_rowm(m,row) = scalar_value;
              result in it being the case that:
                - rowm(m,row) == uniform_matrix<matrix::type>(1,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_rowm (
        matrix& m,
        const matrix_exp& rows
    );
    /*!
        requires
            - rows contains elements of type long
            - 0 <= min(rows) && max(rows) < m.nr() 
            - rows.nr() == 1 || rows.nc() == 1
              (i.e. rows must be a vector)
        ensures
            - statements of the following form:
                - set_rowm(m,rows) = some_matrix;
              result in it being the case that:
                - rowm(m,rows) == some_matrix.

            - statements of the following form:
                - set_rowm(m,rows) = scalar_value;
              result in it being the case that:
                - rowm(m,rows) == uniform_matrix<matrix::type>(nr,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_colm (
        matrix& m,
        long col 
    );
    /*!
        requires
            - 0 <= col < m.nr()
        ensures
            - statements of the following form:
                - set_colm(m,col) = some_matrix;
              result in it being the case that:
                - colm(m,col) == some_matrix.

            - statements of the following form:
                - set_colm(m,col) = scalar_value;
              result in it being the case that:
                - colm(m,col) == uniform_matrix<matrix::type>(nr,1,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

    assignable_matrix_expression set_colm (
        matrix& m,
        const matrix_exp& cols
    );
    /*!
        requires
            - cols contains elements of type long
            - 0 <= min(cols) && max(cols) < m.nc() 
            - cols.nr() == 1 || cols.nc() == 1
              (i.e. cols must be a vector)
        ensures
            - statements of the following form:
                - set_colm(m,cols) = some_matrix;
              result in it being the case that:
                - colm(m,cols) == some_matrix.

            - statements of the following form:
                - set_colm(m,cols) = scalar_value;
              result in it being the case that:
                - colm(m,cols) == uniform_matrix<matrix::type>(nr,nc,scalar_value).
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_SUBEXP_ABSTRACT_

