// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_SUBEXP_
#define DLIB_MATRIx_SUBEXP_

#include "matrix_subexp_abstract.h"
#include "matrix.h"
#include "../geometry/rectangle.h"
#include "matrix_expressions.h"



namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    
    template <long start, long inc, long end>
    const matrix_range_static_exp<start,inc,end> range (
    ) 
    { 
        COMPILE_TIME_ASSERT(inc > 0);
        return matrix_range_static_exp<start,inc,end>(); 
    }

    template <long start, long end>
    const matrix_range_static_exp<start,1,end> range (
    ) 
    { 
        return matrix_range_static_exp<start,1,end>(); 
    }

    inline const matrix_range_exp<long> range (
        long start,
        long end
    ) 
    { 
        return matrix_range_exp<long>(start,end); 
    }

    inline const matrix_range_exp<long> range (
        long start,
        long inc,
        long end
    ) 
    { 
        DLIB_ASSERT(inc > 0, 
            "\tconst matrix_exp range(start, inc, end)"
            << "\n\tstart can't be bigger than end"
            << "\n\tstart: " << start 
            << "\n\tinc:   " << inc
            << "\n\tend:   " << end
            );

        return matrix_range_exp<long>(start,inc,end); 
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const rectangle get_rect (
        const matrix_exp<EXP>& m
    )
    {
        return rectangle(0, 0, m.nc()-1, m.nr()-1);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const matrix_sub_exp<EXP> subm (
        const matrix_exp<EXP>& m,
        long r, 
        long c,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(r >= 0 && c >= 0 && r+nr <= m.nr() && c+nc <= m.nc(), 
            "\tconst matrix_exp subm(const matrix_exp& m, r, c, nr, nc)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tr:      " << r 
            << "\n\tc:      " << c 
            << "\n\tnr:     " << nr 
            << "\n\tnc:     " << nc 
            );

        typedef matrix_sub_exp<EXP> exp;
        return exp(m.ref(),r,c,nr,nc);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const matrix_sub_exp<EXP> subm (
        const matrix_exp<EXP>& m,
        const rectangle& rect
    )
    {
        DLIB_ASSERT(get_rect(m).contains(rect) == true, 
            "\tconst matrix_exp subm(const matrix_exp& m, const rectangle& rect)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trect.left():   " << rect.left()
            << "\n\trect.top():    " << rect.top()
            << "\n\trect.right():  " << rect.right()
            << "\n\trect.bottom(): " << rect.bottom()
            );

        typedef matrix_sub_exp<EXP> exp;
        return exp(m.ref(),rect.top(),rect.left(),rect.height(),rect.width());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP,
        typename EXPr,
        typename EXPc
        >
    const matrix_sub_range_exp<EXP,EXPr,EXPc> subm (
        const matrix_exp<EXP>& m,
        const matrix_exp<EXPr>& rows,
        const matrix_exp<EXPc>& cols
    )
    {
        // the rows and cols matrices must contain elements of type long
        COMPILE_TIME_ASSERT((is_same_type<typename EXPr::type,long>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXPc::type,long>::value == true));

        DLIB_ASSERT(0 <= min(rows) && max(rows) < m.nr() && 0 <= min(cols) && max(cols) < m.nc() &&
                    (rows.nr() == 1 || rows.nc() == 1) && (cols.nr() == 1 || cols.nc() == 1), 
            "\tconst matrix_exp subm(const matrix_exp& m, const matrix_exp& rows, const matrix_exp& cols)"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(rows):  " << min(rows) 
            << "\n\tmax(rows):  " << max(rows) 
            << "\n\tmin(cols):  " << min(cols) 
            << "\n\tmax(cols):  " << max(cols) 
            << "\n\trows.nr():  " << rows.nr()
            << "\n\trows.nc():  " << rows.nc()
            << "\n\tcols.nr():  " << cols.nr()
            << "\n\tcols.nc():  " << cols.nc()
            );

        typedef matrix_sub_range_exp<EXP,EXPr,EXPc> exp;
        return exp(m.ref(),rows.ref(),cols.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_rowm
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = 1;
            const static long NC = EXP::NC;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long row, long, long c)
            { return m(row,c); }

            template <typename M>
            static long nr (const M& m) { return 1; }
            template <typename M>
            static long nc (const M& m) { return m.nc(); }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,long,op_rowm> rowm (
        const matrix_exp<EXP>& m,
        long row
    )
    {
        DLIB_ASSERT(row >= 0 && row < m.nr(), 
            "\tconst matrix_exp rowm(const matrix_exp& m, row)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trow:    " << row 
            );

        typedef matrix_scalar_binary_exp<EXP,long,op_rowm> exp;
        return exp(m.ref(),row);
    }

// ----------------------------------------------------------------------------------------

    struct op_rowm2
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = 1;
            const static long NC = 0;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long row, long length, long r, long c)
            { return m(row,c); }

            template <typename M>
            static long nr (const M& m, long, long) { return 1; }
            template <typename M>
            static long nc (const M& m, long, long length) { return length; }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_ternary_exp<EXP,long,op_rowm2> rowm (
        const matrix_exp<EXP>& m,
        long row,
        long length
    )
    {
        DLIB_ASSERT(row >= 0 && row < m.nr() && 
                    length >= 0 && length <= m.nc(), 
            "\tconst matrix_exp rowm(const matrix_exp& m, row, length)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trow:    " << row 
            << "\n\tlength: " << length 
            );

        typedef matrix_scalar_ternary_exp<EXP,long,op_rowm2> exp;
        return exp(m.ref(),row, length);
    }

// ----------------------------------------------------------------------------------------

    struct op_rowm_range
    {
        template <typename EXP1, typename EXP2>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP1::cost+EXP2::cost;
            typedef typename EXP1::type type;
            typedef typename EXP1::mem_manager_type mem_manager_type;
            const static long NR = EXP2::NC*EXP2::NR;
            const static long NC = EXP1::NC;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& rows , long r, long c)
            { return m1(rows(r),c); }

            template <typename M1, typename M2>
            static long nr (const M1& m1, const M2& rows ) { return rows.size(); }
            template <typename M1, typename M2>
            static long nc (const M1& m1, const M2& ) { return m1.nc(); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_binary_exp<EXP1,EXP2,op_rowm_range> rowm (
        const matrix_exp<EXP1>& m,
        const matrix_exp<EXP2>& rows
    )
    {
        // the rows matrix must contain elements of type long
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,long>::value == true));

        DLIB_ASSERT(0 <= min(rows) && max(rows) < m.nr() && (rows.nr() == 1 || rows.nc() == 1), 
            "\tconst matrix_exp rowm(const matrix_exp& m, const matrix_exp& rows)"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(rows):  " << min(rows) 
            << "\n\tmax(rows):  " << max(rows) 
            << "\n\trows.nr():  " << rows.nr()
            << "\n\trows.nc():  " << rows.nc()
            );

        typedef matrix_binary_exp<EXP1,EXP2,op_rowm_range> exp;
        return exp(m.ref(),rows.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_colm
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = EXP::NR;
            const static long NC = 1;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long col, long r, long)
            { return m(r,col); }

            template <typename M>
            static long nr (const M& m) { return m.nr(); }
            template <typename M>
            static long nc (const M& m) { return 1; }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,long,op_colm> colm (
        const matrix_exp<EXP>& m,
        long col 
    )
    {
        DLIB_ASSERT(col >= 0 && col < m.nc(), 
            "\tconst matrix_exp colm(const matrix_exp& m, row)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tcol:    " << col 
            );

        typedef matrix_scalar_binary_exp<EXP,long,op_colm> exp;
        return exp(m.ref(),col);
    }

// ----------------------------------------------------------------------------------------

    struct op_colm2
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = 0;
            const static long NC = 1;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long col, long length, long r, long c)
            { return m(r,col); }

            template <typename M>
            static long nr (const M& m, long, long length) { return length; }
            template <typename M>
            static long nc (const M& m, long, long) { return 1; }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_ternary_exp<EXP,long,op_colm2> colm (
        const matrix_exp<EXP>& m,
        long col,
        long length
    )
    {
        DLIB_ASSERT(col >= 0 && col < m.nc() && 
                    length >= 0 && length <= m.nr(), 
            "\tconst matrix_exp colm(const matrix_exp& m, col, length)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tcol:    " << col 
            << "\n\tlength: " << length 
            );

        typedef matrix_scalar_ternary_exp<EXP,long,op_colm2> exp;
        return exp(m.ref(),col, length);
    }

// ----------------------------------------------------------------------------------------

    struct op_colm_range
    {
        template <typename EXP1, typename EXP2>
        struct op : has_destructive_aliasing
        {
            typedef typename EXP1::type type;
            typedef typename EXP1::mem_manager_type mem_manager_type;
            const static long NR = EXP1::NR;
            const static long NC = EXP2::NC*EXP2::NR;
            const static long cost = EXP1::cost+EXP2::cost;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& cols , long r, long c)
            { return m1(r,cols(c)); }

            template <typename M1, typename M2>
            static long nr (const M1& m1, const M2& cols ) { return m1.nr(); }
            template <typename M1, typename M2>
            static long nc (const M1& m1, const M2& cols ) { return cols.size(); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_binary_exp<EXP1,EXP2,op_colm_range> colm (
        const matrix_exp<EXP1>& m,
        const matrix_exp<EXP2>& cols
    )
    {
        // the cols matrix must contain elements of type long
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,long>::value == true));

        DLIB_ASSERT(0 <= min(cols) && max(cols) < m.nc() && (cols.nr() == 1 || cols.nc() == 1), 
            "\tconst matrix_exp colm(const matrix_exp& m, const matrix_exp& cols)"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(cols):  " << min(cols) 
            << "\n\tmax(cols):  " << max(cols) 
            << "\n\tcols.nr():  " << cols.nr()
            << "\n\tcols.nc():  " << cols.nc()
            );

        typedef matrix_binary_exp<EXP1,EXP2,op_colm_range> exp;
        return exp(m.ref(),cols.ref());
    }

// ----------------------------------------------------------------------------------------


    template <typename T, long NR, long NC, typename mm, typename l>
    class assignable_sub_matrix
    {
    public:
        typedef T type;
        typedef l layout_type;
        typedef matrix<T,NR,NC,mm,l> matrix_type;

        assignable_sub_matrix(
            matrix<T,NR,NC,mm,l>& m_,
            const rectangle& rect_
        ) : m(m_), rect(rect_) {}

        T& operator() (
            long r,
            long c
        )
        {
            return m(r+rect.top(),c+rect.left());
        }

        const T& operator() (
            long r,
            long c
        ) const
        {
            return m(r+rect.top(),c+rect.left());
        }

        long nr() const { return rect.height(); }
        long nc() const { return rect.width(); }

        template <typename EXP>
        assignable_sub_matrix& operator= (
            const matrix_exp<EXP>& exp
        ) 
        {
            DLIB_ASSERT( exp.nr() == (long)rect.height() && exp.nc() == (long)rect.width(),
                "\tassignable_matrix_expression set_subm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\trect.width() (target matrix):   " << rect.width()
                << "\n\trect.height() (target matrix):  " << rect.height()
                );

            if (exp.destructively_aliases(m) == false)
            {
                matrix_assign(*this, exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to m to 
                // avoid aliasing issues during the copy
                this->operator=(tmp(exp));
            }

            return *this;
        }

        assignable_sub_matrix& operator= (
            const T& value
        )
        {
            for (long r = rect.top(); r <= rect.bottom(); ++r)
            {
                for (long c = rect.left(); c <= rect.right(); ++c)
                {
                    m(r,c) = value;
                }
            }

            return *this;
        }


        matrix<T,NR,NC,mm,l>& m;
        const rectangle rect;
    };


    template <typename T, long NR, long NC, typename mm, typename l>
    assignable_sub_matrix<T,NR,NC,mm,l> set_subm (
        matrix<T,NR,NC,mm,l>& m,
        const rectangle& rect
    )
    {
        DLIB_ASSERT(get_rect(m).contains(rect) == true, 
            "\tassignable_matrix_expression set_subm(matrix& m, const rectangle& rect)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trect.left():   " << rect.left()
            << "\n\trect.top():    " << rect.top()
            << "\n\trect.right():  " << rect.right()
            << "\n\trect.bottom(): " << rect.bottom()
            );


        return assignable_sub_matrix<T,NR,NC,mm,l>(m,rect);
    }


    template <typename T, long NR, long NC, typename mm, typename l>
    assignable_sub_matrix<T,NR,NC,mm,l> set_subm (
        matrix<T,NR,NC,mm,l>& m,
        long r, 
        long c,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(r >= 0 && c >= 0 && r+nr <= m.nr() && c+nc <= m.nc(), 
                    "\tassignable_matrix_expression set_subm(matrix& m, r, c, nr, nc)"
                    << "\n\tYou have specified invalid sub matrix dimensions"
                    << "\n\tm.nr(): " << m.nr()
                    << "\n\tm.nc(): " << m.nc() 
                    << "\n\tr:      " << r 
                    << "\n\tc:      " << c 
                    << "\n\tnr:     " << nr 
                    << "\n\tnc:     " << nc 
        );

        return assignable_sub_matrix<T,NR,NC,mm,l>(m,rectangle(c,r, c+nc-1, r+nr-1));
    }

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename mm, typename l, typename EXPr, typename EXPc>
    class assignable_sub_range_matrix
    {
    public:
        typedef T type;
        typedef l layout_type;
        typedef matrix<T,NR,NC,mm,l> matrix_type;

        assignable_sub_range_matrix(
            matrix<T,NR,NC,mm,l>& m_,
            const EXPr& rows_,
            const EXPc& cols_
        ) : m(m_), rows(rows_), cols(cols_) {}

        T& operator() (
            long r,
            long c
        )
        {
            return m(rows(r),cols(c));
        }

        long nr() const { return rows.size(); }
        long nc() const { return cols.size(); }


        template <typename EXP>
        assignable_sub_range_matrix& operator= (
            const matrix_exp<EXP>& exp
        ) 
        {
            DLIB_ASSERT( exp.nr() == rows.size() && exp.nc() == cols.size(),
                "\tassignable_matrix_expression set_subm(matrix& m, const matrix_exp rows, const matrix_exp cols)"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\trows.size() (target matrix):  " << rows.size()
                << "\n\tcols.size() (target matrix):  " << cols.size()
                );

            if (exp.destructively_aliases(m) == false)
            {
                matrix_assign(*this, exp);
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to m to 
                // avoid aliasing issues during the copy
                this->operator=(tmp(exp));
            }

            return *this;
        }

        assignable_sub_range_matrix& operator= (
            const T& value
        )
        {
            for (long r = 0; r < rows.size(); ++r)
            {
                for (long c = 0; c < cols.size(); ++c)
                {
                    m(rows(r),cols(c)) = value;
                }
            }

            return *this;
        }

    private:

        matrix<T,NR,NC,mm,l>& m;
        const EXPr rows;
        const EXPc cols;
    };

    template <typename T, long NR, long NC, typename mm, typename l, typename EXPr, typename EXPc>
    assignable_sub_range_matrix<T,NR,NC,mm,l,EXPr,EXPc > set_subm (
        matrix<T,NR,NC,mm,l>& m,
        const matrix_exp<EXPr>& rows,
        const matrix_exp<EXPc>& cols
    )
    {
        DLIB_ASSERT(0 <= min(rows) && max(rows) < m.nr() && 0 <= min(cols) && max(cols) < m.nc() &&
                    (rows.nr() == 1 || rows.nc() == 1) && (cols.nr() == 1 || cols.nc() == 1), 
            "\tassignable_matrix_expression set_subm(matrix& m, const matrix_exp& rows, const matrix_exp& cols)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(rows):  " << min(rows) 
            << "\n\tmax(rows):  " << max(rows) 
            << "\n\tmin(cols):  " << min(cols) 
            << "\n\tmax(cols):  " << max(cols) 
            << "\n\trows.nr():  " << rows.nr()
            << "\n\trows.nc():  " << rows.nc()
            << "\n\tcols.nr():  " << cols.nr()
            << "\n\tcols.nc():  " << cols.nc()
            );

        return assignable_sub_range_matrix<T,NR,NC,mm,l,EXPr,EXPc >(m,rows.ref(),cols.ref());
    }

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename mm, typename l, typename EXPr>
    assignable_sub_range_matrix<T,NR,NC,mm,l,EXPr,matrix_range_exp<long> > set_rowm (
        matrix<T,NR,NC,mm,l>& m,
        const matrix_exp<EXPr>& rows
    )
    {
        DLIB_ASSERT(0 <= min(rows) && max(rows) < m.nr() && (rows.nr() == 1 || rows.nc() == 1), 
            "\tassignable_matrix_expression set_rowm(matrix& m, const matrix_exp& rows)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(rows):  " << min(rows) 
            << "\n\tmax(rows):  " << max(rows) 
            << "\n\trows.nr():  " << rows.nr()
            << "\n\trows.nc():  " << rows.nc()
            );

        return assignable_sub_range_matrix<T,NR,NC,mm,l,EXPr,matrix_range_exp<long> >(m,rows.ref(),range(0,m.nc()-1));
    }

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename mm, typename l, typename EXPc>
    assignable_sub_range_matrix<T,NR,NC,mm,l,matrix_range_exp<long>,EXPc > set_colm (
        matrix<T,NR,NC,mm,l>& m,
        const matrix_exp<EXPc>& cols
    )
    {
        DLIB_ASSERT(0 <= min(cols) && max(cols) < m.nc() && (cols.nr() == 1 || cols.nc() == 1), 
            "\tassignable_matrix_expression set_colm(matrix& m, const matrix_exp& cols)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(cols):  " << min(cols) 
            << "\n\tmax(cols):  " << max(cols) 
            << "\n\tcols.nr():  " << cols.nr()
            << "\n\tcols.nc():  " << cols.nc()
            );

        return assignable_sub_range_matrix<T,NR,NC,mm,l,matrix_range_exp<long>,EXPc >(m,range(0,m.nr()-1),cols.ref());
    }

// ----------------------------------------------------------------------------------------

    template <typename T, long NR, long NC, typename mm, typename l>
    class assignable_col_matrix
    {
    public:
        typedef T type;
        typedef l layout_type;
        typedef matrix<T,NR,NC,mm,l> matrix_type;

        assignable_col_matrix(
            matrix<T,NR,NC,mm,l>& m_,
            const long col_ 
        ) : m(m_), col(col_) {}

        T& operator() (
            long r,
            long c
        )
        {
            return m(r,col);
        }

        const T& operator() (
            long r,
            long c
        ) const
        {
            return m(r,col);
        }

        long nr() const { return m.nr(); }
        long nc() const { return 1; }

        template <typename EXP>
        assignable_col_matrix& operator= (
            const matrix_exp<EXP>& exp
        ) 
        {
            DLIB_ASSERT( exp.nc() == 1 && exp.nr() == m.nr(),
                "\tassignable_matrix_expression set_colm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\tm.nr() (target matrix):   " << m.nr()
                );

            if (exp.destructively_aliases(m) == false)
            {
                matrix_assign(*this, exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to m to 
                // avoid aliasing issues during the copy
                this->operator=(tmp(exp));
            }

            return *this;
        }

        assignable_col_matrix& operator= (
            const T& value
        )
        {
            for (long i = 0; i < m.nr(); ++i)
            {
                m(i,col) = value;
            }

            return *this;
        }


        matrix<T,NR,NC,mm,l>& m;
        const long col;
    };


    template <typename T, long NR, long NC, typename mm, typename l>
    assignable_col_matrix<T,NR,NC,mm,l> set_colm (
        matrix<T,NR,NC,mm,l>& m,
        const long col 
    )
    {
        DLIB_ASSERT(col >= 0 && col < m.nc(), 
            "\tassignable_matrix_expression set_colm(matrix& m, col)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tcol:    " << col 
            );


        return assignable_col_matrix<T,NR,NC,mm,l>(m,col);
    }

// ----------------------------------------------------------------------------------------


    template <typename T, long NR, long NC, typename mm, typename l>
    class assignable_row_matrix
    {
    public:
        typedef T type;
        typedef l layout_type;
        typedef matrix<T,NR,NC,mm,l> matrix_type;

        assignable_row_matrix(
            matrix<T,NR,NC,mm,l>& m_,
            const long row_ 
        ) : m(m_), row(row_) {}


        T& operator() (
            long r,
            long c
        )
        {
            return m(row,c);
        }

        const T& operator() (
            long r,
            long c
        ) const
        {
            return m(row,c);
        }

        long nr() const { return 1; }
        long nc() const { return m.nc(); }


        template <typename EXP>
        assignable_row_matrix& operator= (
            const matrix_exp<EXP>& exp
        ) 
        {
            DLIB_ASSERT( exp.nr() == 1 && exp.nc() == m.nc(),
                "\tassignable_matrix_expression set_rowm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\tm.nc() (target matrix):   " << m.nc()
                );

            if (exp.destructively_aliases(m) == false)
            {
                matrix_assign(*this, exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to m to 
                // avoid aliasing issues during the copy
                this->operator=(tmp(exp));
            }

            return *this;
        }

        assignable_row_matrix& operator= (
            const T& value
        )
        {
            for (long i = 0; i < m.nc(); ++i)
            {
                m(row,i) = value;
            }

            return *this;
        }


        matrix<T,NR,NC,mm,l>& m;
        const long row;
    };


    template <typename T, long NR, long NC, typename mm, typename l>
    assignable_row_matrix<T,NR,NC,mm,l> set_rowm (
        matrix<T,NR,NC,mm,l>& m,
        const long row 
    )
    {
        DLIB_ASSERT(row >= 0 && row < m.nr(), 
            "\tassignable_matrix_expression set_rowm(matrix& m, row)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trow:    " << row 
            );


        return assignable_row_matrix<T,NR,NC,mm,l>(m,row);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_SUBEXP_

