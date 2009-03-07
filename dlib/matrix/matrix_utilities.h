// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_UTILITIES_
#define DLIB_MATRIx_UTILITIES_

#include "matrix_utilities_abstract.h"
#include "matrix.h"
#include <cmath>
#include <complex>
#include <limits>
#include "../pixel.h"
#include "../stl_checked.h"
#include <vector>
#include <algorithm>
#include "../std_allocator.h"
#include "matrix_expressions.h"



namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    void find_min_and_max (
        const matrix_exp<EXP>& m,
        typename EXP::type& min_val,
        typename EXP::type& max_val
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype find_min_and_max(const matrix_exp& m, min_val, max_val)"
            << "\n\tYou can't ask for the min and max of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        min_val = m(0,0);
        max_val = min_val;
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (temp > max_val)
                    max_val = temp;
                if (temp < min_val)
                    min_val = temp;
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    long index_of_max (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0 && (m.nr() == 1 || m.nc() == 1), 
            "\tlong index_of_max(const matrix_exp& m)"
            << "\n\tm must be a row or column matrix"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0);
        long best_idx = 0;
        for (long i = 1; i < m.size(); ++i)
        {
            type temp = m(i);
            if (temp > val)
            {
                val = temp;
                best_idx = i;
            }
        }
        return best_idx;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    long index_of_min (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0 && (m.nr() == 1 || m.nc() == 1), 
            "\tlong index_of_min(const matrix_exp& m)"
            << "\n\tm must be a row or column matrix"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0);
        long best_idx = 0;
        for (long i = 1; i < m.size(); ++i)
        {
            type temp = m(i);
            if (temp < val)
            {
                val = temp;
                best_idx = i;
            }
        }
        return best_idx;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type max (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype max(const matrix_exp& m)"
            << "\n\tYou can't ask for the max() of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (temp > val)
                    val = temp;
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type min (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype min(const matrix_exp& m)"
            << "\n\tYou can't ask for the min() of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (temp < val)
                    val = temp;
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type length (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.nr() == 1 || m.nc() == 1, 
            "\ttype length(const matrix_exp& m)"
            << "\n\tm must be a row or column vector"
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        return std::sqrt(sum(squared(m)));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type length_squared (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.nr() == 1 || m.nc() == 1, 
            "\ttype length_squared(const matrix_exp& m)"
            << "\n\tm must be a row or column vector"
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        return sum(squared(m));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    
    template <
        typename array_type
        >
    const typename enable_if<is_matrix<array_type>,array_type>::type& 
    array_to_matrix (
        const array_type& array
    )
    {
        return array;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename array_type
        >
    const typename disable_if<is_matrix<array_type>,matrix_array2d_exp<array_type> >::type 
    array_to_matrix (
        const array_type& array
    )
    {
        return matrix_array2d_exp<array_type>(array);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type
        >
    const typename disable_if<is_matrix<vector_type>,matrix_array_exp<vector_type> >::type 
    vector_to_matrix (
        const vector_type& vector
    )
    {
        typedef matrix_array_exp<vector_type> exp;
        return exp(vector);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename vector_type
        >
    const typename enable_if<is_matrix<vector_type>,vector_type>::type& vector_to_matrix (
        const vector_type& vector
    )
    /*!
        This overload catches the case where the argument to this function is
        already a matrix.
    !*/
    {
        return vector;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename value_type,
        typename alloc
        >
    const matrix_std_vector_exp<std::vector<value_type,alloc> > vector_to_matrix (
        const std::vector<value_type,alloc>& vector
    )
    {
        typedef matrix_std_vector_exp<std::vector<value_type,alloc> > exp;
        return exp(vector);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename value_type,
        typename alloc
        >
    const matrix_std_vector_exp<std_vector_c<value_type,alloc> > vector_to_matrix (
        const std_vector_c<value_type,alloc>& vector
    )
    {
        typedef matrix_std_vector_exp<std_vector_c<value_type,alloc> > exp;
        return exp(vector);
    }

// ----------------------------------------------------------------------------------------

    struct op_trans 
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = EXP::NC;
            const static long NC = EXP::NR;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return m(c,r); }

            template <typename M>
            static long nr (const M& m) { return m.nc(); }
            template <typename M>
            static long nc (const M& m) { return m.nr(); }
        }; 
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_trans> trans (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_trans> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    template <long R, long C>
    struct op_removerc
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+2;
            const static long NR = (EXP::NR==0) ? 0 : (EXP::NR - 1);
            const static long NC = (EXP::NC==0) ? 0 : (EXP::NC - 1);
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (r < R)
                {
                    if (c < C)
                        return m(r,c); 
                    else
                        return m(r,c+1); 
                }
                else
                {
                    if (c < C)
                        return m(r+1,c); 
                    else
                        return m(r+1,c+1); 
                }
            }

            template <typename M>
            static long nr (const M& m) { return m.nr() - 1; }
            template <typename M>
            static long nc (const M& m) { return m.nc() - 1; }
        };
    };

    struct op_removerc2
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+2;
            const static long NR = (EXP::NR==0) ? 0 : (EXP::NR - 1);
            const static long NC = (EXP::NC==0) ? 0 : (EXP::NC - 1);
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long R, long C, long r, long c)
            { 
                if (r < R)
                {
                    if (c < C)
                        return m(r,c); 
                    else
                        return m(r,c+1); 
                }
                else
                {
                    if (c < C)
                        return m(r+1,c); 
                    else
                        return m(r+1,c+1); 
                }
            }

            template <typename M, typename S>
            static long nr (const M& m, S&, S&) { return m.nr() - 1; }
            template <typename M, typename S>
            static long nc (const M& m, S&, S&) { return m.nc() - 1; }
        };
    };

    template <
        long R,
        long C,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_removerc<R,C> > removerc (
        const matrix_exp<EXP>& m
    )
    {
        // you can't remove a row from a matrix with only one row
        COMPILE_TIME_ASSERT((EXP::NR > R && R >= 0) || EXP::NR == 0);
        // you can't remove a column from a matrix with only one column 
        COMPILE_TIME_ASSERT((EXP::NC > C && C >= 0) || EXP::NR == 0);
        DLIB_ASSERT(m.nr() > R && R >= 0 && m.nc() > C && C >= 0, 
            "\tconst matrix_exp removerc<R,C>(const matrix_exp& m)"
            << "\n\tYou can't remove a row/column from a matrix if it doesn't have that row/column"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tR:      " << R 
            << "\n\tC:      " << C 
            );
        typedef matrix_unary_exp<EXP,op_removerc<R,C> > exp;
        return exp(m.ref());
    }

    template <
        typename EXP
        >
    const matrix_scalar_ternary_exp<EXP,long,op_removerc2>  removerc (
        const matrix_exp<EXP>& m,
        long R,
        long C
    )
    {
        DLIB_ASSERT(m.nr() > R && R >= 0 && m.nc() > C && C >= 0, 
            "\tconst matrix_exp removerc(const matrix_exp& m,R,C)"
            << "\n\tYou can't remove a row/column from a matrix if it doesn't have that row/column"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tR:      " << R 
            << "\n\tC:      " << C 
            );
        typedef matrix_scalar_ternary_exp<EXP,long,op_removerc2 > exp;
        return exp(m.ref(),R,C);
    }

// ----------------------------------------------------------------------------------------

    template <long C>
    struct op_remove_col
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+1;
            const static long NR = EXP::NR;
            const static long NC = (EXP::NC==0) ? 0 : (EXP::NC - 1);
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (c < C)
                {
                    return m(r,c); 
                }
                else
                {
                    return m(r,c+1); 
                }
            }

            template <typename M>
            static long nr (const M& m) { return m.nr(); }
            template <typename M>
            static long nc (const M& m) { return m.nc() - 1; }
        };
    };

    struct op_remove_col2
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+1;
            const static long NR = EXP::NR;
            const static long NC = (EXP::NC==0) ? 0 : (EXP::NC - 1);
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long C, long r, long c)
            { 
                if (c < C)
                {
                    return m(r,c); 
                }
                else
                {
                    return m(r,c+1); 
                }
            }

            template <typename M>
            static long nr (const M& m) { return m.nr(); }
            template <typename M>
            static long nc (const M& m) { return m.nc() - 1; }
        };
    };

    template <
        long C,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_remove_col<C> > remove_col (
        const matrix_exp<EXP>& m
    )
    {
        // You can't remove the given column from the matrix because the matrix doesn't
        // have a column with that index.
        COMPILE_TIME_ASSERT((EXP::NC > C && C >= 0) || EXP::NC == 0);
        DLIB_ASSERT(m.nc() > C && C >= 0 , 
            "\tconst matrix_exp remove_col<C>(const matrix_exp& m)"
            << "\n\tYou can't remove a col from a matrix if it doesn't have it"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tC:      " << C 
            );
        typedef matrix_unary_exp<EXP,op_remove_col<C> > exp;
        return exp(m.ref());
    }

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,long,op_remove_col2> remove_col (
        const matrix_exp<EXP>& m,
        long C
    )
    {
        DLIB_ASSERT(m.nc() > C && C >= 0 , 
            "\tconst matrix_exp remove_col(const matrix_exp& m,C)"
            << "\n\tYou can't remove a col from a matrix if it doesn't have it"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tC:      " << C 
            );
        typedef matrix_scalar_binary_exp<EXP,long,op_remove_col2> exp;
        return exp(m.ref(),C);
    }

// ----------------------------------------------------------------------------------------

    template <long R>
    struct op_remove_row
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+1;
            const static long NR = (EXP::NR==0) ? 0 : (EXP::NR - 1);
            const static long NC = EXP::NC;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (r < R)
                {
                    return m(r,c); 
                }
                else
                {
                    return m(r+1,c); 
                }
            }

            template <typename M>
            static long nr (const M& m) { return m.nr() - 1; }
            template <typename M>
            static long nc (const M& m) { return m.nc(); }
        };
    };

    struct op_remove_row2
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+1;
            const static long NR = (EXP::NR==0) ? 0 : (EXP::NR - 1);
            const static long NC = EXP::NC;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long R, long r, long c)
            { 
                if (r < R)
                {
                    return m(r,c); 
                }
                else
                {
                    return m(r+1,c); 
                }
            }

            template <typename M>
            static long nr (const M& m) { return m.nr() - 1; }
            template <typename M>
            static long nc (const M& m) { return m.nc(); }
        };
    };

    template <
        long R,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_remove_row<R> > remove_row (
        const matrix_exp<EXP>& m
    )
    {
        // You can't remove the given row from the matrix because the matrix doesn't
        // have a row with that index.
        COMPILE_TIME_ASSERT((EXP::NR > R && R >= 0) || EXP::NR == 0);
        DLIB_ASSERT(m.nr() > R && R >= 0, 
            "\tconst matrix_exp remove_row<R>(const matrix_exp& m)"
            << "\n\tYou can't remove a row from a matrix if it doesn't have it"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tR:      " << R 
            );
        typedef matrix_unary_exp<EXP,op_remove_row<R> > exp;
        return exp(m.ref());
    }

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,long,op_remove_row2> remove_row (
        const matrix_exp<EXP>& m,
        long R
    )
    {
        DLIB_ASSERT(m.nr() > R && R >= 0, 
            "\tconst matrix_exp remove_row(const matrix_exp& m, long R)"
            << "\n\tYou can't remove a row from a matrix if it doesn't have it"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tR:      " << R 
            );
        typedef matrix_scalar_binary_exp<EXP,long,op_remove_row2 > exp;
        return exp(m.ref(),R);
    }

// ----------------------------------------------------------------------------------------

    struct op_diagm
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost+1;
            const static long N = EXP::NC*EXP::NR;
            const static long NR = N;
            const static long NC = N;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (r==c)
                    return m(r); 
                else
                    return 0;
            }

            template <typename M>
            static long nr (const M& m) { return (m.nr()>m.nc())? m.nr():m.nc(); }
            template <typename M>
            static long nc (const M& m) { return (m.nr()>m.nc())? m.nr():m.nc(); }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_diagm> diagm (
        const matrix_exp<EXP>& m
    )
    {
        // You can only make a diagonal matrix out of a row or column vector
        COMPILE_TIME_ASSERT(EXP::NR == 0 || EXP::NR == 1 || EXP::NC == 1 || EXP::NC == 0);
        DLIB_ASSERT(m.nr() == 1 || m.nc() == 1, 
            "\tconst matrix_exp diagm(const matrix_exp& m)"
            << "\n\tYou can only apply diagm() to a row or column matrix"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            );
        typedef matrix_unary_exp<EXP,op_diagm> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_diag
    {
        template <typename EXP>
        struct op : has_destructive_aliasing
        {
            const static long cost = EXP::cost;
            const static long NR = (EXP::NC&&EXP::NR)? (tmin<EXP::NR,EXP::NC>::value) : (0);
            const static long NC = 1;
            typedef typename EXP::type type;
            typedef typename EXP::mem_manager_type mem_manager_type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return m(r,r); }

            template <typename M>
            static long nr (const M& m) { return std::min(m.nc(),m.nr()); }
            template <typename M>
            static long nc (const M& m) { return 1; }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_diag> diag (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_diag> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    template <typename target_type>
    struct op_cast
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost;
            typedef target_type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return static_cast<target_type>(m(r,c)); }
        };
    };

    template <
        typename target_type,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_cast<target_type> > matrix_cast (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_cast<target_type> > exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long NR,
        long NC,
        typename MM,
        typename U,
        typename L
        >
    typename disable_if<is_matrix<U>,void>::type set_all_elements (
        matrix<T,NR,NC,MM,L>& m,
        const U& value
    )
    {
        // The value you are trying to assign to each element of the m matrix
        // doesn't have the appropriate type.
        COMPILE_TIME_ASSERT(is_matrix<T>::value == is_matrix<U>::value);

        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                m(r,c) = static_cast<T>(value);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long NR,
        long NC,
        typename MM,
        typename U,
        typename L
        >
    typename enable_if<is_matrix<U>,void>::type set_all_elements (
        matrix<T,NR,NC,MM,L>& m,
        const U& value
    )
    {
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                m(r,c) = value;
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    inline const typename matrix_exp<EXP>::matrix_type tmp (
        const matrix_exp<EXP>& m
    )
    {
        return typename matrix_exp<EXP>::matrix_type (m);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename lazy_disable_if<is_matrix<typename EXP::type>, EXP>::type sum (
        const matrix_exp<EXP>& m
    )
    {
        typedef typename matrix_exp<EXP>::type type;

        type val = 0;
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                val += m(r,c);
            }
        }
        return val;
    }

    template <
        typename EXP
        >
    const typename lazy_enable_if<is_matrix<typename EXP::type>, EXP>::type sum (
        const matrix_exp<EXP>& m
    )
    {
        typedef typename matrix_exp<EXP>::type type;

        type val;
        if (m.size() > 0)
            val.set_size(m(0,0).nr(),m(0,0).nc()); 
        set_all_elements(val,0);

        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                val += m(r,c);
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    inline const typename matrix_exp<EXP>::type mean (
        const matrix_exp<EXP>& m
    )
    {
        return sum(m)/(m.nr()*m.nc());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename lazy_disable_if<is_matrix<typename EXP::type>, EXP>::type variance (
        const matrix_exp<EXP>& m
    )
    {
        const typename matrix_exp<EXP>::type avg = mean(m);

        typedef typename matrix_exp<EXP>::type type;

        type val = 0;
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                val += std::pow(m(r,c) - avg,2);
            }
        }

        if (m.nr() * m.nc() == 1)
            return val;
        else
            return val/(m.nr()*m.nc() - 1);
    }

    template <
        typename EXP
        >
    const typename lazy_enable_if<is_matrix<typename EXP::type>, EXP >::type variance (
        const matrix_exp<EXP>& m
    )
    {
        const typename matrix_exp<EXP>::type avg = mean(m);

        typedef typename matrix_exp<EXP>::type type;

        type val;
        if (m.size() > 0)
            val.set_size(m(0,0).nr(), m(0,0).nc());

        set_all_elements(val,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                val += pow(m(r,c) - avg,2);
            }
        }

        if (m.nr() * m.nc() <= 1)
            return val;
        else
            return val/(m.nr()*m.nc() - 1);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const matrix<typename EXP::type::type, EXP::type::NR, EXP::type::NR, typename EXP::mem_manager_type> covariance (
        const matrix_exp<EXP>& m
    )
    {
        // perform static checks to make sure m is a column vector 
        COMPILE_TIME_ASSERT(EXP::NR == 0 || EXP::NR > 1);
        COMPILE_TIME_ASSERT(EXP::NC == 1 || EXP::NC == 0);

        // perform static checks to make sure the matrices contained in m are column vectors
        COMPILE_TIME_ASSERT(EXP::type::NC == 1 || EXP::type::NC == 0 );

        DLIB_ASSERT(m.nr() > 1 && m.nc() == 1, 
            "\tconst matrix covariance(const matrix_exp& m)"
            << "\n\tYou can only apply covariance() to a column matrix"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            );
#ifdef ENABLE_ASSERTS
        for (long i = 0; i < m.nr(); ++i)
        {
            DLIB_ASSERT(m(0).nr() == m(i).nr() && m(i).nr() > 0 && m(i).nc() == 1, 
                   "\tconst matrix covariance(const matrix_exp& m)"
                   << "\n\tYou can only apply covariance() to a column matrix of column matrices"
                   << "\n\tm(0).nr(): " << m(0).nr()
                   << "\n\tm(i).nr(): " << m(i).nr() 
                   << "\n\tm(i).nc(): " << m(i).nc() 
                   << "\n\ti:         " << i 
                );
        }
#endif

        // now perform the actual calculation of the covariance matrix.
        matrix<typename EXP::type::type, EXP::type::NR, EXP::type::NR, typename EXP::mem_manager_type> cov(m(0).nr(),m(0).nr());
        set_all_elements(cov,0);

        const matrix<double,EXP::type::NR,EXP::type::NC, typename EXP::mem_manager_type> avg = mean(m);

        for (long r = 0; r < m.nr(); ++r)
        {
            cov += (m(r) - avg)*trans(m(r) - avg);
        }

        cov *= 1.0 / (m.nr() - 1.0);
        return cov;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type prod (
        const matrix_exp<EXP>& m
    )
    {
        typedef typename matrix_exp<EXP>::type type;

        type val = 1;
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                val *= m(r,c);
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T 
        >
    struct op_uniform_matrix_3 : has_nondestructive_aliasing 
    {
        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        typedef T type;
        static type apply (const T& val, long r, long c)
        { return val; }
    };

    template <
        typename T
        >
    const dynamic_matrix_scalar_unary_exp<T,op_uniform_matrix_3<T> > uniform_matrix (
        long nr,
        long nc,
        const T& val
    )
    {
        DLIB_ASSERT(nr > 0 && nc > 0, 
            "\tconst matrix_exp uniform_matrix<T>(nr, nc)"
            << "\n\tnr and nc have to be bigger than 0"
            << "\n\tnr: " << nr
            << "\n\tnc: " << nc
            );
        typedef dynamic_matrix_scalar_unary_exp<T,op_uniform_matrix_3<T> > exp;
        return exp(nr,nc,val);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        long NR_, 
        long NC_ 
        >
    struct op_uniform_matrix_2 : has_nondestructive_aliasing 
    {
        const static long cost = 1;
        const static long NR = NR_;
        const static long NC = NC_;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        typedef T type;
        static type apply (const T& val, long r, long c)
        { return val; }
    };

    template <
        typename T,
        long NR, 
        long NC
        >
    const matrix_scalar_unary_exp<T,op_uniform_matrix_2<T,NR,NC> > uniform_matrix (
        const T& val
    )
    {
        COMPILE_TIME_ASSERT(NR > 0 && NC > 0);

        typedef matrix_scalar_unary_exp<T,op_uniform_matrix_2<T,NR,NC> > exp;
        return exp(val);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        long NR_, 
        long NC_, 
        T val
        >
    struct op_uniform_matrix : has_nondestructive_aliasing
    {
        const static long cost = 1;
        const static long NR = NR_;
        const static long NC = NC_;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        typedef T type;
        static type apply ( long r, long c)
        { return val; }
    };

    template <
        typename T, 
        long NR, 
        long NC, 
        T val
        >
    const matrix_zeroary_exp<op_uniform_matrix<T,NR,NC,val> > uniform_matrix (
    )
    {
        COMPILE_TIME_ASSERT(NR > 0 && NC > 0);
        typedef matrix_zeroary_exp<op_uniform_matrix<T,NR,NC,val> > exp;
        return exp();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T 
        >
    struct op_identity_matrix_2 : has_nondestructive_aliasing 
    {
        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        typedef T type;
        static type apply (const T&, long r, long c)
        { return static_cast<type>(r == c); }
    };

    template <
        typename T
        >
    const dynamic_matrix_scalar_unary_exp<T,op_identity_matrix_2<T> > identity_matrix (
        const long& size 
    )
    {
        DLIB_ASSERT(size > 0, 
            "\tconst matrix_exp identity_matrix<T>(size)"
            << "\n\tsize must be bigger than 0"
            << "\n\tsize: " << size 
            );
        typedef dynamic_matrix_scalar_unary_exp<T,op_identity_matrix_2<T> > exp;
        // the scalar value of the dynamic_matrix_scalar_unary_exp just isn't
        // used by this operator
        return exp(size,size,0);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        long N
        >
    struct op_identity_matrix : has_nondestructive_aliasing
    {
        const static long cost = 1;
        const static long NR = N;
        const static long NC = N;
        typedef typename memory_manager<char>::kernel_1a mem_manager_type;
        typedef T type;
        static type apply ( long r, long c)
        { return static_cast<type>(r == c); }

        template <typename M>
        static long nr (const M&) { return NR; }
        template <typename M>
        static long nc (const M&) { return NC; }
    };

    template <
        typename T, 
        long N
        >
    const matrix_zeroary_exp<op_identity_matrix<T,N> > identity_matrix (
    )
    {
        COMPILE_TIME_ASSERT(N > 0);

        typedef matrix_zeroary_exp<op_identity_matrix<T,N> > exp;
        return exp();
    }

// ----------------------------------------------------------------------------------------

    template <long R, long C>
    struct op_rotate
    {
        template <typename EXP>
        struct op : has_destructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost + 1;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return m((r+R)%m.nr(),(c+C)%m.nc()); }
        };
    };

    template <
        long R,
        long C,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_rotate<R,C> > rotate (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_rotate<R,C> > exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_pointwise_multiply
    {
        template <typename EXP1, typename EXP2>
        struct op : public has_nondestructive_aliasing, public preserves_dimensions<EXP1,EXP2>
        {
            typedef typename EXP1::type type;
            const static long cost = EXP1::cost + EXP2::cost + 1;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& m2 , long r, long c)
            { return m1(r,c)*m2(r,c); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    inline const matrix_binary_exp<EXP1,EXP2,op_pointwise_multiply> pointwise_multiply (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc(), 
            "\tconst matrix_exp::type pointwise_multiply(const matrix_exp& a, const matrix_exp& b)"
            << "\n\tYou can only make a do a pointwise multiply with two equally sized matrices"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            );
        typedef matrix_binary_exp<EXP1,EXP2,op_pointwise_multiply> exp;
        return exp(a.ref(),b.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_pointwise_multiply3
    {
        template <typename EXP1, typename EXP2, typename EXP3>
        struct op : public has_nondestructive_aliasing, public preserves_dimensions<EXP1,EXP2,EXP3>
        {
            typedef typename EXP1::type type;
            const static long cost = EXP1::cost + EXP2::cost + EXP3::cost + 2;

            template <typename M1, typename M2, typename M3>
            static type apply ( const M1& m1, const M2& m2, const M3& m3 , long r, long c)
            { return m1(r,c)*m2(r,c)*m3(r,c); }
        };
    };

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3
        >
    inline const matrix_ternary_exp<EXP1,EXP2,EXP3,op_pointwise_multiply3> 
        pointwise_multiply (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b, 
        const matrix_exp<EXP3>& c
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,typename EXP3::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NR == 0 || EXP2::NC == 0);
        COMPILE_TIME_ASSERT(EXP2::NR == EXP3::NR || EXP2::NR == 0 || EXP3::NR == 0);
        COMPILE_TIME_ASSERT(EXP2::NC == EXP3::NC || EXP2::NC == 0 || EXP3::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc() &&
               b.nr() == c.nr() &&
               b.nc() == c.nc(), 
            "\tconst matrix_exp::type pointwise_multiply(a,b,c)"
            << "\n\tYou can only make a do a pointwise multiply between equally sized matrices"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            << "\n\tc.nr(): " << c.nr()
            << "\n\tc.nc(): " << c.nc() 
            );
        typedef matrix_ternary_exp<EXP1,EXP2,EXP3,op_pointwise_multiply3> exp; 

        return exp(a.ref(),b.ref(),c.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_pointwise_multiply4
    {
        template <typename EXP1, typename EXP2, typename EXP3, typename EXP4>
        struct op : public has_nondestructive_aliasing, public preserves_dimensions<EXP1,EXP2,EXP3,EXP4>
        {
            typedef typename EXP1::type type;
            const static long cost = EXP1::cost + EXP2::cost + EXP3::cost + EXP4::cost + 3;

            template <typename M1, typename M2, typename M3, typename M4>
            static type apply ( const M1& m1, const M2& m2, const M3& m3, const M4& m4 , long r, long c)
            { return m1(r,c)*m2(r,c)*m3(r,c)*m4(r,c); }
        };
    };

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3,
        typename EXP4
        >
    inline const matrix_fourary_exp<EXP1,EXP2,EXP3,EXP4,op_pointwise_multiply4> pointwise_multiply (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b, 
        const matrix_exp<EXP3>& c,
        const matrix_exp<EXP4>& d
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,typename EXP3::type>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXP3::type,typename EXP4::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0 );
        COMPILE_TIME_ASSERT(EXP2::NR == EXP3::NR || EXP2::NR == 0 || EXP3::NR == 0);
        COMPILE_TIME_ASSERT(EXP2::NC == EXP3::NC || EXP2::NC == 0 || EXP3::NC == 0);
        COMPILE_TIME_ASSERT(EXP3::NR == EXP4::NR || EXP3::NR == 0 || EXP4::NR == 0);
        COMPILE_TIME_ASSERT(EXP3::NC == EXP4::NC || EXP3::NC == 0 || EXP4::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc() &&
               b.nr() == c.nr() &&
               b.nc() == c.nc() &&
               c.nr() == d.nr() &&
               c.nc() == d.nc(), 
            "\tconst matrix_exp::type pointwise_multiply(a,b,c,d)"
            << "\n\tYou can only make a do a pointwise multiply between equally sized matrices"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            << "\n\tc.nr(): " << c.nr()
            << "\n\tc.nc(): " << c.nc() 
            << "\n\td.nr(): " << d.nr()
            << "\n\td.nc(): " << d.nc() 
            );

        typedef matrix_fourary_exp<EXP1,EXP2,EXP3,EXP4,op_pointwise_multiply4> exp;
        return exp(a.ref(),b.ref(),c.ref(),d.ref());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename P,
        int type = static_switch<
            pixel_traits<P>::grayscale,
            pixel_traits<P>::rgb,
            pixel_traits<P>::hsi,
            pixel_traits<P>::rgb_alpha
            >::value
        >
    struct pixel_to_vector_helper;

    template <typename P>
    struct pixel_to_vector_helper<P,1>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,2>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.red);
            m(1) = static_cast<typename M::type>(pixel.green);
            m(2) = static_cast<typename M::type>(pixel.blue);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,3>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.h);
            m(1) = static_cast<typename M::type>(pixel.s);
            m(2) = static_cast<typename M::type>(pixel.i);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,4>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.red);
            m(1) = static_cast<typename M::type>(pixel.green);
            m(2) = static_cast<typename M::type>(pixel.blue);
            m(3) = static_cast<typename M::type>(pixel.alpha);
        }
    };


    template <
        typename T,
        typename P
        >
    inline const matrix<T,pixel_traits<P>::num,1> pixel_to_vector (
        const P& pixel
    )
    {
        COMPILE_TIME_ASSERT(pixel_traits<P>::num > 0);
        matrix<T,pixel_traits<P>::num,1> m;
        pixel_to_vector_helper<P>::assign(m,pixel);
        return m;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename P,
        int type = static_switch<
            pixel_traits<P>::grayscale,
            pixel_traits<P>::rgb,
            pixel_traits<P>::hsi,
            pixel_traits<P>::rgb_alpha
            >::value
        >
    struct vector_to_pixel_helper;

    template <typename P>
    struct vector_to_pixel_helper<P,1>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel = static_cast<unsigned char>(m(0));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,2>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.red = static_cast<unsigned char>(m(0));
            pixel.green = static_cast<unsigned char>(m(1));
            pixel.blue = static_cast<unsigned char>(m(2));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,3>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.h = static_cast<unsigned char>(m(0));
            pixel.s = static_cast<unsigned char>(m(1));
            pixel.i = static_cast<unsigned char>(m(2));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,4>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.red = static_cast<unsigned char>(m(0));
            pixel.green = static_cast<unsigned char>(m(1));
            pixel.blue = static_cast<unsigned char>(m(2));
            pixel.alpha = static_cast<unsigned char>(m(3));
        }
    };

    template <
        typename P,
        typename EXP
        >
    inline void vector_to_pixel (
        P& pixel,
        const matrix_exp<EXP>& vector 
    )
    {
        COMPILE_TIME_ASSERT(pixel_traits<P>::num == matrix_exp<EXP>::NR);
        COMPILE_TIME_ASSERT(matrix_exp<EXP>::NC == 1);
        vector_to_pixel_helper<P>::assign(pixel,vector);
    }

// ----------------------------------------------------------------------------------------

    template <long lower, long upper>
    struct op_clamp
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            typedef typename EXP::type type;
            const static long cost = EXP::cost + 1;

            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                const type temp = m(r,c);
                if (temp > static_cast<type>(upper))
                    return static_cast<type>(upper);
                else if (temp < static_cast<type>(lower))
                    return static_cast<type>(lower);
                else
                    return temp;
            }
        };
    };

    template <
        long l, 
        long u,
        typename EXP
        >
    const matrix_unary_exp<EXP,op_clamp<l,u> > clamp (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_clamp<l,u> > exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP1,
        typename EXP2
        >
    bool equal (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b,
        const typename EXP1::type eps = 100*std::numeric_limits<typename EXP1::type>::epsilon()
    )
    {
        // check if the dimensions don't match
        if (a.nr() != b.nr() || a.nc() != b.nc())
            return false;

        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                if (std::abs(a(r,c)-b(r,c)) > eps)
                    return false;
            }
        }

        // no non-equal points found so we return true 
        return true;
    }

// ----------------------------------------------------------------------------------------

    struct op_scale_columns
    {
        template <typename EXP1, typename EXP2>
        struct op : has_nondestructive_aliasing
        {
            const static long cost = EXP1::cost + EXP2::cost + 1;
            typedef typename EXP1::type type;
            typedef typename EXP1::mem_manager_type mem_manager_type;
            const static long NR = EXP1::NR;
            const static long NC = EXP1::NC;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& m2 , long r, long c)
            { return m1(r,c)*m2(c); }

            template <typename M1, typename M2>
            static long nr (const M1& m1, const M2& ) { return m1.nr(); }
            template <typename M1, typename M2>
            static long nc (const M1& m1, const M2& ) { return m1.nc(); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_binary_exp<EXP1,EXP2,op_scale_columns> scale_columns (
        const matrix_exp<EXP1>& m,
        const matrix_exp<EXP2>& v 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT(EXP2::NC == 1 || EXP2::NC == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NR || EXP1::NC == 0 || EXP2::NR == 0);

        DLIB_ASSERT(v.nc() == 1 && v.nr() == m.nc(), 
            "\tconst matrix_exp scale_columns(m, v)"
            << "\n\tv must be a column vector and its length must match the number of columns in m"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tv.nr(): " << v.nr()
            << "\n\tv.nc(): " << v.nc() 
            );
        typedef matrix_binary_exp<EXP1,EXP2,op_scale_columns> exp;
        return exp(m.ref(),v.ref());
    }

// ----------------------------------------------------------------------------------------

    struct sort_columns_sort_helper
    {
        template <typename T>
        bool operator() (
            const T& item1,
            const T& item2
        ) const
        {
            return item1.first < item2.first;
        }
    };

    template <
        typename T, long NR, long NC, typename mm, typename l1,
        long NR2, long NC2, typename mm2, typename l2
        >
    void sort_columns (
        matrix<T,NR,NC,mm,l1>& m,
        matrix<T,NR2,NC2,mm2,l2>& v
    )
    {
        COMPILE_TIME_ASSERT(NC2 == 1 || NC2 == 0);
        COMPILE_TIME_ASSERT(NC == NR2 || NC == 0 || NR2 == 0);

        DLIB_ASSERT(v.nc() == 1 && v.nr() == m.nc(), 
            "\tconst matrix_exp sort_columns(m, v)"
            << "\n\tv must be a column vector and its length must match the number of columns in m"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tv.nr(): " << v.nr()
            << "\n\tv.nc(): " << v.nc() 
            );



        // Now we have to sort the given vectors in the m matrix according
        // to how big their corresponding v(column index) values are.
        typedef std::pair<T, matrix<T,0,1,mm> > col_pair;
        typedef std_allocator<col_pair, mm> alloc;
        std::vector<col_pair,alloc> colvalues;
        col_pair p;
        for (long r = 0; r < v.nr(); ++r)
        {
            p.first = v(r);
            p.second = colm(m,r);
            colvalues.push_back(p);
        }
        std::sort(colvalues.begin(), colvalues.end(), sort_columns_sort_helper());
        
        for (long i = 0; i < v.nr(); ++i)
        {
            v(i) = colvalues[i].first;
            set_colm(m,i) = colvalues[i].second;
        }

    }

// ----------------------------------------------------------------------------------------

    template <
        typename T, long NR, long NC, typename mm, typename l1,
        long NR2, long NC2, typename mm2, typename l2
        >
    void rsort_columns (
        matrix<T,NR,NC,mm,l1>& m,
        matrix<T,NR2,NC2,mm2,l2>& v
    )
    {
        COMPILE_TIME_ASSERT(NC2 == 1 || NC2 == 0);
        COMPILE_TIME_ASSERT(NC == NR2 || NC == 0 || NR2 == 0);

        DLIB_ASSERT(v.nc() == 1 && v.nr() == m.nc(), 
            "\tconst matrix_exp rsort_columns(m, v)"
            << "\n\tv must be a column vector and its length must match the number of columns in m"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tv.nr(): " << v.nr()
            << "\n\tv.nc(): " << v.nc() 
            );



        // Now we have to sort the given vectors in the m matrix according
        // to how big their corresponding v(column index) values are.
        typedef std::pair<T, matrix<T,0,1,mm> > col_pair;
        typedef std_allocator<col_pair, mm> alloc;
        std::vector<col_pair,alloc> colvalues;
        col_pair p;
        for (long r = 0; r < v.nr(); ++r)
        {
            p.first = v(r);
            p.second = colm(m,r);
            colvalues.push_back(p);
        }
        std::sort(colvalues.rbegin(), colvalues.rend(), sort_columns_sort_helper());
        
        for (long i = 0; i < v.nr(); ++i)
        {
            v(i) = colvalues[i].first;
            set_colm(m,i) = colvalues[i].second;
        }

    }

// ----------------------------------------------------------------------------------------

    struct op_tensor_product
    {
        template <typename EXP1, typename EXP2>
        struct op : public has_destructive_aliasing
        {
            const static long cost = EXP1::cost + EXP2::cost + 1;
            const static long NR = EXP1::NR*EXP2::NR;
            const static long NC = EXP1::NC*EXP2::NC;
            typedef typename EXP1::type type;
            typedef typename EXP1::mem_manager_type mem_manager_type;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& m2 , long r, long c)
            { 
                return m1(r/m2.nr(),c/m2.nc())*m2(r%m2.nr(),c%m2.nc()); 
            }


            template <typename M1, typename M2>
            static long nr (const M1& m1, const M2& m2 ) { return m1.nr()*m2.nr(); }
            template <typename M1, typename M2>
            static long nc (const M1& m1, const M2& m2 ) { return m1.nc()*m2.nc(); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    inline const matrix_binary_exp<EXP1,EXP2,op_tensor_product> tensor_product (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        typedef matrix_binary_exp<EXP1,EXP2,op_tensor_product> exp;
        return exp(a.ref(),b.ref());
    }

// ----------------------------------------------------------------------------------------


    struct op_lowerm
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+1;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (r >= c)
                    return m(r,c); 
                else
                    return 0;
            }

            template <typename M>
            static type apply ( const M& m, const type& s, long r, long c)
            { 
                if (r > c)
                    return m(r,c); 
                else if (r==c)
                    return s;
                else
                    return 0;
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_lowerm> lowerm (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_lowerm> exp;
        return exp(m.ref());
    }

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP, typename EXP::type,op_lowerm> lowerm (
        const matrix_exp<EXP>& m,
        typename EXP::type s
        )
    {
        typedef matrix_scalar_binary_exp<EXP, typename EXP::type, op_lowerm> exp;
        return exp(m.ref(),s);
    }

// ----------------------------------------------------------------------------------------

    struct op_upperm
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+1;
            typedef typename EXP::type type;

            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                if (r <= c)
                    return m(r,c); 
                else
                    return 0;
            }

            template <typename M>
            static type apply ( const M& m, const type& s, long r, long c)
            { 
                if (r < c)
                    return m(r,c); 
                else if (r==c)
                    return s;
                else
                    return 0;
            }
        };
    };


    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_upperm> upperm (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_upperm> exp;
        return exp(m.ref());
    }

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP, typename EXP::type,op_upperm> upperm (
        const matrix_exp<EXP>& m,
        typename EXP::type s
        )
    {
        typedef matrix_scalar_binary_exp<EXP, typename EXP::type ,op_upperm> exp;
        return exp(m.ref(),s);
    }

// ----------------------------------------------------------------------------------------

    template <typename rand_gen>
    inline const matrix<double> randm( 
        long nr,
        long nc,
        rand_gen& rnd
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0, 
            "\tconst matrix randm(nr, nc, rnd)"
            << "\n\tInvalid inputs to this function"
            << "\n\tnr: " << nr 
            << "\n\tnc: " << nc 
            );

        matrix<double> m(nr,nc);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                m(r,c) = rnd.get_random_double();
            }
        }

        return m;
    }

// ----------------------------------------------------------------------------------------

    inline const matrix<double> randm( 
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0, 
            "\tconst matrix randm(nr, nc)"
            << "\n\tInvalid inputs to this function"
            << "\n\tnr: " << nr 
            << "\n\tnc: " << nc 
            );

        matrix<double> m(nr,nc);
        // make a double that contains RAND_MAX + the smallest number that still
        // makes the resulting double slightly bigger than static_cast<double>(RAND_MAX)
        double max_val = RAND_MAX;
        max_val += std::numeric_limits<double>::epsilon()*RAND_MAX;

        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                m(r,c) = std::rand()/max_val;
            }
        }

        return m;
    }

// ----------------------------------------------------------------------------------------

    inline const matrix_range_exp<double> linspace (
        double start,
        double end,
        long num
    ) 
    { 
        DLIB_ASSERT(num >= 0, 
            "\tconst matrix_exp linspace(start, end, num)"
            << "\n\tInvalid inputs to this function"
            << "\n\tstart: " << start 
            << "\n\tend:   " << end
            << "\n\tnum:   " << num 
            );

        return matrix_range_exp<double>(start,end,num,false); 
    }

// ----------------------------------------------------------------------------------------
    
    inline const matrix_log_range_exp<double> logspace (
        double start,
        double end,
        long num
    ) 
    { 
        DLIB_ASSERT(num >= 0, 
            "\tconst matrix_exp logspace(start, end, num)"
            << "\n\tInvalid inputs to this function"
            << "\n\tstart: " << start 
            << "\n\tend:   " << end
            << "\n\tnum:   " << num 
            );

        return matrix_log_range_exp<double>(start,end,num); 
    }

// ----------------------------------------------------------------------------------------

    struct op_cart_prod
    {
        template <typename EXP1, typename EXP2>
        struct op : has_destructive_aliasing 
        {
            const static long cost = EXP1::cost+EXP2::cost+1;
            typedef typename EXP1::type type;

            typedef typename EXP1::mem_manager_type mem_manager_type;
            const static long NR = EXP1::NR+EXP2::NR;
            const static long NC = EXP1::NC*EXP2::NC;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& m2 , long r, long c)
            { 
                if (r < m1.nr())
                    return m1(r, c/m2.nc());
                else
                    return m2(r-m1.nr(), c%m2.nc());
            }

            template <typename M1, typename M2>
            static long nr (const M1& m1, const M2& m2) { return m1.nr() + m2.nr(); }
            template <typename M1, typename M2>
            static long nc (const M1& m1, const M2& m2) { return m1.nc() * m2.nc(); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_binary_exp<EXP1,EXP2,op_cart_prod> cartesian_product (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));

        typedef matrix_binary_exp<EXP1,EXP2,op_cart_prod> exp;
        return exp(a.ref(),b.ref());
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_UTILITIES_

