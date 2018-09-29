// Copyright (C) 2012 Massachusetts Institute of Technology, Lincoln Laboratory
// License: Boost Software License   See LICENSE.txt for the full license.
// Authors: Davis E. King (davis@dlib.net)
/*
                               READ THIS FIRST
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                               \############/
                                \##########/
                                 \########/
                                  \######/
                                   \####/
                                    \##/
                                     \/

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    See example_mex_function.cpp for a discussion of how to use the mex wrapper.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                                     /\
                                    /##\
                                   /####\
                                  /######\
                                 /########\
                                /##########\
                               /############\
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                                   ######
                               READ THIS FIRST
*/

// Copyright (C) 2012 Massachusetts Institute of Technology, Lincoln Laboratory
// License: Boost Software License   See LICENSE.txt for the full license.
// Authors: Davis E. King (davis@dlib.net)



































// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                            BEGIN IMPLEMENTATION DETAILS
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

#include "../matrix.h"
#include "../array2d.h"
#include "../array.h"
#include "../image_transforms.h"
#include "../is_kind.h"
#include "../string.h"
#include "../any.h" // for sig_traits
#include "../hash.h"
#include <tuple>
#include <map>

#if defined(_MSC_VER)
#define DLL_EXPORT_SYM __declspec(dllexport)
#endif
#include "mex.h"
#include <sstream>
#include "call_matlab.h"

// ----------------------------------------------------------------------------------------

#ifdef ARG_1_DEFAULT 
#define ELSE_ASSIGN_ARG_1 else A1 = ARG_1_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_1
#endif

#ifdef ARG_2_DEFAULT 
#define ELSE_ASSIGN_ARG_2 else A2 = ARG_2_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_2
#endif

#ifdef ARG_3_DEFAULT 
#define ELSE_ASSIGN_ARG_3 else A3 = ARG_3_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_3
#endif

#ifdef ARG_4_DEFAULT 
#define ELSE_ASSIGN_ARG_4 else A4 = ARG_4_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_4
#endif

#ifdef ARG_5_DEFAULT 
#define ELSE_ASSIGN_ARG_5 else A5 = ARG_5_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_5
#endif

#ifdef ARG_6_DEFAULT 
#define ELSE_ASSIGN_ARG_6 else A6 = ARG_6_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_6
#endif

#ifdef ARG_7_DEFAULT 
#define ELSE_ASSIGN_ARG_7 else A7 = ARG_7_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_7
#endif

#ifdef ARG_8_DEFAULT 
#define ELSE_ASSIGN_ARG_8 else A8 = ARG_8_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_8
#endif

#ifdef ARG_9_DEFAULT 
#define ELSE_ASSIGN_ARG_9 else A9 = ARG_9_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_9
#endif

#ifdef ARG_10_DEFAULT 
#define ELSE_ASSIGN_ARG_10 else A10 = ARG_10_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_10
#endif

#ifdef ARG_11_DEFAULT 
#define ELSE_ASSIGN_ARG_11 else A11 = ARG_11_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_11
#endif

#ifdef ARG_12_DEFAULT 
#define ELSE_ASSIGN_ARG_12 else A12 = ARG_12_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_12
#endif

#ifdef ARG_13_DEFAULT 
#define ELSE_ASSIGN_ARG_13 else A13 = ARG_13_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_13
#endif

#ifdef ARG_14_DEFAULT 
#define ELSE_ASSIGN_ARG_14 else A14 = ARG_14_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_14
#endif

#ifdef ARG_15_DEFAULT 
#define ELSE_ASSIGN_ARG_15 else A15 = ARG_15_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_15
#endif

#ifdef ARG_16_DEFAULT 
#define ELSE_ASSIGN_ARG_16 else A16 = ARG_16_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_16
#endif

#ifdef ARG_17_DEFAULT 
#define ELSE_ASSIGN_ARG_17 else A17 = ARG_17_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_17
#endif

#ifdef ARG_18_DEFAULT 
#define ELSE_ASSIGN_ARG_18 else A18 = ARG_18_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_18
#endif

#ifdef ARG_19_DEFAULT 
#define ELSE_ASSIGN_ARG_19 else A19 = ARG_19_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_19
#endif

#ifdef ARG_20_DEFAULT 
#define ELSE_ASSIGN_ARG_20 else A20 = ARG_20_DEFAULT;
#else
#define ELSE_ASSIGN_ARG_20
#endif

// ----------------------------------------------------------------------------------------

namespace mex_binding
{
    using namespace dlib;

    template <typename T>
    struct is_input_type 
    {
        const static unsigned long value = (!is_same_type<void,T>::value && (!is_reference_type<T>::value || is_const_type<T>::value )) ? 1 : 0;
    };
    template <typename T>
    struct is_output_type 
    {
        const static unsigned long value = (!is_same_type<void,T>::value && is_reference_type<T>::value && !is_const_type<T>::value) ? 1 : 0;
    };


    template <typename funct>
    struct funct_traits
    {
        const static unsigned long num_inputs = is_input_type<typename sig_traits<funct>::arg1_type>::value +
            is_input_type<typename sig_traits<funct>::arg2_type>::value +
            is_input_type<typename sig_traits<funct>::arg3_type>::value +
            is_input_type<typename sig_traits<funct>::arg4_type>::value +
            is_input_type<typename sig_traits<funct>::arg5_type>::value +
            is_input_type<typename sig_traits<funct>::arg6_type>::value +
            is_input_type<typename sig_traits<funct>::arg7_type>::value +
            is_input_type<typename sig_traits<funct>::arg8_type>::value +
            is_input_type<typename sig_traits<funct>::arg9_type>::value +
            is_input_type<typename sig_traits<funct>::arg10_type>::value + 
            is_input_type<typename sig_traits<funct>::arg11_type>::value + 
            is_input_type<typename sig_traits<funct>::arg12_type>::value + 
            is_input_type<typename sig_traits<funct>::arg13_type>::value + 
            is_input_type<typename sig_traits<funct>::arg14_type>::value + 
            is_input_type<typename sig_traits<funct>::arg15_type>::value + 
            is_input_type<typename sig_traits<funct>::arg16_type>::value + 
            is_input_type<typename sig_traits<funct>::arg17_type>::value + 
            is_input_type<typename sig_traits<funct>::arg18_type>::value + 
            is_input_type<typename sig_traits<funct>::arg19_type>::value + 
            is_input_type<typename sig_traits<funct>::arg20_type>::value; 

        const static unsigned long num_outputs= is_output_type<typename sig_traits<funct>::arg1_type>::value +
            is_output_type<typename sig_traits<funct>::arg2_type>::value +
            is_output_type<typename sig_traits<funct>::arg3_type>::value +
            is_output_type<typename sig_traits<funct>::arg4_type>::value +
            is_output_type<typename sig_traits<funct>::arg5_type>::value +
            is_output_type<typename sig_traits<funct>::arg6_type>::value +
            is_output_type<typename sig_traits<funct>::arg7_type>::value +
            is_output_type<typename sig_traits<funct>::arg8_type>::value +
            is_output_type<typename sig_traits<funct>::arg9_type>::value +
            is_output_type<typename sig_traits<funct>::arg10_type>::value + 
            is_output_type<typename sig_traits<funct>::arg11_type>::value + 
            is_output_type<typename sig_traits<funct>::arg12_type>::value + 
            is_output_type<typename sig_traits<funct>::arg13_type>::value + 
            is_output_type<typename sig_traits<funct>::arg14_type>::value + 
            is_output_type<typename sig_traits<funct>::arg15_type>::value + 
            is_output_type<typename sig_traits<funct>::arg16_type>::value + 
            is_output_type<typename sig_traits<funct>::arg17_type>::value + 
            is_output_type<typename sig_traits<funct>::arg18_type>::value + 
            is_output_type<typename sig_traits<funct>::arg19_type>::value + 
            is_output_type<typename sig_traits<funct>::arg20_type>::value; 
    };

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct is_array_type
    {
        // true if T is std::vector or dlib::array
        const static bool value = is_std_vector<T>::value || dlib::is_array<T>::value;

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename enabled = void 
        >
    struct inner_type
    {
        typedef T type;
    };

    template < typename T>
    struct inner_type<T, typename dlib::enable_if_c<is_matrix<T>::value || is_array2d<T>::value || dlib::is_array<T>::value >::type>
    {
        typedef typename T::type type;
    };

    template < typename T>
    struct inner_type<T, typename dlib::enable_if<is_std_vector<T> >::type>
    {
        typedef typename T::value_type type;
    };


// -------------------------------------------------------

    struct user_hit_ctrl_c {};

// -------------------------------------------------------

    template <typename T>
    void validate_and_populate_arg (
        long arg_idx,
        const mxArray *prhs,
        T& arg
    ); 

    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        matlab_struct& arg
    );

    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        matlab_object& arg
    );

    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        std::string& arg
    );

// -------------------------------------------------------

    template <typename T>
    struct is_column_major_matrix : public default_is_kind_value {};

    template <
        typename T,
        long num_rows,
        long num_cols,
        typename mem_manager
        >
    struct is_column_major_matrix<matrix<T,num_rows,num_cols,mem_manager,column_major_layout> > 
    { static const bool value = true; }; 

// -------------------------------------------------------

    string escape_percent(const string& str)
    {
        string temp;
        for(auto c : str)
        {
            if (c != '%')
            {
                temp += c;
            }
            else
            {
                temp += c;
                temp += c;
            }
        }
        return temp;
    }

    string escape_percent(const std::ostringstream& sout)
    {
        return escape_percent(sout.str());
    }

// -------------------------------------------------------

    template <
        typename matrix_type
        >
    typename dlib::enable_if_c<is_matrix<matrix_type>::value || is_array2d<matrix_type>::value >::type
    clear_mat (
        matrix_type& m
    )  
    {
        m.set_size(0,0);
    }

    template <
        typename matrix_type
        >
    typename dlib::disable_if_c<is_matrix<matrix_type>::value || is_array2d<matrix_type>::value >::type
    clear_mat (
        matrix_type& 
    )  
    {
    }

// -------------------------------------------------------

    template <
        typename matrix_type,
        typename EXP
        >
    typename dlib::enable_if_c<is_matrix<matrix_type>::value && is_same_type<typename inner_type<matrix_type>::type,typename EXP::type>::value >::type
    assign_mat (
        const long arg_idx,
        matrix_type& m,
        const matrix_exp<EXP>& src
    )  
    {
        if (matrix_type::NR != 0 && matrix_type::NR != src.nc())
        {
            std::ostringstream sout;
            sout << "Argument " << arg_idx+1 << " expects a matrix with " << matrix_type::NR << " rows but got one with " << src.nc();
            throw invalid_args_exception(sout);
        }
        if (matrix_type::NC != 0 && matrix_type::NC != src.nr())
        {
            std::ostringstream sout;
            sout << "Argument " << arg_idx+1 << " expects a matrix with " << matrix_type::NC << " columns but got one with " << src.nr();
            throw invalid_args_exception(sout);
        }


        m = trans(src);
    }

    template <
        typename matrix_type,
        typename EXP
        >
    typename dlib::enable_if_c<is_array2d<matrix_type>::value && is_same_type<typename inner_type<matrix_type>::type,typename EXP::type>::value >::type
    assign_mat (
        const long arg_idx,
        matrix_type& m,
        const matrix_exp<EXP>& src
    )  
    {
        assign_image(m , trans(src));
    }

    template <
        typename matrix_type,
        typename EXP
        >
    typename disable_if_c<(is_array2d<matrix_type>::value || is_matrix<matrix_type>::value) && 
    is_same_type<typename inner_type<matrix_type>::type,typename EXP::type>::value >::type
    assign_mat (
        const long arg_idx,
        matrix_type& ,
        const matrix_exp<EXP>& 
    ) 
    {
        std::ostringstream sout;
        sout << "mex_function has some bug in it related to processing input argument " << arg_idx+1;
        throw invalid_args_exception(sout);
    }


// -------------------------------------------------------

    template <
        typename T,
        typename U
        >
    typename dlib::enable_if_c<is_built_in_scalar_type<T>::value || is_same_type<T,bool>::value >::type
    assign_scalar (
        const long arg_idx,
        T& dest,
        const U& src
    )  
    {
        if (is_signed_type<U>::value && src < 0 && is_unsigned_type<T>::value)
        {
            std::ostringstream sout;
            sout << "Error, input argument " << arg_idx+1 << " must be a non-negative number.";
            throw invalid_args_exception(sout);
        }
        else
        {
            dest = src;
        }
    }

    template <
        typename T,
        typename U
        >
    typename dlib::disable_if_c<is_built_in_scalar_type<T>::value || is_same_type<T,bool>::value >::type
    assign_scalar (
        const long arg_idx,
        T& ,
        const U& 
    )  
    {
        std::ostringstream sout;
        sout << "mex_function has some bug in it related to processing input argument " << arg_idx+1;
        throw invalid_args_exception(sout);
    }


// -------------------------------------------------------

    void assign_function_handle (
        const long arg_idx,
        function_handle& dest,
        const mxArray* src
    )  
    {
        const_cast<void*&>(dest.h) = (void*)src;
    }

    template <
        typename T
        >
    void assign_function_handle (
        const long arg_idx,
        T& ,
        const mxArray* 
    )  
    {
        std::ostringstream sout;
        sout << "mex_function has some bug in it related to processing input argument " << arg_idx+1;
        throw invalid_args_exception(sout);
    }


// -------------------------------------------------------

    template <
        typename T
        >
    typename dlib::enable_if<is_array_type<T> >::type
    assign_std_vector (
        const long arg_idx,
        T& dest,
        const mxArray* src
    )  
    {
        const long nr = mxGetM(src);
        const long nc = mxGetN(src);

        typedef typename inner_type<T>::type type;

        if (!mxIsCell(src))
        {
            std::ostringstream sout;
            sout << "Input argument " << arg_idx+1 << " must be a cell array";
            throw invalid_args_exception(sout);
        }
        if (nr != 1 && nc != 1)
        {
            std::ostringstream sout;
            sout << "Input argument " << arg_idx+1 << " must be a cell array with exactly 1 row or 1 column (i.e. a row or column vector)";
            throw invalid_args_exception(sout);
        }

        const long size = nr*nc;
        dest.resize(size);

        for (unsigned long i = 0; i < dest.size(); ++i)
        {
            try
            {
                validate_and_populate_arg(i, mxGetCell(src, i), dest[i]);
            }
            catch (invalid_args_exception& e)
            {
                std::ostringstream sout;
                sout << "Error in argument " << arg_idx+1 << ": element " << i+1 << " of cell array not the expected type.\n";
                sout << "\t" << e.what();
                throw invalid_args_exception(sout);
            }
        }

    }

    template <
        typename T
        >
    typename disable_if<is_array_type<T> >::type
    assign_std_vector (
        const long arg_idx,
        T& ,
        const mxArray*
    )  
    {
        std::ostringstream sout;
        sout << "mex_function has some bug in it related to processing input argument " << arg_idx+1;
        throw invalid_args_exception(sout);
    }

// -------------------------------------------------------

    template <typename T> 
    void assign_image (
        const long arg_idx,
        T&,
        const dlib::uint8* data,
        long nr,
        long nc
    )
    {
        std::ostringstream sout;
        sout << "mex_function has some bug in it related to processing input argument " << arg_idx+1;
        throw invalid_args_exception(sout);
    }

    template <typename MM>
    void assign_image(
        const long ,
        array2d<dlib::rgb_pixel,MM>& img,
        const dlib::uint8* data,
        long nr,
        long nc
    )
    {
        img.set_size(nr, nc);
        for (long c = 0; c < img.nc(); ++c)
            for (long r = 0; r < img.nr(); ++r)
                img[r][c].red = *data++;
        for (long c = 0; c < img.nc(); ++c)
            for (long r = 0; r < img.nr(); ++r)
                img[r][c].green = *data++;
        for (long c = 0; c < img.nc(); ++c)
            for (long r = 0; r < img.nr(); ++r)
                img[r][c].blue = *data++;
    }

// -------------------------------------------------------

    template <typename T>
    void call_private_set_mxArray(T&, mxArray*) {}
    void call_private_set_mxArray(matrix_colmajor& item, mxArray* m) { item._private_set_mxArray(m); }
    void call_private_set_mxArray(fmatrix_colmajor& item, mxArray* m) { item._private_set_mxArray(m); }

// -------------------------------------------------------

    template <typename T>
    void validate_and_populate_arg (
        long arg_idx,
        const mxArray *prhs,
        T& arg
    ) 
    {
        using namespace mex_binding;
        if (is_built_in_scalar_type<T>::value || is_same_type<T,bool>::value)
        {
            if( !(mxIsDouble(prhs) || mxIsSingle(prhs) || mxIsLogical(prhs) ) || 
                mxIsComplex(prhs) ||
                mxGetNumberOfElements(prhs)!=1 ) 
            {
                std::ostringstream sout;
                sout << "Input argument " << arg_idx+1 << " must be a scalar";
                throw invalid_args_exception(sout);
            }

            assign_scalar(arg_idx, arg , mxGetScalar(prhs));
        }
        else if (is_matrix<T>::value || is_array2d<T>::value)
        {
            if (prhs == NULL)
            {
                clear_mat(arg);
                return;
            }

            typedef typename inner_type<T>::type type;

            const int num_dims = mxGetNumberOfDimensions(prhs);
            const long nr = mxGetM(prhs);
            const long nc = mxGetN(prhs);

            if (is_same_type<type,dlib::rgb_pixel>::value)
            {
                if (!(num_dims == 3 && mxGetDimensions(prhs)[2] == 3 && mxIsUint8(prhs)))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a 3-D NxMx3 image matrix of uint8";
                    throw invalid_args_exception(sout);
                }

                const long rows = mxGetDimensions(prhs)[0];
                const long cols = mxGetDimensions(prhs)[1];
                assign_image(arg_idx, arg , (const dlib::uint8*)mxGetData(prhs), rows, cols);
                return;
            }

            if (num_dims != 2)
            {
                std::ostringstream sout;
                sout << "Input argument " << arg_idx+1 << " must be a 2-D matrix (got a " << num_dims << "-D matrix)";
                throw invalid_args_exception(sout);
            }


            if (is_same_type<type,double>::value)
            {
                if (!mxIsDouble(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of doubles";
                    throw invalid_args_exception(sout);
                }
                if (is_column_major_matrix<T>::value)
                    call_private_set_mxArray(arg, (mxArray*)prhs);
                else
                    assign_mat(arg_idx, arg , pointer_to_matrix(mxGetPr(prhs), nc, nr));
            }
            else if (is_same_type<type, float>::value)
            {
                if (!mxIsSingle(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of single/float";
                    throw invalid_args_exception(sout);
                }

                if (is_column_major_matrix<T>::value)
                    call_private_set_mxArray(arg,(mxArray*)prhs);
                else
                    assign_mat(arg_idx, arg , pointer_to_matrix((const float*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, bool>::value)
            {
                if (!mxIsLogical(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of logical elements.";
                    throw invalid_args_exception(sout);
                }
                DLIB_CASSERT(sizeof(mxLogical) == sizeof(bool),"logical matrices are not supported by the mex wrapper when mxLogical isn't a bool.");

                assign_mat(arg_idx, arg , pointer_to_matrix((const bool*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::uint8>::value)
            {
                if (!mxIsUint8(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of uint8";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const dlib::uint8*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::int8>::value)
            {
                if (!mxIsInt8(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of int8";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const dlib::int8*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::int16>::value ||
                    (is_same_type<type, short>::value && sizeof(short) == sizeof(dlib::int16)))
            {
                if (!mxIsInt16(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of int16";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::uint16>::value ||
                    (is_same_type<type, unsigned short>::value && sizeof(unsigned short) == sizeof(dlib::uint16)))
            {
                if (!mxIsUint16(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of uint16";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::int32>::value ||
                    (is_same_type<type, int>::value && sizeof(int) == sizeof(dlib::int32)) ||
                    (is_same_type<type, long>::value && sizeof(long) == sizeof(dlib::int32)))
            {
                if (!mxIsInt32(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of int32";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::uint32>::value ||
                    (is_same_type<type, unsigned int>::value && sizeof(unsigned int) == sizeof(dlib::uint32)) ||
                    (is_same_type<type, unsigned long>::value && sizeof(unsigned long) == sizeof(dlib::uint32)))
            {
                if (!mxIsUint32(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of uint32";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::uint64>::value ||
                    (is_same_type<type, unsigned int>::value && sizeof(unsigned int) == sizeof(dlib::uint64)) ||
                    (is_same_type<type, unsigned long>::value && sizeof(unsigned long) == sizeof(dlib::uint64)))
            {
                if (!mxIsUint64(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of uint64";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else if (is_same_type<type, dlib::int64>::value ||
                    (is_same_type<type, int>::value && sizeof(int) == sizeof(dlib::int64)) ||
                    (is_same_type<type, long>::value && sizeof(long) == sizeof(dlib::int64)))
            {
                if (!mxIsInt64(prhs) || mxIsComplex(prhs))
                {
                    std::ostringstream sout;
                    sout << "Input argument " << arg_idx+1 << " must be a matrix of int64";
                    throw invalid_args_exception(sout);
                }

                assign_mat(arg_idx, arg , pointer_to_matrix((const type*)mxGetData(prhs), nc, nr));
            }
            else
            {
                throw invalid_args_exception("mex_function uses unsupported matrix type");
            }
        }
        else if (is_array_type<T>::value)
        {
            assign_std_vector(arg_idx, arg, prhs);

        }
        else if (is_same_type<T,function_handle>::value)
        {
            if (!mxIsClass(prhs, "function_handle"))
            {
                std::ostringstream sout;
                sout << "Input argument " << arg_idx+1 << " must be a function handle.";
                throw invalid_args_exception(sout);
            }
            assign_function_handle(arg_idx, arg, prhs);
        }
        else
        {
            throw invalid_args_exception("mex_function uses unsupported input argument type");
        }
    }

    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        matlab_struct& arg
    )
    {
        if (!mxIsStruct(prhs))
        {
            std::ostringstream sout;
            sout << "Input argument " << arg_idx+1 << " must be a struct";
            throw invalid_args_exception(sout);
        }

        arg.set_struct_handle(arg_idx, prhs);
    }


    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        matlab_object& arg
    )
    {
        arg.set_object_handle(arg_idx, prhs);
    }


    void validate_and_populate_arg(
        long arg_idx,
        const mxArray *prhs,
        std::string& arg
    )
    {
        if (!mxIsChar(prhs))
        {
            std::ostringstream sout;
            sout << "Input argument " << arg_idx+1 << " must be a char string";
            throw invalid_args_exception(sout);
        }

        const long nr = mxGetM(prhs);
        const long nc = mxGetN(prhs);
        const long size = nr*nc;
        arg.resize(size+1);
        if (mxGetString(prhs, &arg[0], arg.size()))
        {
            std::ostringstream sout;
            sout << "Input argument " << arg_idx+1 << " encountered an error while calling mxGetString()";
            throw invalid_args_exception(sout);
        }
        arg.resize(size);
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    typename dlib::enable_if<is_same_type<dlib::rgb_pixel,typename EXP::type> >::type assign_image_to_matlab (
        dlib::uint8* mat,
        const matrix_exp<EXP>& item
    )
    {
        for (long c = 0; c < item.nc(); ++c)
            for (long r = 0; r < item.nr(); ++r)
                *mat++ = item(r,c).red;
        for (long c = 0; c < item.nc(); ++c)
            for (long r = 0; r < item.nr(); ++r)
                *mat++ = item(r,c).green;
        for (long c = 0; c < item.nc(); ++c)
            for (long r = 0; r < item.nr(); ++r)
                *mat++ = item(r,c).blue;
    }

    template <typename T, typename EXP>
    typename disable_if<is_same_type<dlib::rgb_pixel,typename EXP::type> >::type assign_image_to_matlab (
        T* mat,
        const matrix_exp<EXP>& 
    )
    {
        mexErrMsgIdAndTxt("mex_function:validate_and_populate_arg",
                          "mex_function uses unsupported output image argument type");
    }

    template <typename T>
    typename dlib::enable_if<is_matrix<T> >::type assign_to_matlab(
        mxArray*& plhs,
        const T& item
    ) 
    {
        typedef typename T::type type;

        type* mat = 0;

        if (is_same_type<double, type>::value)
        {
            plhs = mxCreateDoubleMatrix(item.nr(),
                                        item.nc(),
                                        mxREAL);

            mat = (type*)mxGetPr(plhs);
        }
        else if (is_same_type<float, type>::value )
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxSINGLE_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<bool, type>::value )
        {
            plhs = mxCreateLogicalMatrix(item.nr(),
                                         item.nc());

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::uint8, type>::value )
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxUINT8_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::int8, type>::value )
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxINT8_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::int16, type>::value ||
                 (is_same_type<short,type>::value && sizeof(short) == sizeof(dlib::int16)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxINT16_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::uint16, type>::value ||
                 (is_same_type<unsigned short,type>::value && sizeof(unsigned short) == sizeof(dlib::uint16)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxUINT16_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::int32, type>::value ||
                 (is_same_type<long,type>::value && sizeof(long) == sizeof(dlib::int32)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxINT32_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::uint32, type>::value ||
                 (is_same_type<unsigned long,type>::value && sizeof(unsigned long) == sizeof(dlib::uint32)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxUINT32_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::uint64, type>::value ||
                 (is_same_type<unsigned long,type>::value && sizeof(unsigned long) == sizeof(dlib::uint64)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxUINT64_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::int64, type>::value  || 
                 (is_same_type<long,type>::value && sizeof(long) == sizeof(dlib::int64)))
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxINT64_CLASS,
                                         mxREAL);

            mat = (type*)mxGetData(plhs);
        }
        else if (is_same_type<dlib::rgb_pixel, type>::value)
        {
            mwSize dims[3] = {(mwSize)item.nr(), (mwSize)item.nc(), 3};
            plhs = mxCreateNumericArray(3, dims, mxUINT8_CLASS, mxREAL);

            assign_image_to_matlab((dlib::uint8*)mxGetData(plhs), item);
            return;
        }
        else
        {
            mexErrMsgIdAndTxt("mex_function:validate_and_populate_arg",
                              "mex_function uses unsupported output argument type");
        }


        const_temp_matrix<T> m(item);

        for (long c = 0; c < m.nc(); ++c)
        {
            for ( long r = 0; r < m.nr(); ++r)
            {
                *mat++ = m(r,c);
            }
        }
    }

    void assign_to_matlab(
        mxArray*& plhs,
        matrix_colmajor& item
    )
    {
        if(item._private_is_owned_by_matlab())
        {
            // Don't need to do a copy if it's this kind of matrix since we can just
            // pull the underlying mxArray out directly and thus avoid a copy.
            plhs = item._private_release_mxArray();
            // If there isn't anything there because the matrix is empty then set it to an
            // empty matrix.
            if (!plhs)
                plhs = mxCreateDoubleMatrix(item.nr(),
                                            item.nc(),
                                            mxREAL);
        }
        else
        {
            plhs = mxCreateDoubleMatrix(item.nr(),
                                        item.nc(),
                                        mxREAL);
            if (item.size() != 0)
                memcpy(mxGetPr(plhs), &item(0,0), item.size()*sizeof(double));
        }
    }

    void assign_to_matlab(
        mxArray*& plhs,
        fmatrix_colmajor& item
    )
    {
        if(item._private_is_owned_by_matlab())
        {
            // Don't need to do a copy if it's this kind of matrix since we can just
            // pull the underlying mxArray out directly and thus avoid a copy.
            plhs = item._private_release_mxArray();
            // If there isn't anything there because the matrix is empty then set it to an
            // empty matrix.
            if (!plhs)
                plhs = mxCreateNumericMatrix(item.nr(),
                                            item.nc(),
                                            mxSINGLE_CLASS,
                                            mxREAL);
        }
        else
        {
            plhs = mxCreateNumericMatrix(item.nr(),
                                         item.nc(),
                                         mxSINGLE_CLASS,
                                         mxREAL);
            if (item.size() != 0)
                memcpy(mxGetPr(plhs), &item(0,0), item.size()*sizeof(float));
        }
    }

    void assign_to_matlab(
        mxArray*& plhs,
        matlab_struct& item
    )
    {
        plhs = (mxArray*)item.release_struct_to_matlab();
    }

    void assign_to_matlab(
        mxArray*& plhs,
        matlab_object& item
    )
    {
        plhs = (mxArray*)item.release_object_to_matlab();
    }

    void assign_to_matlab(
        mxArray*& plhs,
        const std::string& item
    ) 
    {
        plhs = mxCreateString(item.c_str());
    }

    template <typename T, typename MM>
    void assign_to_matlab(
        mxArray*& plhs,
        const array2d<T,MM>& item
    ) 
    {
        assign_to_matlab(plhs,array_to_matrix(item));
    }

    template <typename T>
    typename dlib::disable_if_c<is_matrix<T>::value || is_array_type<T>::value || 
                                is_same_type<T,function_handle>::value>::type assign_to_matlab(
        mxArray*& plhs,
        const T& item
    ) 
    {
        plhs = mxCreateDoubleScalar(item);
    }


    void assign_to_matlab (
        mxArray*& plhs,
        const char* str
    )
    {
        assign_to_matlab(plhs, std::string(str));
    }

    void assign_to_matlab(
        mxArray*& plhs,
        const function_handle& h
    )
    {
    }

    template <typename T>
    typename dlib::enable_if<is_array_type<T> >::type assign_to_matlab(
        mxArray*& plhs,
        const T& item
    ) 
    {
        mwSize dims[1] = {item.size()};
        plhs = mxCreateCellArray(1,dims);
        for (unsigned long i = 0; i < item.size(); ++i)
        {
            mxArray* next = 0;
            assign_to_matlab(next, item[i]);
            mxSetCell(plhs, i, next);
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    void mark_owned_by_matlab (const T&){}

    void mark_owned_by_matlab(matrix_colmajor& item) { item._private_mark_owned_by_matlab(); }
    void mark_owned_by_matlab(fmatrix_colmajor& item) { item._private_mark_owned_by_matlab(); }

// ----------------------------------------------------------------------------------------

    template <
        unsigned long num_args
        >
    struct call_mex_function_helper;

    template <>
    struct call_mex_function_helper<0>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int , mxArray **,
            int , const mxArray **
        ) const
        {
            f();
        }
    };

    template <>
    struct call_mex_function_helper<1>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;

            typename basic_type<arg1_type>::type A1;

            mark_owned_by_matlab(A1);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;

            f(A1);

            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<2>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;

            f(A1,A2);

            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<3>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;

            f(A1,A2,A3);

            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<4>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;

            f(A1,A2,A3,A4);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<5>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;

            f(A1,A2,A3,A4,A5);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
        }
    };


    template <>
    struct call_mex_function_helper<6>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;

            f(A1,A2,A3,A4,A5,A6);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
        }
    };


    template <>
    struct call_mex_function_helper<7>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;

            f(A1,A2,A3,A4,A5,A6,A7);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
        }
    };


    template <>
    struct call_mex_function_helper<8>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;

            f(A1,A2,A3,A4,A5,A6,A7,A8);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
        }
    };


    template <>
    struct call_mex_function_helper<9>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
        }
    };



    template <>
    struct call_mex_function_helper<10>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<11>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<12>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<13>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<14>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<15>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<16>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;
            typedef typename sig_traits<funct>::arg16_type arg16_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;
            typename basic_type<arg16_type>::type A16;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);
            mark_owned_by_matlab(A16);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;
            if (i < nrhs && is_input_type<arg16_type>::value) {validate_and_populate_arg(i,prhs[i],A16); ++i;} ELSE_ASSIGN_ARG_16;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
            if (is_output_type<arg16_type>::value) {assign_to_matlab(plhs[i],A16); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<17>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;
            typedef typename sig_traits<funct>::arg16_type arg16_type;
            typedef typename sig_traits<funct>::arg17_type arg17_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;
            typename basic_type<arg16_type>::type A16;
            typename basic_type<arg17_type>::type A17;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);
            mark_owned_by_matlab(A16);
            mark_owned_by_matlab(A17);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;
            if (i < nrhs && is_input_type<arg16_type>::value) {validate_and_populate_arg(i,prhs[i],A16); ++i;} ELSE_ASSIGN_ARG_16;
            if (i < nrhs && is_input_type<arg17_type>::value) {validate_and_populate_arg(i,prhs[i],A17); ++i;} ELSE_ASSIGN_ARG_17;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
            if (is_output_type<arg16_type>::value) {assign_to_matlab(plhs[i],A16); ++i;}
            if (is_output_type<arg17_type>::value) {assign_to_matlab(plhs[i],A17); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<18>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;
            typedef typename sig_traits<funct>::arg16_type arg16_type;
            typedef typename sig_traits<funct>::arg17_type arg17_type;
            typedef typename sig_traits<funct>::arg18_type arg18_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;
            typename basic_type<arg16_type>::type A16;
            typename basic_type<arg17_type>::type A17;
            typename basic_type<arg18_type>::type A18;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);
            mark_owned_by_matlab(A16);
            mark_owned_by_matlab(A17);
            mark_owned_by_matlab(A18);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;
            if (i < nrhs && is_input_type<arg16_type>::value) {validate_and_populate_arg(i,prhs[i],A16); ++i;} ELSE_ASSIGN_ARG_16;
            if (i < nrhs && is_input_type<arg17_type>::value) {validate_and_populate_arg(i,prhs[i],A17); ++i;} ELSE_ASSIGN_ARG_17;
            if (i < nrhs && is_input_type<arg18_type>::value) {validate_and_populate_arg(i,prhs[i],A18); ++i;} ELSE_ASSIGN_ARG_18;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
            if (is_output_type<arg16_type>::value) {assign_to_matlab(plhs[i],A16); ++i;}
            if (is_output_type<arg17_type>::value) {assign_to_matlab(plhs[i],A17); ++i;}
            if (is_output_type<arg18_type>::value) {assign_to_matlab(plhs[i],A18); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<19>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;
            typedef typename sig_traits<funct>::arg16_type arg16_type;
            typedef typename sig_traits<funct>::arg17_type arg17_type;
            typedef typename sig_traits<funct>::arg18_type arg18_type;
            typedef typename sig_traits<funct>::arg19_type arg19_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;
            typename basic_type<arg16_type>::type A16;
            typename basic_type<arg17_type>::type A17;
            typename basic_type<arg18_type>::type A18;
            typename basic_type<arg19_type>::type A19;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);
            mark_owned_by_matlab(A16);
            mark_owned_by_matlab(A17);
            mark_owned_by_matlab(A18);
            mark_owned_by_matlab(A19);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;
            if (i < nrhs && is_input_type<arg16_type>::value) {validate_and_populate_arg(i,prhs[i],A16); ++i;} ELSE_ASSIGN_ARG_16;
            if (i < nrhs && is_input_type<arg17_type>::value) {validate_and_populate_arg(i,prhs[i],A17); ++i;} ELSE_ASSIGN_ARG_17;
            if (i < nrhs && is_input_type<arg18_type>::value) {validate_and_populate_arg(i,prhs[i],A18); ++i;} ELSE_ASSIGN_ARG_18;
            if (i < nrhs && is_input_type<arg19_type>::value) {validate_and_populate_arg(i,prhs[i],A19); ++i;} ELSE_ASSIGN_ARG_19;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
            if (is_output_type<arg16_type>::value) {assign_to_matlab(plhs[i],A16); ++i;}
            if (is_output_type<arg17_type>::value) {assign_to_matlab(plhs[i],A17); ++i;}
            if (is_output_type<arg18_type>::value) {assign_to_matlab(plhs[i],A18); ++i;}
            if (is_output_type<arg19_type>::value) {assign_to_matlab(plhs[i],A19); ++i;}
        }
    };

    template <>
    struct call_mex_function_helper<20>
    {
        template <typename funct>
        void callit(
            const funct& f,
            int nlhs, mxArray *plhs[],
            int nrhs, const mxArray *prhs[]
        ) const
        {
            typedef typename sig_traits<funct>::arg1_type arg1_type;
            typedef typename sig_traits<funct>::arg2_type arg2_type;
            typedef typename sig_traits<funct>::arg3_type arg3_type;
            typedef typename sig_traits<funct>::arg4_type arg4_type;
            typedef typename sig_traits<funct>::arg5_type arg5_type;
            typedef typename sig_traits<funct>::arg6_type arg6_type;
            typedef typename sig_traits<funct>::arg7_type arg7_type;
            typedef typename sig_traits<funct>::arg8_type arg8_type;
            typedef typename sig_traits<funct>::arg9_type arg9_type;
            typedef typename sig_traits<funct>::arg10_type arg10_type;
            typedef typename sig_traits<funct>::arg11_type arg11_type;
            typedef typename sig_traits<funct>::arg12_type arg12_type;
            typedef typename sig_traits<funct>::arg13_type arg13_type;
            typedef typename sig_traits<funct>::arg14_type arg14_type;
            typedef typename sig_traits<funct>::arg15_type arg15_type;
            typedef typename sig_traits<funct>::arg16_type arg16_type;
            typedef typename sig_traits<funct>::arg17_type arg17_type;
            typedef typename sig_traits<funct>::arg18_type arg18_type;
            typedef typename sig_traits<funct>::arg19_type arg19_type;
            typedef typename sig_traits<funct>::arg20_type arg20_type;

            typename basic_type<arg1_type>::type A1;
            typename basic_type<arg2_type>::type A2;
            typename basic_type<arg3_type>::type A3;
            typename basic_type<arg4_type>::type A4;
            typename basic_type<arg5_type>::type A5;
            typename basic_type<arg6_type>::type A6;
            typename basic_type<arg7_type>::type A7;
            typename basic_type<arg8_type>::type A8;
            typename basic_type<arg9_type>::type A9;
            typename basic_type<arg10_type>::type A10;
            typename basic_type<arg11_type>::type A11;
            typename basic_type<arg12_type>::type A12;
            typename basic_type<arg13_type>::type A13;
            typename basic_type<arg14_type>::type A14;
            typename basic_type<arg15_type>::type A15;
            typename basic_type<arg16_type>::type A16;
            typename basic_type<arg17_type>::type A17;
            typename basic_type<arg18_type>::type A18;
            typename basic_type<arg19_type>::type A19;
            typename basic_type<arg20_type>::type A20;

            mark_owned_by_matlab(A1);
            mark_owned_by_matlab(A2);
            mark_owned_by_matlab(A3);
            mark_owned_by_matlab(A4);
            mark_owned_by_matlab(A5);
            mark_owned_by_matlab(A6);
            mark_owned_by_matlab(A7);
            mark_owned_by_matlab(A8);
            mark_owned_by_matlab(A9);
            mark_owned_by_matlab(A10);
            mark_owned_by_matlab(A11);
            mark_owned_by_matlab(A12);
            mark_owned_by_matlab(A13);
            mark_owned_by_matlab(A14);
            mark_owned_by_matlab(A15);
            mark_owned_by_matlab(A16);
            mark_owned_by_matlab(A17);
            mark_owned_by_matlab(A18);
            mark_owned_by_matlab(A19);
            mark_owned_by_matlab(A20);

            int i = 0;
            if (i < nrhs && is_input_type<arg1_type>::value) {validate_and_populate_arg(i,prhs[i],A1); ++i;} ELSE_ASSIGN_ARG_1;
            if (i < nrhs && is_input_type<arg2_type>::value) {validate_and_populate_arg(i,prhs[i],A2); ++i;} ELSE_ASSIGN_ARG_2;
            if (i < nrhs && is_input_type<arg3_type>::value) {validate_and_populate_arg(i,prhs[i],A3); ++i;} ELSE_ASSIGN_ARG_3;
            if (i < nrhs && is_input_type<arg4_type>::value) {validate_and_populate_arg(i,prhs[i],A4); ++i;} ELSE_ASSIGN_ARG_4;
            if (i < nrhs && is_input_type<arg5_type>::value) {validate_and_populate_arg(i,prhs[i],A5); ++i;} ELSE_ASSIGN_ARG_5;
            if (i < nrhs && is_input_type<arg6_type>::value) {validate_and_populate_arg(i,prhs[i],A6); ++i;} ELSE_ASSIGN_ARG_6;
            if (i < nrhs && is_input_type<arg7_type>::value) {validate_and_populate_arg(i,prhs[i],A7); ++i;} ELSE_ASSIGN_ARG_7;
            if (i < nrhs && is_input_type<arg8_type>::value) {validate_and_populate_arg(i,prhs[i],A8); ++i;} ELSE_ASSIGN_ARG_8;
            if (i < nrhs && is_input_type<arg9_type>::value) {validate_and_populate_arg(i,prhs[i],A9); ++i;} ELSE_ASSIGN_ARG_9;
            if (i < nrhs && is_input_type<arg10_type>::value) {validate_and_populate_arg(i,prhs[i],A10); ++i;} ELSE_ASSIGN_ARG_10;
            if (i < nrhs && is_input_type<arg11_type>::value) {validate_and_populate_arg(i,prhs[i],A11); ++i;} ELSE_ASSIGN_ARG_11;
            if (i < nrhs && is_input_type<arg12_type>::value) {validate_and_populate_arg(i,prhs[i],A12); ++i;} ELSE_ASSIGN_ARG_12;
            if (i < nrhs && is_input_type<arg13_type>::value) {validate_and_populate_arg(i,prhs[i],A13); ++i;} ELSE_ASSIGN_ARG_13;
            if (i < nrhs && is_input_type<arg14_type>::value) {validate_and_populate_arg(i,prhs[i],A14); ++i;} ELSE_ASSIGN_ARG_14;
            if (i < nrhs && is_input_type<arg15_type>::value) {validate_and_populate_arg(i,prhs[i],A15); ++i;} ELSE_ASSIGN_ARG_15;
            if (i < nrhs && is_input_type<arg16_type>::value) {validate_and_populate_arg(i,prhs[i],A16); ++i;} ELSE_ASSIGN_ARG_16;
            if (i < nrhs && is_input_type<arg17_type>::value) {validate_and_populate_arg(i,prhs[i],A17); ++i;} ELSE_ASSIGN_ARG_17;
            if (i < nrhs && is_input_type<arg18_type>::value) {validate_and_populate_arg(i,prhs[i],A18); ++i;} ELSE_ASSIGN_ARG_18;
            if (i < nrhs && is_input_type<arg19_type>::value) {validate_and_populate_arg(i,prhs[i],A19); ++i;} ELSE_ASSIGN_ARG_19;
            if (i < nrhs && is_input_type<arg20_type>::value) {validate_and_populate_arg(i,prhs[i],A20); ++i;} ELSE_ASSIGN_ARG_20;

            f(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20);


            i = 0;
            if (is_output_type<arg1_type>::value) {assign_to_matlab(plhs[i],A1); ++i;}
            if (is_output_type<arg2_type>::value) {assign_to_matlab(plhs[i],A2); ++i;}
            if (is_output_type<arg3_type>::value) {assign_to_matlab(plhs[i],A3); ++i;}
            if (is_output_type<arg4_type>::value) {assign_to_matlab(plhs[i],A4); ++i;}
            if (is_output_type<arg5_type>::value) {assign_to_matlab(plhs[i],A5); ++i;}
            if (is_output_type<arg6_type>::value) {assign_to_matlab(plhs[i],A6); ++i;}
            if (is_output_type<arg7_type>::value) {assign_to_matlab(plhs[i],A7); ++i;}
            if (is_output_type<arg8_type>::value) {assign_to_matlab(plhs[i],A8); ++i;}
            if (is_output_type<arg9_type>::value) {assign_to_matlab(plhs[i],A9); ++i;}
            if (is_output_type<arg10_type>::value) {assign_to_matlab(plhs[i],A10); ++i;}
            if (is_output_type<arg11_type>::value) {assign_to_matlab(plhs[i],A11); ++i;}
            if (is_output_type<arg12_type>::value) {assign_to_matlab(plhs[i],A12); ++i;}
            if (is_output_type<arg13_type>::value) {assign_to_matlab(plhs[i],A13); ++i;}
            if (is_output_type<arg14_type>::value) {assign_to_matlab(plhs[i],A14); ++i;}
            if (is_output_type<arg15_type>::value) {assign_to_matlab(plhs[i],A15); ++i;}
            if (is_output_type<arg16_type>::value) {assign_to_matlab(plhs[i],A16); ++i;}
            if (is_output_type<arg17_type>::value) {assign_to_matlab(plhs[i],A17); ++i;}
            if (is_output_type<arg18_type>::value) {assign_to_matlab(plhs[i],A18); ++i;}
            if (is_output_type<arg19_type>::value) {assign_to_matlab(plhs[i],A19); ++i;}
            if (is_output_type<arg20_type>::value) {assign_to_matlab(plhs[i],A20); ++i;}
        }
    };

// ----------------------------------------------------------------------------------------

    template <typename T> struct is_matlab_object                         { const static bool value = false; };
    template <>           struct is_matlab_object <matlab_object>         { const static bool value = true; };
    template <>           struct is_matlab_object <const matlab_object>   { const static bool value = true; };
    template <>           struct is_matlab_object <matlab_object&>        { const static bool value = true; };
    template <>           struct is_matlab_object <const matlab_object&>  { const static bool value = true; };

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    void call_mex_function (
        const funct& f,
        int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]
    )
    {
        const long expected_nrhs = funct_traits<funct>::num_inputs;
        const long expected_nlhs = funct_traits<funct>::num_outputs;
        const long expected_args = expected_nrhs + expected_nlhs;

        long defaulted_args = 0;

        #ifdef ARG_1_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg1_type>::value);
            #ifndef ARG_2_DEFAULT
                // You can't define a default for argument 1 if you don't define one for argument 2 also.
                COMPILE_TIME_ASSERT(expected_args < 2);
            #endif
            COMPILE_TIME_ASSERT(1 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_2_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg2_type>::value);
            #ifndef ARG_3_DEFAULT
                // You can't define a default for argument 2 if you don't define one for argument 3 also.
                COMPILE_TIME_ASSERT(expected_args < 3);
            #endif
            COMPILE_TIME_ASSERT(2 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_3_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg3_type>::value);
            #ifndef ARG_4_DEFAULT
                // You can't define a default for argument 3 if you don't define one for argument 4 also.
                COMPILE_TIME_ASSERT(expected_args < 4);
            #endif
            COMPILE_TIME_ASSERT(3 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_4_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg4_type>::value);
            #ifndef ARG_5_DEFAULT
                // You can't define a default for argument 4 if you don't define one for argument 5 also.
                COMPILE_TIME_ASSERT(expected_args < 5);
            #endif
            COMPILE_TIME_ASSERT(4 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_5_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg5_type>::value);
            #ifndef ARG_6_DEFAULT
                // You can't define a default for argument 5 if you don't define one for argument 6 also.
                COMPILE_TIME_ASSERT(expected_args < 6);
            #endif
            COMPILE_TIME_ASSERT(5 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_6_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg6_type>::value);
            #ifndef ARG_7_DEFAULT
                // You can't define a default for argument 6 if you don't define one for argument 7 also.
                COMPILE_TIME_ASSERT(expected_args < 7);
            #endif
            COMPILE_TIME_ASSERT(6 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_7_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg7_type>::value);
            #ifndef ARG_8_DEFAULT
                // You can't define a default for argument 7 if you don't define one for argument 8 also.
                COMPILE_TIME_ASSERT(expected_args < 8);
            #endif
            COMPILE_TIME_ASSERT(7 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_8_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg8_type>::value);
            #ifndef ARG_9_DEFAULT
                // You can't define a default for argument 8 if you don't define one for argument 9 also.
                COMPILE_TIME_ASSERT(expected_args < 9);
            #endif
            COMPILE_TIME_ASSERT(8 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_9_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg9_type>::value);
            #ifndef ARG_10_DEFAULT
                // You can't define a default for argument 9 if you don't define one for argument 10 also.
                COMPILE_TIME_ASSERT(expected_args < 10);
            #endif
            COMPILE_TIME_ASSERT(9 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_10_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg10_type>::value);
            #ifndef ARG_11_DEFAULT
                // You can't define a default for argument 10 if you don't define one for argument 11 also.
                COMPILE_TIME_ASSERT(expected_args < 11);
            #endif
            COMPILE_TIME_ASSERT(10 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_11_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg11_type>::value);
            #ifndef ARG_12_DEFAULT
                // You can't define a default for argument 11 if you don't define one for argument 12 also.
                COMPILE_TIME_ASSERT(expected_args < 12);
            #endif
            COMPILE_TIME_ASSERT(11 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_12_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg12_type>::value);
            #ifndef ARG_13_DEFAULT
                // You can't define a default for argument 12 if you don't define one for argument 13 also.
                COMPILE_TIME_ASSERT(expected_args < 13);
            #endif
            COMPILE_TIME_ASSERT(12 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_13_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg13_type>::value);
            #ifndef ARG_14_DEFAULT
                // You can't define a default for argument 13 if you don't define one for argument 14 also.
                COMPILE_TIME_ASSERT(expected_args < 14);
            #endif
            COMPILE_TIME_ASSERT(13 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_14_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg14_type>::value);
            #ifndef ARG_15_DEFAULT
                // You can't define a default for argument 14 if you don't define one for argument 15 also.
                COMPILE_TIME_ASSERT(expected_args < 15);
            #endif
            COMPILE_TIME_ASSERT(14 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_15_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg15_type>::value);
            #ifndef ARG_16_DEFAULT
                // You can't define a default for argument 15 if you don't define one for argument 16 also.
                COMPILE_TIME_ASSERT(expected_args < 16);
            #endif
            COMPILE_TIME_ASSERT(15 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_16_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg16_type>::value);
            #ifndef ARG_17_DEFAULT
                // You can't define a default for argument 16 if you don't define one for argument 17 also.
                COMPILE_TIME_ASSERT(expected_args < 17);
            #endif
            COMPILE_TIME_ASSERT(16 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_17_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg17_type>::value);
            #ifndef ARG_18_DEFAULT
                // You can't define a default for argument 17 if you don't define one for argument 18 also.
                COMPILE_TIME_ASSERT(expected_args < 18);
            #endif
            COMPILE_TIME_ASSERT(17 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_18_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg18_type>::value);
            #ifndef ARG_19_DEFAULT
                // You can't define a default for argument 18 if you don't define one for argument 19 also.
                COMPILE_TIME_ASSERT(expected_args < 19);
            #endif
            COMPILE_TIME_ASSERT(18 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_19_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg19_type>::value);
            #ifndef ARG_20_DEFAULT
                // You can't define a default for argument 19 if you don't define one for argument 20 also.
                COMPILE_TIME_ASSERT(expected_args < 20);
            #endif
            COMPILE_TIME_ASSERT(19 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif
        #ifdef ARG_20_DEFAULT
            ++defaulted_args;
            // You can only set an argument's default value if it is an input argument.
            COMPILE_TIME_ASSERT(is_input_type<typename sig_traits<funct>::arg20_type>::value);
            COMPILE_TIME_ASSERT(20 <= expected_args); // You can't define a default for an argument that doesn't exist.
        #endif


        //  Arguments with type matlab_object are optional in both input and output.
        int num_optional_inputs = 0;
        int num_optional_outputs = 0;
        if (is_matlab_object<typename sig_traits<funct>::arg20_type>::value) if (is_input_type<typename sig_traits<funct>::arg20_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg19_type>::value) if (is_input_type<typename sig_traits<funct>::arg19_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg18_type>::value) if (is_input_type<typename sig_traits<funct>::arg18_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg17_type>::value) if (is_input_type<typename sig_traits<funct>::arg17_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg16_type>::value) if (is_input_type<typename sig_traits<funct>::arg16_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg15_type>::value) if (is_input_type<typename sig_traits<funct>::arg15_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg14_type>::value) if (is_input_type<typename sig_traits<funct>::arg14_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg13_type>::value) if (is_input_type<typename sig_traits<funct>::arg13_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg12_type>::value) if (is_input_type<typename sig_traits<funct>::arg12_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg11_type>::value) if (is_input_type<typename sig_traits<funct>::arg11_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg10_type>::value) if (is_input_type<typename sig_traits<funct>::arg10_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg9_type>::value) if (is_input_type<typename sig_traits<funct>::arg9_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg8_type>::value) if (is_input_type<typename sig_traits<funct>::arg8_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg7_type>::value) if (is_input_type<typename sig_traits<funct>::arg7_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg6_type>::value) if (is_input_type<typename sig_traits<funct>::arg6_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg5_type>::value) if (is_input_type<typename sig_traits<funct>::arg5_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg4_type>::value) if (is_input_type<typename sig_traits<funct>::arg4_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg3_type>::value) if (is_input_type<typename sig_traits<funct>::arg3_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg2_type>::value) if (is_input_type<typename sig_traits<funct>::arg2_type>::value) ++num_optional_inputs; else ++num_optional_outputs;
        if (is_matlab_object<typename sig_traits<funct>::arg1_type>::value) if (is_input_type<typename sig_traits<funct>::arg1_type>::value) ++num_optional_inputs; else ++num_optional_outputs;


        /* check for proper number of arguments */
        if(nrhs > expected_nrhs || nrhs < expected_nrhs - defaulted_args - num_optional_inputs) 
        {
            std::ostringstream sout;
            sout << "Expected between " << expected_nrhs-defaulted_args - num_optional_inputs 
                << " and " << expected_nrhs << " input arguments, got " << nrhs << ".";

            mexErrMsgIdAndTxt("mex_function:nrhs",
                              escape_percent(sout).c_str());
        }

        if (nlhs > expected_nlhs)
        {
            std::ostringstream sout;
            sout << "Expected at most " << expected_nlhs << " output arguments, got " << nlhs << ".";

            mexErrMsgIdAndTxt("mex_function:nlhs",
                              escape_percent(sout).c_str());
        }

        call_mex_function_helper<sig_traits<funct>::num_args> helper;
        helper.callit(f, nlhs, plhs, nrhs, prhs);

    }

// ----------------------------------------------------------------------------------------

    class mex_streambuf : public std::streambuf
    {

    public:
        mex_streambuf (
        ) 
        {
            buf.resize(1000);
            setp(&buf[0], &buf[0] + buf.size()-2);

            // make cout send data to mex_streambuf
            oldbuf = std::cout.rdbuf(this);
        }

        ~mex_streambuf()
        {
            // put cout back to the way we found it before running our mex function.
            std::cout.rdbuf(oldbuf);
        }


    protected:


        int sync (
        )
        {
            int num = static_cast<int>(pptr()-pbase());
            if (num != 0)
            {
                check_for_matlab_ctrl_c();

                buf[num] = 0; // null terminate the string
                mexPrintf("%s",&buf[0]);
                mexEvalString("drawnow"); // flush print to screen
                pbump(-num);
            }
            return 0;
        }

        int_type overflow (
            int_type c
        )
        {
            if (c != EOF)
            {
                *pptr() = c;
                pbump(1);
            }
            sync();
            return c;
        }

    private:
        std::vector<char> buf;
        std::streambuf* oldbuf;

    };

    class mex_warn_streambuf : public std::streambuf
    {

    public:
        mex_warn_streambuf (
        ) 
        {
            buf.resize(1000);
            setp(&buf[0], &buf[0] + buf.size()-2);

            // make cout send data to mex_warn_streambuf
            oldbuf = std::cerr.rdbuf(this);
        }

        ~mex_warn_streambuf()
        {
            // put cerr back to the way we found it before running our mex function.
            std::cerr.rdbuf(oldbuf);
        }

    protected:


        int sync (
        )
        {
            int num = static_cast<int>(pptr()-pbase());
            if (num != 0)
            {
                check_for_matlab_ctrl_c();

                buf[num] = 0; // null terminate the string
                mexWarnMsgTxt(&buf[0]);
                mexEvalString("drawnow"); // flush print to screen
                pbump(-num);
            }
            return 0;
        }

        int_type overflow (
            int_type c
        )
        {
            if (c != EOF)
            {
                *pptr() = c;
                pbump(1);
            }
            sync();
            return c;
        }

    private:
        std::vector<char> buf;
        std::streambuf* oldbuf;

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T>
    void setup_input_args (
        mxArray*& array,
        const T& item,
        int& nrhs
    )
    {
        assign_to_matlab(array, item);
        ++nrhs;
    }

    void setup_input_args (
        mxArray*& array,
        const function_handle& item,
        int& nrhs
    )
    {
        array = static_cast<mxArray*>(item.h);
        ++nrhs;
    }

    template <typename T>
    void setup_input_args (
        mxArray*& array,
        const output_decorator<T>& item,
        int& nrhs
    )
    {
    }

    template <typename T>
    void setup_output_args (
        const std::string& function_name,
        mxArray* array,
        const T& item,
        int& nrhs
    )
    {
    }

    template <typename T>
    void setup_output_args (
        const std::string& function_name,
        mxArray* array,
        const output_decorator<T>& item,
        int& i
    )
    {
        try
        {
            validate_and_populate_arg(i,array,const_cast<T&>(item.item));
            ++i;
        }
        catch (invalid_args_exception& e)
        {
            throw dlib::error("Error occurred calling MATLAB function '" + function_name + "' from mex file. \n"
                              "The MATLAB function didn't return what we expected it to.  \nIn particular, return" + string(e.what()));
        }
    }

    void call_matlab_for_real (
        int nlhs,
        mxArray* plhs[],
        int nrhs,
        mxArray* prhs[],
        const std::string& function_name
    )
    {
        int status = mexCallMATLAB(nlhs, plhs, nrhs, prhs, function_name.c_str());
        if (status)
        {
            throw dlib::error("Error, an exception was thrown when we tried to call the MATLAB function '" + function_name + "'.");
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

namespace dlib
{
    void call_matlab (
        const std::string& function_name
    ) 
    {
        using namespace mex_binding;

        call_matlab_for_real(0,NULL,0,NULL, function_name);
    }

    template <typename T1>
    void free_callback_resources (
        int nlhs,
        mxArray* plhs[],
        int nrhs,
        mxArray* prhs[]
    )
    {
        // free resources
        for (int i = 0; i < nlhs; ++i)
            mxDestroyArray(plhs[i]);

        for (int i = 0; i < nrhs; ++i)
        {
            // don't call mxDestroyArray() on function handles (which should only ever be in prhs[0])
            if (i == 0 && dlib::is_same_type<T1,function_handle>::value)
                continue;
            mxDestroyArray(prhs[i]);
        }
    }

    template <
        typename T1
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1
    ) 
    {
        using namespace mex_binding;
        const int num_args = 1;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2
    ) 
    {
        using namespace mex_binding;
        const int num_args = 2;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3
    ) 
    {
        using namespace mex_binding;
        const int num_args = 3;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4
    ) 
    {
        using namespace mex_binding;
        const int num_args = 4;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5
    ) 
    {
        using namespace mex_binding;
        const int num_args = 5;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6
    ) 
    {
        using namespace mex_binding;
        const int num_args = 6;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7
    ) 
    {
        using namespace mex_binding;
        const int num_args = 7;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8
    ) 
    {
        using namespace mex_binding;
        const int num_args = 8;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }



    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9
    ) 
    {
        using namespace mex_binding;
        const int num_args = 9;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }


    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10
    ) 
    {
        using namespace mex_binding;
        const int num_args = 10;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11
    ) 
    {
        using namespace mex_binding;
        const int num_args = 11;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12
    ) 
    {
        using namespace mex_binding;
        const int num_args = 12;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12,
        typename T13
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12,
        const T13& A13
    ) 
    {
        using namespace mex_binding;
        const int num_args = 13;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12,
        typename T13,
        typename T14
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12,
        const T13& A13,
        const T14& A14
    ) 
    {
        using namespace mex_binding;
        const int num_args = 14;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12,
        typename T13,
        typename T14,
        typename T15
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12,
        const T13& A13,
        const T14& A14,
        const T15& A15
    ) 
    {
        using namespace mex_binding;
        const int num_args = 15;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12,
        typename T13,
        typename T14,
        typename T15,
        typename T16
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12,
        const T13& A13,
        const T14& A14,
        const T15& A15,
        const T16& A16
    ) 
    {
        using namespace mex_binding;
        const int num_args = 16;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);
        setup_input_args(prhs[nrhs], A16, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);
        setup_output_args(function_name, plhs[i], A16, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, 
        typename T2,
        typename T3,
        typename T4,
        typename T5,
        typename T6,
        typename T7,
        typename T8,
        typename T9,
        typename T10,
        typename T11,
        typename T12,
        typename T13,
        typename T14,
        typename T15,
        typename T16,
        typename T17
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1,
        const T2& A2,
        const T3& A3,
        const T4& A4,
        const T5& A5,
        const T6& A6,
        const T7& A7,
        const T8& A8,
        const T9& A9,
        const T10& A10,
        const T11& A11,
        const T12& A12,
        const T13& A13,
        const T14& A14,
        const T15& A15,
        const T16& A16,
        const T17& A17
    ) 
    {
        using namespace mex_binding;
        const int num_args = 17;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);
        setup_input_args(prhs[nrhs], A16, nrhs);
        setup_input_args(prhs[nrhs], A17, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);
        setup_output_args(function_name, plhs[i], A16, i);
        setup_output_args(function_name, plhs[i], A17, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9, typename T10, typename T11, typename T12,
        typename T13, typename T14, typename T15, typename T16, typename T17, typename T18
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
        const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const
        T12& A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const
        T17& A17, const T18& A18
    ) 
    {
        using namespace mex_binding;
        const int num_args = 18;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);
        setup_input_args(prhs[nrhs], A16, nrhs);
        setup_input_args(prhs[nrhs], A17, nrhs);
        setup_input_args(prhs[nrhs], A18, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);
        setup_output_args(function_name, plhs[i], A16, i);
        setup_output_args(function_name, plhs[i], A17, i);
        setup_output_args(function_name, plhs[i], A18, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9, typename T10, typename T11, typename T12,
        typename T13, typename T14, typename T15, typename T16, typename T17, typename T18,
        typename T19
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
        const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const
        T12& A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const
        T17& A17, const T18& A18, const T19& A19
    ) 
    {
        using namespace mex_binding;
        const int num_args = 19;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);
        setup_input_args(prhs[nrhs], A16, nrhs);
        setup_input_args(prhs[nrhs], A17, nrhs);
        setup_input_args(prhs[nrhs], A18, nrhs);
        setup_input_args(prhs[nrhs], A19, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);
        setup_output_args(function_name, plhs[i], A16, i);
        setup_output_args(function_name, plhs[i], A17, i);
        setup_output_args(function_name, plhs[i], A18, i);
        setup_output_args(function_name, plhs[i], A19, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

    template <
        typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9, typename T10, typename T11, typename T12,
        typename T13, typename T14, typename T15, typename T16, typename T17, typename T18,
        typename T19, typename T20
        >
    void call_matlab (
        const std::string& function_name,
        const T1& A1, const T2& A2, const T3& A3, const T4& A4, const T5& A5, const T6& A6,
        const T7& A7, const T8& A8, const T9& A9, const T10& A10, const T11& A11, const
        T12& A12, const T13& A13, const T14& A14, const T15& A15, const T16& A16, const
        T17& A17, const T18& A18, const T19& A19, const T20& A20
    ) 
    {
        using namespace mex_binding;
        const int num_args = 20;
        mxArray* plhs[num_args] = {0};
        mxArray* prhs[num_args] = {0};

        int nrhs = 0;
        setup_input_args(prhs[nrhs], A1, nrhs);
        setup_input_args(prhs[nrhs], A2, nrhs);
        setup_input_args(prhs[nrhs], A3, nrhs);
        setup_input_args(prhs[nrhs], A4, nrhs);
        setup_input_args(prhs[nrhs], A5, nrhs);
        setup_input_args(prhs[nrhs], A6, nrhs);
        setup_input_args(prhs[nrhs], A7, nrhs);
        setup_input_args(prhs[nrhs], A8, nrhs);
        setup_input_args(prhs[nrhs], A9, nrhs);
        setup_input_args(prhs[nrhs], A10, nrhs);
        setup_input_args(prhs[nrhs], A11, nrhs);
        setup_input_args(prhs[nrhs], A12, nrhs);
        setup_input_args(prhs[nrhs], A13, nrhs);
        setup_input_args(prhs[nrhs], A14, nrhs);
        setup_input_args(prhs[nrhs], A15, nrhs);
        setup_input_args(prhs[nrhs], A16, nrhs);
        setup_input_args(prhs[nrhs], A17, nrhs);
        setup_input_args(prhs[nrhs], A18, nrhs);
        setup_input_args(prhs[nrhs], A19, nrhs);
        setup_input_args(prhs[nrhs], A20, nrhs);

        const int nlhs = num_args - nrhs;
        call_matlab_for_real(nlhs,plhs,nrhs,prhs, function_name);

        int i = 0;
        setup_output_args(function_name, plhs[i], A1, i);
        setup_output_args(function_name, plhs[i], A2, i);
        setup_output_args(function_name, plhs[i], A3, i);
        setup_output_args(function_name, plhs[i], A4, i);
        setup_output_args(function_name, plhs[i], A5, i);
        setup_output_args(function_name, plhs[i], A6, i);
        setup_output_args(function_name, plhs[i], A7, i);
        setup_output_args(function_name, plhs[i], A8, i);
        setup_output_args(function_name, plhs[i], A9, i);
        setup_output_args(function_name, plhs[i], A10, i);
        setup_output_args(function_name, plhs[i], A11, i);
        setup_output_args(function_name, plhs[i], A12, i);
        setup_output_args(function_name, plhs[i], A13, i);
        setup_output_args(function_name, plhs[i], A14, i);
        setup_output_args(function_name, plhs[i], A15, i);
        setup_output_args(function_name, plhs[i], A16, i);
        setup_output_args(function_name, plhs[i], A17, i);
        setup_output_args(function_name, plhs[i], A18, i);
        setup_output_args(function_name, plhs[i], A19, i);
        setup_output_args(function_name, plhs[i], A20, i);

        free_callback_resources<T1>(nlhs,plhs,nrhs,prhs);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    matlab_object::~matlab_object(
    )
    {
        if (handle && should_free)
        {
            mxDestroyArray((mxArray*)handle);
            handle = 0;
        }
    }

    template <typename T> 
    matlab_object::
    operator T(
    ) const
    {
        T item;
        get(item);
        return item;
    }

    template <typename T> 
    void matlab_object::
    get(
        T& item
    ) const 
    {
        if (handle == 0)
            throw dlib::invalid_args_exception("An attempt was made to access an empty matlab_object.");

        mex_binding::validate_and_populate_arg(arg_idx,(mxArray*)handle,item);
    }

    template <typename T> 
    matlab_object& matlab_object::
    operator= (
        const T& new_val
    )
    {
        mxArray* item;
        mex_binding::assign_to_matlab(item, new_val);
        handle = item;
        should_free = true;
        return *this;
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T>
    matlab_struct::sub::operator T() const
    {
        T item;
        get(item);
        return item;
    }

    template <typename T>
    void matlab_struct::sub::get(T& item) const
    {
        if (struct_handle == 0)
            throw dlib::error("Attempt to access data in an empty struct.");

        mxArray* temp = mxGetFieldByNumber((const mxArray*)struct_handle, 0, field_idx);
        if (temp == 0)
            throw dlib::error("Attempt to access data in an empty struct.");

        try
        {
            mex_binding::validate_and_populate_arg(0,temp,item);
        }
        catch(mex_binding::invalid_args_exception& e)
        {
            std::ostringstream sout;
            sout << "Struct field '" << mxGetFieldNameByNumber((const mxArray*)struct_handle, field_idx) << "' can't be interpreted as the requested type."
                << endl << e.what();
            throw dlib::error(sout.str());
        }
    }

    const matlab_struct::sub matlab_struct::
    operator[] (const std::string& name) const
    {
        if (struct_handle == 0)
            throw dlib::error("Struct does not have a field named '" + name + "'.");

        matlab_struct::sub temp;
        temp.struct_handle = struct_handle;
        temp.field_idx = mxGetFieldNumber((const mxArray*)struct_handle, name.c_str());
        if (temp.field_idx == -1 )
            throw dlib::error("Struct does not have a field named '" + name + "'.");
        return temp;
    }

    matlab_struct::sub matlab_struct::
    operator[] (const std::string& name) 
    {
        if (struct_handle == 0)
        {
            // We make a struct from scratch and mark that we will free it unless it gets
            // written back to matlab by assign_to_matlab().
            mwSize dims[1] = {1};
            const char* name_str = name.c_str();
            struct_handle = mxCreateStructArray(1, dims, 1, &name_str);
            should_free = true;
            if (struct_handle == 0)
                throw dlib::error("Error creating struct from within mex function.");
        }


        matlab_struct::sub temp;
        temp.struct_handle = struct_handle;
        if ((temp.field_idx=mxGetFieldNumber((mxArray*)struct_handle, name.c_str())) == -1)
        {
            if ((temp.field_idx=mxAddField((mxArray*)struct_handle, name.c_str())) == -1)
            {
                throw dlib::error("Unable to add field '"+name + "' to struct.");
            }
        }
        return temp;
    }

    const matlab_struct::sub matlab_struct::sub::
    operator[] (const std::string& name) const
    {
        if (struct_handle == 0)
            throw dlib::error("Struct does not have a field named '" + name + "'.");

        matlab_struct::sub temp;
        temp.struct_handle = mxGetFieldByNumber((const mxArray*)struct_handle, 0, field_idx);
        if (temp.struct_handle == 0)
            throw dlib::error("Failure to get struct field while calling mxGetFieldByNumber()");

        if (!mxIsStruct((const mxArray*)temp.struct_handle))
            throw dlib::error("Struct sub-field element '"+name+"' is not another struct.");

        temp.field_idx = mxGetFieldNumber((const mxArray*)temp.struct_handle, name.c_str());
        if (temp.field_idx == -1 )
            throw dlib::error("Struct does not have a field named '" + name + "'.");
        return temp;
    }

    matlab_struct::sub matlab_struct::sub::
    operator[] (const std::string& name) 
    {
        if (struct_handle == 0)
            throw dlib::error("Struct does not have a field named '" + name + "'.");

        matlab_struct::sub temp;
        temp.struct_handle = mxGetFieldByNumber((const mxArray*)struct_handle, 0, field_idx);
        // We are replacing this field with a struct if it exists and isn't already a struct
        if (temp.struct_handle != 0 && !mxIsStruct((const mxArray*)temp.struct_handle))
        {
            mxDestroyArray((mxArray*)temp.struct_handle);
            temp.struct_handle = 0;
        }
        if (temp.struct_handle == 0)
        {
            mwSize dims[1] = {1};
            temp.struct_handle = mxCreateStructArray(1, dims, 0, 0);
            if (temp.struct_handle == 0)
                throw dlib::error("Failure to create new sub-struct field");
            mxSetFieldByNumber((mxArray*)struct_handle, 0, field_idx, (mxArray*)temp.struct_handle);
        }


        if ((temp.field_idx=mxGetFieldNumber((mxArray*)temp.struct_handle, name.c_str())) == -1)
        {
            if ((temp.field_idx=mxAddField((mxArray*)temp.struct_handle, name.c_str())) == -1)
            {
                throw dlib::error("Unable to add field '"+name + "' to struct.");
            }
        }
        return temp;
    }

    bool matlab_struct::has_field (
        const std::string& name
    ) const
    {
        if (struct_handle == 0)
            return false;
        return mxGetFieldNumber((const mxArray*)struct_handle, name.c_str()) != -1;
    }

    bool matlab_struct::sub::has_field (
        const std::string& name
    ) const
    {
        if (struct_handle == 0)
            return false;
        mxArray* temp = mxGetFieldByNumber((const mxArray*)struct_handle, 0, field_idx);
        if (temp == 0 || !mxIsStruct(temp))
            return false;
        return mxGetFieldNumber(temp, name.c_str()) != -1;
    }

    template <typename T>
    matlab_struct::sub& matlab_struct::sub::operator= (
        const T& new_val 
    )
    {
        // Delete anything in the field before we overwrite it
        mxArray* item = mxGetFieldByNumber((mxArray*)struct_handle, 0, field_idx);
        if (item != 0)
        {
            mxDestroyArray((mxArray*)item);
            item = 0;
        }

        // Now set the field
        mex_binding::assign_to_matlab(item, new_val);
        mxSetFieldByNumber((mxArray*)struct_handle, 0, field_idx, item);

        return *this;
    }

    matlab_struct::
    ~matlab_struct (
    )
    {
        if (struct_handle && should_free)
        {
            mxDestroyArray((mxArray*)struct_handle);
            struct_handle = 0;
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void call_matlab (
        const function_handle& funct 
    )
    {
        call_matlab("feval", funct);
    }

    extern "C" bool utIsInterruptPending();
    void check_for_matlab_ctrl_c(
    )
    {
        if (utIsInterruptPending())
            throw mex_binding::user_hit_ctrl_c();
    }
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

#ifdef MEX_CLASS_NAME
template <typename T, typename mfp_type>
class mex_class_wrapper
{
public:
    mex_class_wrapper(T& obj_, mfp_type mfp_) : obj(obj_), mfp(mfp_) {}

    template <typename... Args>
    void operator()(Args&&... args) const
    {
        (obj.*mfp)(std::forward<Args>(args)...);
    }

    mfp_type mfp;
    T& obj;
};

template <typename T, typename mfp_type>
mex_class_wrapper<T,mfp_type> wrap_mex_class(T& obj, mfp_type mfp) { return mex_class_wrapper<T,mfp_type>(obj, mfp); }

namespace dlib
{
    template <typename T, typename mfp_type>
    struct sig_traits<mex_class_wrapper<T,mfp_type>>
    : public sig_traits<mfp_type>
    {};

    template <size_t i, typename T, bool is_good = i < std::tuple_size<T>::value>
    struct tuple_element_default_void
    {
        typedef void type;
    };

    template <size_t i, typename T>
    struct tuple_element_default_void<i,T,true>
    {
        typedef typename std::tuple_element<i,T>::type type;
    };

    template <typename class_type, typename return_type, typename... Args>
    struct sig_traits<return_type(class_type::*)(Args...) >
    {
        enum { num_args = sizeof...(Args) };

        typedef return_type result_type;

        template <size_t i>
        struct arg
        {
            typedef typename tuple_element_default_void<i-1, std::tuple<Args...>>::type type;
        };

        // These are here because that's how things are defined in sig_traits (since it is
        // older than C++11, along with most of the other code in this file)
        typedef typename arg<1>::type arg1_type;
        typedef typename arg<2>::type arg2_type;
        typedef typename arg<3>::type arg3_type;
        typedef typename arg<4>::type arg4_type;
        typedef typename arg<5>::type arg5_type;
        typedef typename arg<6>::type arg6_type;
        typedef typename arg<7>::type arg7_type;
        typedef typename arg<8>::type arg8_type;
        typedef typename arg<9>::type arg9_type;
        typedef typename arg<10>::type arg10_type;
        typedef typename arg<11>::type arg11_type;
        typedef typename arg<12>::type arg12_type;
        typedef typename arg<13>::type arg13_type;
        typedef typename arg<14>::type arg14_type;
        typedef typename arg<15>::type arg15_type;
        typedef typename arg<16>::type arg16_type;
        typedef typename arg<17>::type arg17_type;
        typedef typename arg<18>::type arg18_type;
        typedef typename arg<19>::type arg19_type;
        typedef typename arg<20>::type arg20_type;
    };

    template <typename class_type, typename return_type, typename... Args>
    struct sig_traits<return_type(class_type::*)(Args...) const>
    {
        enum { num_args = sizeof...(Args) };

        typedef return_type result_type;

        template <size_t i>
        struct arg
        {
            typedef typename tuple_element_default_void<i-1, std::tuple<Args...>>::type type;
        };

        // These are here because that's how things are defined in sig_traits (since it is
        // older than C++11, along with most of the other code in this file)
        typedef typename arg<1>::type arg1_type;
        typedef typename arg<2>::type arg2_type;
        typedef typename arg<3>::type arg3_type;
        typedef typename arg<4>::type arg4_type;
        typedef typename arg<5>::type arg5_type;
        typedef typename arg<6>::type arg6_type;
        typedef typename arg<7>::type arg7_type;
        typedef typename arg<8>::type arg8_type;
        typedef typename arg<9>::type arg9_type;
        typedef typename arg<10>::type arg10_type;
        typedef typename arg<11>::type arg11_type;
        typedef typename arg<12>::type arg12_type;
        typedef typename arg<13>::type arg13_type;
        typedef typename arg<14>::type arg14_type;
        typedef typename arg<15>::type arg15_type;
        typedef typename arg<16>::type arg16_type;
        typedef typename arg<17>::type arg17_type;
        typedef typename arg<18>::type arg18_type;
        typedef typename arg<19>::type arg19_type;
        typedef typename arg<20>::type arg20_type;
    };
}

// ----------------------------------------------------------------------------------------


template <size_t I>
struct visit_impl
{
    template <typename T, typename F>
    static void visit(T& tup, size_t idx, F fun)
    {
        if (idx == I - 1) fun(std::get<I - 1>(tup));
        else visit_impl<I - 1>::visit(tup, idx, fun);
    }
};

template <>
struct visit_impl<0>
{
    template <typename T, typename F>
    static void visit(T& tup, size_t idx, F fun) { DLIB_CASSERT(false,"this should never happen"); }
};

template <typename F, typename... Ts>
void visit_at(std::tuple<Ts...> const& tup, size_t idx, F fun)
{
    visit_impl<sizeof...(Ts)>::visit(tup, idx, fun);
}

template <typename F, typename... Ts>
void visit_at(std::tuple<Ts...>& tup, size_t idx, F fun)
{
    visit_impl<sizeof...(Ts)>::visit(tup, idx, fun);
}

class mex_class_dispatch
{
public:
    mex_class_dispatch(
        MEX_CLASS_NAME* ptr_,
        int nlhs_, 
        mxArray** plhs_,
        int nrhs_, 
        const mxArray** prhs_
    ) : 
        ptr(ptr_),
        nlhs(nlhs_),
        plhs(plhs_),
        nrhs(nrhs_),
        prhs(prhs_)
    {}

    template <typename funct>
    void operator() (const funct& mfp)
    {
        mex_binding::call_mex_function(wrap_mex_class(*ptr,mfp), nlhs, plhs, nrhs, prhs);
    }

private:
    MEX_CLASS_NAME* ptr;
    int nlhs; 
    mxArray** plhs;
    int nrhs; 
    const mxArray** prhs;
};

class class_factory_type : dlib::noncopyable
{
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a container class for all the MEX_CLASS_NAME objects we create.  It allows
            us to track what we have created and make sure the MATLAB user doesn't do any
            double frees or use any stale pointers.   

            It also helps us deal with the problem that would otherwise arise when a mex file
            is unloaded from MATLAB when there are still active pointers to MEX_CLASS_NAME objects
            in MATLAB, since we will be able to detect stale pointers.
    !*/
public:

    class_factory_type()
    {
        seed = (uint64)time(0);
    }

    ~class_factory_type()
    {
        for (auto i : object_table)
            delete i.second;
    }

    template <typename ...T>
    uint64 create(T&& ...args)
    {
        MEX_CLASS_NAME* item = new MEX_CLASS_NAME(std::forward<T>(args)...);
        uint64 id = (uint64)item;
        // Now generate a unique id that incorporates our seed value. The point of doing
        // this is to avoid any chance that a mex file will get unloaded and then reloaded
        // and start constructing objects with the same addresses, while old stale objects
        // at those addresses are still stored in matlab, which would then call into the
        // mex file and make things go crazy.  So here we try to generate ID numbers that
        // are globally unique.
        uint64 i = 0;
        id = murmur_hash3_128bit_3(id, seed, ++i).first;
        // very unlikely but make sure there aren't any hash collisions.
        while(object_table.count(id) != 0)
            id = murmur_hash3_128bit_3(id, seed, ++i).first;

        object_table[id] = item;
        return id;
    }

    void free(uint64 item)
    {
        if (object_table.count(item) == 0)
        {
            throw dlib::error("An attempt to deallocate a mex class object with an invalid pointer was detected.");
        }

        delete object_table[item];
        object_table.erase(item);
    }

    MEX_CLASS_NAME* access(uint64 item) // convert numeric ID to pointer to object that can be used.
    {
        if (object_table.count(item) == 0)
        {
            throw dlib::error("An attempt to access a mex class object with an invalid pointer was detected.");
        }

        return object_table[item];
    }

private:

    std::map<uint64, MEX_CLASS_NAME*> object_table;
    uint64 seed;
} class_factory;

// ----------------------------------------------------------------------------------------

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X) 
#define FE_2(WHAT, X, ...) WHAT(X),FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X),FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X),FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X),FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X),FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X),FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X),FE_7(WHAT, __VA_ARGS__)
#define FE_9(WHAT, X, ...) WHAT(X),FE_8(WHAT, __VA_ARGS__)
#define FE_10(WHAT, X, ...) WHAT(X),FE_9(WHAT, __VA_ARGS__)
#define FE_11(WHAT, X, ...) WHAT(X),FE_10(WHAT, __VA_ARGS__)
#define FE_12(WHAT, X, ...) WHAT(X),FE_11(WHAT, __VA_ARGS__)
#define FE_13(WHAT, X, ...) WHAT(X),FE_12(WHAT, __VA_ARGS__)
#define FE_14(WHAT, X, ...) WHAT(X),FE_13(WHAT, __VA_ARGS__)
#define FE_15(WHAT, X, ...) WHAT(X),FE_14(WHAT, __VA_ARGS__)
#define FE_16(WHAT, X, ...) WHAT(X),FE_15(WHAT, __VA_ARGS__)
#define FE_17(WHAT, X, ...) WHAT(X),FE_16(WHAT, __VA_ARGS__)
#define FE_18(WHAT, X, ...) WHAT(X),FE_17(WHAT, __VA_ARGS__)
#define FE_19(WHAT, X, ...) WHAT(X),FE_18(WHAT, __VA_ARGS__)
#define FE_20(WHAT, X, ...) WHAT(X),FE_19(WHAT, __VA_ARGS__)
//... repeat as needed
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,NAME,...) NAME 
#define FOR_EACH(action,...) GET_MACRO(__VA_ARGS__,FE_20,FE_19,FE_18,FE_17,FE_16,FE_15,FE_14,FE_13,FE_12,FE_11,FE_10,FE_9,FE_8,FE_7,FE_6,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)
#define MEX_CLASS_ANNOTATE(x) &MEX_CLASS_NAME::x 

// Now make a tuple containing all the member function pointers to our MEX_CLASS_NAME
auto mex_class_methods = std::make_tuple(FOR_EACH(MEX_CLASS_ANNOTATE, MEX_CLASS_METHODS));


#endif // MEX_CLASS_NAME

// ----------------------------------------------------------------------------------------

bool is_string(const mxArray* arr, const char* str)
{
    if (mxIsChar(arr))
    {
        char ch[20];
        DLIB_CASSERT(mxGetString(arr, ch, sizeof(ch))==0, "Unable to retrieve string");
        ch[sizeof(ch)-1] = 0;// ensure NULL termination regardless of what MATLAB does.
        return strcmp(str,ch)==0;
    }
    return false;
}

// ----------------------------------------------------------------------------------------

/* The gateway function called by MATLAB*/
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    // Only remap cout and cerr if we aren't using octave since octave already does this.
#if !defined(OCTAVE_IMPORT) && !defined(OCTAVE_API)
    // make it so cout prints to mexPrintf()
    mex_binding::mex_streambuf sb;
    // make it so cerr prints to mexWarnMsgTxt()
    mex_binding::mex_warn_streambuf wsb;
#endif

    try
    {
#ifdef MEX_CLASS_NAME
        if (nrhs == 0)
        {
            #define DEF2STR(x) DEF2STR2((x))
            #define DEF2STR2(x) #x

            string classname = trim(string(DEF2STR(MEX_CLASS_NAME)), " \t()");
            std::vector<string> methods = split(trim(string(DEF2STR(MEX_CLASS_METHODS)), " \t()"), " \t,");

            string mex_filename = trim(string(DEF2STR(MEX_FILENAME))," \t()");
            bool has_load_obj = false;
            size_t load_obj_idx = 0;

            cout << "classdef " << classname << " < handle\n"
                 << "    properties (Access = private)\n"
                 << "        cpp_ptr\n"
                 << "    end\n"
                 << "\n"
                 << "    methods\n"
                 << "        function this = "<<classname<<"()\n"
                 << "            this.cpp_ptr = "<<mex_filename<<"('construct');\n"
                 << "        end\n"
                 << "\n"
                 << "        function copied_obj = clone(this)\n"
                 << "            %Returns a new independent object that is a copy of this.\n"
                 << "            copied_obj = "<<classname<<"();\n"
                 << "            copied_obj.cpp_ptr = "<<mex_filename<<"(this.cpp_ptr,'clone');\n"
                 << "        end\n"
                 << "\n";
            for (size_t i = 0; i < methods.size(); ++i)
            {
                if (methods[i] == "load_obj")
                {
                    has_load_obj = true;
                    load_obj_idx = i;
                }
                else
                {
                    cout << "        function varargout = "<<methods[i]<<"(this, varargin) \n"
                        << "            [varargout{1:nargout}] = "<<mex_filename<<"(this.cpp_ptr, "<<i+1<<", varargin{:}); \n"
                        << "        end \n\n";
                }
            }
            cout << "    end\n\n";

            cout << "    methods(Access=private) \n"
                << "        function delete(this) \n"
                << "            "<<mex_filename<<"(this.cpp_ptr); \n"
                << "        end         \n";
            if (has_load_obj)
            {
                cout << "        function varargout = load_obj(this, varargin) \n"
                    << "            [varargout{1:nargout}] = "<<mex_filename<<"(this.cpp_ptr, "<<load_obj_idx+1<<", varargin{:}); \n"
                    << "        end \n";
            }
            cout << "    end \n\n";

            if (has_load_obj)
            {
                cout << "    methods(Static) \n"
                    << "        function this = loadobj(in) \n"
                    << "            this = "<<classname<<"(); \n"
                    << "            this.load_obj(in); \n"
                    << "        end          \n"
                    << "    end \n";
            }
            cout << "end \n";
        }
        else if (nrhs == 1) 
        {
            // this is a constructor call
            if (is_string(prhs[0],"construct"))
            {
                DLIB_CASSERT(nlhs == 1, "If you want to construct a new object then you must assign the pointer to something.");
                plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
                uint64* ptr_int = (uint64*)mxGetData(plhs[0]);
                *ptr_int = class_factory.create();
            }
            else // destructor call
            {
                DLIB_CASSERT(mxIsUint64(prhs[0]) && mxGetNumberOfElements(prhs[0])==1, "When calling a class destructor the first argument must be a pointer (a UINT64 in matlab)");
                const uint64 ptr_int = *((uint64*)mxGetData(prhs[0]));
                class_factory.free(ptr_int);
            }
        }
        else // a regular function call
        {
            DLIB_CASSERT(mxIsUint64(prhs[0]) && mxGetNumberOfElements(prhs[0])==1, "When calling a class member function the first argument must be a pointer (a UINT64 in matlab)");
            if (is_string(prhs[1], "clone"))
            {
                DLIB_CASSERT(nlhs == 1, "If you want to construct a new object then you must assign the pointer to something.");
                const uint64 ptr_int = *((uint64*)mxGetData(prhs[0]));

                MEX_CLASS_NAME* ptr = class_factory.access(ptr_int);

                plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
                uint64* ptr_int2 = (uint64*)mxGetData(plhs[0]);
                // copy construct a new object
                *ptr_int2 = class_factory.create(*ptr);
            }
            else
            {
                DLIB_CASSERT(mxIsDouble(prhs[1]) && mxGetNumberOfElements(prhs[1])==1, "When calling a class member function the second argument must be a number indicating which member function");
                const uint64 ptr_int = *((uint64*)mxGetData(prhs[0]));
                const int funct_idx = *(mxGetPr(prhs[1]));

                auto num_registered_functions = std::tuple_size<decltype(mex_class_methods)>::value;
                DLIB_CASSERT(1 <= funct_idx && funct_idx <= num_registered_functions, "Invalid function index provided.");

                MEX_CLASS_NAME* ptr = class_factory.access(ptr_int);

                // we used the first two arguments to decide what function to call.  So adjust nrhs
                // and prhs so the member function never sees them.
                mex_class_dispatch dispatch(ptr, nlhs, plhs, nrhs-2, prhs+2);
                // now invoke the member function,  subtract 1 to convert to 0 indexing.
                visit_at(mex_class_methods, funct_idx-1, dispatch);
            }
        }
#else
        mex_binding::call_mex_function(mex_function, nlhs, plhs, nrhs, prhs);
#endif
    }
    catch (mex_binding::invalid_args_exception& e)
    {
        mexErrMsgIdAndTxt("mex_function:validate_and_populate_arg",
                            mex_binding::escape_percent(e.what()).c_str());
    }
    catch (mex_binding::user_hit_ctrl_c& )
    {
        // do nothing, just return to matlab
    }
    catch (std::exception& e)
    {
        mexErrMsgIdAndTxt("mex_function:error",
                            mex_binding::escape_percent(e.what()).c_str());
    }

    cout << flush;
    cerr << flush;
}

// ----------------------------------------------------------------------------------------

