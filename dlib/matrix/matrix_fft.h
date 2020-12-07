// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FFt_Hh_
#define DLIB_FFt_Hh_

#include "matrix_fft_abstract.h"
#include "matrix_utilities.h"
#include "../hash.h"
#include "../algs.h"

#ifdef DLIB_USE_MKL_FFT
#include "mkl_fft.h"
#else
#include "kiss_fft.h"
#endif

namespace dlib
{
#ifdef DLIB_USE_MKL_FFT
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> fft (const matrix_exp<EXP>& data)
    {
        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        matrix<typename EXP::type> eval(data); //potentially 2 copies: 1 for evaluating the matrix expression and 1 for doing the in-place FFT. hmm...
        mkl_fft({(int)eval.nr(), (int)eval.nc()}, &eval(0,0), &eval(0,0), false);
        return eval;
    }

// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> ifft (const matrix_exp<EXP>& data)
    {
        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        matrix<typename EXP::type> eval;
        if (data.size() == 0)
            return eval;
        eval = data;
        mkl_fft({(int)eval.nr(), (int)eval.nc()}, &eval(0,0), &eval(0,0), true);
        eval /= data.size();
        return eval;
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        mkl_fft({(int)data.nr(),(int)data.nc()}, &data(0,0), &data(0,0), false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        mkl_fft({(int)data.nr(), (int)data.nc()}, &data(0,0), &data(0,0), true);
    }

// ----------------------------------------------------------------------------------------

#else
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> fft (const matrix_exp<EXP>& data)
    {
        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        matrix<typename EXP::type> eval(data); //potentially 2 copies: 1 for evaluating the matrix expression and 1 for doing the out-of-place FFT. hmm...
        kiss_fft({(int)eval.nr(), (int)eval.nc()}, &eval(0,0), &eval(0,0), false);
        return eval;
    }

// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> ifft (const matrix_exp<EXP>& data)
    {
        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        matrix<typename EXP::type> eval;
        if (data.size() == 0)
            return eval;
        eval = data;
        kiss_fft({(int)eval.nr(),(int)eval.nc()}, &eval(0,0), &eval(0,0), true);
        eval /= data.size();
        return eval;
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        kiss_fft({(int)data.nr(), (int)data.nc()}, &data(0,0), &data(0,0), false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        kiss_fft({(int)data.nr(), (int)data.nc()}, &data(0,0), &data(0,0), true);
    }

// ----------------------------------------------------------------------------------------
    
#endif // DLIB_USE_MKL_FFT
}

#endif // DLIB_FFt_Hh_

