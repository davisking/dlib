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
        mkl_fft({eval.nr(),eval.nc()}, &eval(0,0), &eval(0,0), false);
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
        mkl_fft({eval.nr(),eval.nc()}, &eval(0,0), &eval(0,0), true);
        eval /= data.size();
        return eval;
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        mkl_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        mkl_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), true);
    }

// ----------------------------------------------------------------------------------------

#else
     
// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    matrix<std::complex<T>,NR,NC,MM,L> fft (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        matrix<std::complex<T>,NR,NC,MM,L> out(in.nr(), in.nc());
        kiss_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), false);
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> fft (const matrix_exp<EXP>& data)
    {
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");
        matrix<typename EXP::type> in(data);
        return fft(in);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    matrix<std::complex<T>,NR,NC,MM,L> ifft (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        matrix<std::complex<T>,NR,NC,MM,L> out(in.nr(), in.nc());
        if (in.size() != 0)
        {
            kiss_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), true);
            out /= out.size();
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<typename EXP::type> ifft (const matrix_exp<EXP>& data)
    {
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");
        matrix<typename EXP::type> in(data);
        return ifft(in);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        kiss_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        kiss_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), true);
    }
    
// ----------------------------------------------------------------------------------------
    
    constexpr long fftr_nc_size(long nc)
    {
        return nc == 0 ? 0 : nc/2+1;
    }
    
    constexpr long ifftr_nc_size(long nc)
    {
        return nc == 0 ? 0 : 2*(nc-1);
    }
  
// ----------------------------------------------------------------------------------------
    
    template<typename T, long NR, long NC, typename MM, typename L>
    matrix<std::complex<T>,NR,fftr_nc_size(NC),MM,L> fftr (const matrix<T,NR,NC,MM,L>& in)
    {
        DLIB_ASSERT(in.nc() % 2 == 0, "last dimension needs to be even otherwise ifftr(fftr(data)) won't have matching dimensions : " << in.nc());
        matrix<std::complex<T>,NR,fftr_nc_size(NC),MM,L> out(in.nr(), fftr_nc_size(in.nc()));
        kiss_fftr({in.nr(),in.nc()}, &in(0,0), &out(0,0));
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<std::complex<typename EXP::type>> fftr (const matrix_exp<EXP>& data)
    {
        static_assert(std::is_floating_point<typename EXP::type>::value, "input should be a real floating point type.");
        matrix<typename EXP::type> in(data);
        return fftr(in);
    }

// ----------------------------------------------------------------------------------------
    
    template<typename T, long NR, long NC, typename MM, typename L>
    matrix<T,NR,ifftr_nc_size(NC),MM,L> ifftr (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        matrix<T,NR,ifftr_nc_size(NC),MM,L> out(in.nr(), ifftr_nc_size(in.nc()));
        if (in.size() != 0)
        {
            kiss_fftri({out.nr(),out.nc()}, &in(0,0), &out(0,0));
            out /= out.size();
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<remove_complex_t<typename EXP::type>> ifftr (const matrix_exp<EXP>& data)
    {
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");        
        matrix<typename EXP::type> in(data);
        return ifftr(in);
    }

// ----------------------------------------------------------------------------------------
    
#endif // DLIB_USE_MKL_FFT
}

#endif // DLIB_FFt_Hh_

