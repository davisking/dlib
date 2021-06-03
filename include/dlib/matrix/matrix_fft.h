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
// ----------------------------------------------------------------------------------------
    
    constexpr bool is_power_of_two (const unsigned long n)
    {
        return n == 0 ? true : (n & (n - 1)) == 0;
    }
    
// ----------------------------------------------------------------------------------------
    
    constexpr long fftr_nc_size(long nc)
    {
        return nc == 0 ? 0 : nc/2+1;
    }
    
// ----------------------------------------------------------------------------------------   
    
    constexpr long ifftr_nc_size(long nc)
    {
        return nc == 0 ? 0 : 2*(nc-1);
    }
  
// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    matrix<std::complex<T>,0,1> fft (const std::vector<std::complex<T>, Alloc>& in)
    {
        //complex FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<std::complex<T>,0,1> out(in.size());
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({(long)in.size()}, &in[0], &out(0,0), false);
#else
            kiss_fft({(long)in.size()}, &in[0], &out(0,0), false);
#endif
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, long NR, long NC, typename MM, typename L >
    matrix<std::complex<T>,NR,NC,MM,L> fft (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        //complex FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<std::complex<T>,NR,NC,MM,L> out(in.nr(), in.nc());
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), false);
#else
            kiss_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), false);
#endif
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    typename EXP::matrix_type fft (const matrix_exp<EXP>& data)
    {
        //complex FFT for expression template
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");
        typename EXP::matrix_type in(data);
        return fft(in);
    }
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    matrix<std::complex<T>,0,1> ifft (const std::vector<std::complex<T>, Alloc>& in)
    {
        //complex FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<std::complex<T>,0,1> out(in.size());
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({(long)in.size()}, &in[0], &out(0,0), true);
#else
            kiss_fft({(long)in.size()}, &in[0], &out(0,0), true);
#endif
            out /= out.size();
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, long NR, long NC, typename MM, typename L >
    matrix<std::complex<T>,NR,NC,MM,L> ifft (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        //inverse complex FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<std::complex<T>,NR,NC,MM,L> out(in.nr(), in.nc());
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), true);
#else
            kiss_fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), true);
#endif
            out /= out.size();
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    typename EXP::matrix_type ifft (const matrix_exp<EXP>& data)
    {
        //inverse complex FFT for expression template
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");
        typename EXP::matrix_type in(data);
        return ifft(in);
    }

// ----------------------------------------------------------------------------------------
    
    template<typename T, long NR, long NC, typename MM, typename L>
    matrix<std::complex<T>,NR,fftr_nc_size(NC),MM,L> fftr (const matrix<T,NR,NC,MM,L>& in)
    {
        //real FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        DLIB_ASSERT(in.nc() % 2 == 0, "last dimension " << in.nc() << " needs to be even otherwise ifftr(fftr(data)) won't have matching dimensions");
        matrix<std::complex<T>,NR,fftr_nc_size(NC),MM,L> out(in.nr(), fftr_nc_size(in.nc()));
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fftr({in.nr(),in.nc()}, &in(0,0), &out(0,0));
#else
            kiss_fftr({in.nr(),in.nc()}, &in(0,0), &out(0,0));
#endif
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<add_complex_t<typename EXP::type>> fftr (const matrix_exp<EXP>& data)
    {
        //real FFT for expression template
        static_assert(std::is_floating_point<typename EXP::type>::value, "input should be real");
        matrix<typename EXP::type> in(data);
        return fft(in);
    }
    
// ----------------------------------------------------------------------------------------
    
    template<typename T, long NR, long NC, typename MM, typename L>
    matrix<T,NR,ifftr_nc_size(NC),MM,L> ifftr (const matrix<std::complex<T>,NR,NC,MM,L>& in)
    {
        //inverse real FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<T,NR,ifftr_nc_size(NC),MM,L> out(in.nr(), ifftr_nc_size(in.nc()));
        if (in.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_ifftr({out.nr(),out.nc()}, &in(0,0), &out(0,0));
#else
            kiss_ifftr({out.nr(),out.nc()}, &in(0,0), &out(0,0));
#endif
            out /= out.size();
        }
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<remove_complex_t<typename EXP::type>> ifftr (const matrix_exp<EXP>& data)
    {
        //inverse real FFT for expression template
        static_assert(is_complex<typename EXP::type>::value, "input should be complex");        
        matrix<typename EXP::type> in(data);
        return ifftr(in);
    }
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    void fft_inplace (std::vector<std::complex<T>, Alloc>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({(long)data.size()}, &data[0], &data[0], false);
#else
            kiss_fft({(long)data.size()}, &data[0], &data[0], false);
#endif
        }
    }
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), false);
#else
            kiss_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), false);
#endif
        }
    }

// ----------------------------------------------------------------------------------------

    template < typename T, typename Alloc >
    void ifft_inplace (std::vector<std::complex<T>, Alloc>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({(long)data.size()}, &data[0], &data[0], true);
#else
            kiss_fft({(long)data.size()}, &data[0], &data[0], true);
#endif
        }
    }
    
// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
        {
#ifdef DLIB_USE_MKL_FFT
            mkl_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), true);
#else
            kiss_fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), true);
#endif
        }
    }

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_FFt_Hh_

