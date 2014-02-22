// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FFt_H__
#define DLIB_FFt_H__

#include "matrix_fft_abstract.h"
#include "matrix_utilities.h"
#include "../hash.h"
#include "../algs.h"

#ifdef DLIB_USE_FFTW
#include <fftw3.h>
#endif // DLIB_USE_FFTW

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline unsigned long reverse_bits (
            unsigned long val,
            unsigned long num
        )
        {
            unsigned long temp = 0;
            for (unsigned long i = 0; i < num; ++i)
            {
                temp <<= 1;
                temp |= val&0x1;
                val >>= 1;
            }
            return temp;
        }

        template <typename EXP>
        void permute (
            const matrix_exp<EXP>& data, 
            typename EXP::matrix_type& outdata 
        )  
        {
            outdata.set_size(data.size());
            if (data.size() == 0)
                return;

            const unsigned long num = static_cast<unsigned long>(std::log((double)data.size())/std::log(2.0) + 0.5);
            for (unsigned long i = 0; i < (unsigned long)data.size(); ++i)
            {
                outdata(impl::reverse_bits(i,num)) = data(i);
            }
        }

    }

// ----------------------------------------------------------------------------------------

    inline bool is_power_of_two (
        const unsigned long& value
    )
    {
        if (value == 0)
            return true;
        else
            return count_bits(value) == 1;
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    typename EXP::matrix_type fft (
        const matrix_exp<EXP>& data
    )  
    {
        if (data.size() == 0)
            return data;

        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        // make sure requires clause is not broken
        DLIB_CASSERT(is_vector(data) && is_power_of_two(data.size()),
            "\t void ifft(data)"
            << "\n\t data must be a vector with a size that is a power of two."
            << "\n\t is_vector(data): " << is_vector(data)
            << "\n\t data.size():     " << data.size()
            );

        typedef typename EXP::type::value_type T;

        typename EXP::matrix_type outdata(data);

        const long half = outdata.size()/2;

        typedef std::complex<T> ct;
        matrix<ct,0,1,typename EXP::mem_manager_type> twiddle_factors(half);

        // compute the complex root of unity w
        const T temp = -2.0*pi/outdata.size();
        ct w = ct(std::cos(temp),std::sin(temp));

        ct w_pow = 1;

        // compute the twiddle factors
        for (long j = 0; j < twiddle_factors.size(); ++j)
        {
            twiddle_factors(j) = w_pow; 
            w_pow *= w;
        }


        // now compute the decimation in frequency.  This first
        // outer loop loops log2(outdata.size()) number of times
        long skip = 1;
        for (long step = half; step != 0; step >>= 1)
        {
            // do blocks of butterflies in this loop
            for (long j = 0; j < outdata.size(); j += step*2)
            {
                // do step butterflies
                for (long k = 0; k < step; ++k)
                {
                    const long a_idx = j+k;
                    const long b_idx = j+k+step;
                    const ct a = outdata(a_idx) + outdata(b_idx);
                    const ct b = (outdata(a_idx) - outdata(b_idx))*twiddle_factors(k*skip);
                    outdata(a_idx) = a;
                    outdata(b_idx) = b;
                }
            }
            skip *= 2;
        }

        typename EXP::matrix_type outperm;
        impl::permute(outdata, outperm);
        return outperm;
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    typename EXP::matrix_type ifft (
        const matrix_exp<EXP>& data
    )  
    {
        if (data.size() == 0)
            return data;

        // You have to give a complex matrix
        COMPILE_TIME_ASSERT(is_complex<typename EXP::type>::value);
        // make sure requires clause is not broken
        DLIB_CASSERT(is_vector(data) && is_power_of_two(data.size()),
            "\t void ifft(data)"
            << "\n\t data must be a vector with a size that is a power of two."
            << "\n\t is_vector(data): " << is_vector(data)
            << "\n\t data.size():     " << data.size()
            );


        typedef typename EXP::type::value_type T;

        typename EXP::matrix_type outdata;
        impl::permute(data,outdata);

        const long half = outdata.size()/2;

        typedef std::complex<T> ct;
        matrix<ct,0,1,typename EXP::mem_manager_type> twiddle_factors(half);

        // compute the complex root of unity w
        const T temp = 2.0*pi/outdata.size();
        ct w = ct(std::cos(temp),std::sin(temp));

        ct w_pow = 1;

        // compute the twiddle factors
        for (long j = 0; j < twiddle_factors.size(); ++j)
        {
            twiddle_factors(j) = w_pow; 
            w_pow *= w;
        }

        // now compute the inverse decimation in frequency.  This first
        // outer loop loops log2(outdata.size()) number of times
        long skip = half;
        for (long step = 1; step <= half; step <<= 1)
        {
            // do blocks of butterflies in this loop
            for (long j = 0; j < outdata.size(); j += step*2)
            {
                // do step butterflies
                for (long k = 0; k < step; ++k)
                {
                    const long a_idx = j+k;
                    const long b_idx = j+k+step;
                    outdata(b_idx) *= twiddle_factors(k*skip);
                    const ct a = outdata(a_idx) + outdata(b_idx);
                    const ct b = outdata(a_idx) - outdata(b_idx);
                    outdata(a_idx) = a;
                    outdata(b_idx) = b;
                }
            }
            skip /= 2;
        }

        outdata /= outdata.size();
        return outdata;
    }

// ----------------------------------------------------------------------------------------

#ifdef DLIB_USE_FFTW

    template <long NR, long NC, typename MM, typename L>
    matrix<std::complex<double>,NR,NC,MM,L> call_fftw_fft(
        const matrix<std::complex<double>,NR,NC,MM,L>& data
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(is_vector(data) && is_power_of_two(data.size()),
            "\t void fft(data)"
            << "\n\t data must be a vector with a size that is a power of two."
            << "\n\t is_vector(data): " << is_vector(data)
            << "\n\t data.size():     " << data.size()
            );

        matrix<std::complex<double>,NR,NC,MM,L> m2(data.nr(),data.nc());
        fftw_complex *in, *out;
        fftw_plan p;
        in = (fftw_complex*)&data(0);
        out = (fftw_complex*)&m2(0);
        p = fftw_plan_dft_1d(data.size(), in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p); 
        fftw_destroy_plan(p);
        return m2;
    }

    template <long NR, long NC, typename MM, typename L>
    matrix<std::complex<double>,NR,NC,MM,L> call_fftw_ifft(
        const matrix<std::complex<double>,NR,NC,MM,L>& data
    )
    {
        // make sure requires clause is not broken
        DLIB_CASSERT(is_vector(data) && is_power_of_two(data.size()),
            "\t void ifft(data)"
            << "\n\t data must be a vector with a size that is a power of two."
            << "\n\t is_vector(data): " << is_vector(data)
            << "\n\t data.size():     " << data.size()
            );

        matrix<std::complex<double>,NR,NC,MM,L> m2(data.nr(),data.nc());
        fftw_complex *in, *out;
        fftw_plan p;
        in = (fftw_complex*)&data(0);
        out = (fftw_complex*)&m2(0);
        p = fftw_plan_dft_1d(data.size(), in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p); 
        fftw_destroy_plan(p);
        return m2/data.size();
    }

// ----------------------------------------------------------------------------------------

// call FFTW for these cases:
    inline matrix<std::complex<double>,0,1> fft (const matrix<std::complex<double>,0,1>& data) {return call_fftw_fft(data);}
    inline matrix<std::complex<double>,0,1> ifft(const matrix<std::complex<double>,0,1>& data) {return call_fftw_ifft(data);}
    inline matrix<std::complex<double>,1,0> fft (const matrix<std::complex<double>,1,0>& data) {return call_fftw_fft(data);}
    inline matrix<std::complex<double>,1,0> ifft(const matrix<std::complex<double>,1,0>& data) {return call_fftw_ifft(data);}
    inline matrix<std::complex<double> > fft (const matrix<std::complex<double> >& data) {return call_fftw_fft(data);}
    inline matrix<std::complex<double> > ifft(const matrix<std::complex<double> >& data) {return call_fftw_ifft(data);}

#endif // DLIB_USE_FFTW

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FFt_H__

