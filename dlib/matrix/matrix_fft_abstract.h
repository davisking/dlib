// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FFt_ABSTRACT_Hh_
#ifdef DLIB_FFt_ABSTRACT_Hh_

#include "matrix_abstract.h"
#include "../algs.h"

namespace dlib
{
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    typename EXP::matrix_type fft (
        const matrix_exp<EXP>& data
    );  
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - Computes the 1 or 2 dimensional discrete Fourier transform of the given data
              matrix and returns it.  In particular, we return a matrix D such that:
                - D.nr() == data.nr()
                - D.nc() == data.nc()
                - D(0,0) == the DC term of the Fourier transform.
                - starting with D(0,0), D contains progressively higher frequency components
                  of the input data.
                - ifft(D) == data
    !*/
    
// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    matrix<std::complex<T>,0,1> fft (
        const std::vector<std::complex<T>, Alloc>& data
    );
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - Computes the 1 dimensional discrete Fourier transform of the given data
              vector and returns it.  In particular, we return a matrix D such that:
                - D.nr() == data.size()
                - D.nc() == 1
                - D(0,0) == the DC term of the Fourier transform.
                - starting with D(0,0), D contains progressively higher frequency components
                  of the input data.
                - ifft(D) == data
    !*/
    
// ----------------------------------------------------------------------------------------

    template <typename EXP>
    typename EXP::matrix_type ifft (
        const matrix_exp<EXP>& data
    );  
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - Computes the 1 or 2 dimensional inverse discrete Fourier transform of the
              given data vector and returns it.  In particular, we return a matrix D such
              that:
                - D.nr() == data.nr()
                - D.nc() == data.nc()
                - fft(D) == data 
    !*/

// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    matrix<std::complex<T>,0,1> ifft (
        const std::vector<std::complex<T>, Alloc>& data
    )
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - Computes the 1 dimensional inverse discrete Fourier transform of the
              given data vector and returns it.  In particular, we return a matrix D such
              that:
                - D.nr() == data.size()
                - D.nc() == 1
                - fft(D) == data 
    !*/
    
// ----------------------------------------------------------------------------------------
        
    template <typename EXP>
    matrix<add_complex_t<typename EXP::type>> fftr (
        const matrix_exp<EXP>& data
    );  
    /*!
        requires
            - data contains elements of type double, float, or long double.
            - data.nc() is even
        ensures
            - Computes the 1 or 2 dimensional real discrete Fourier transform of the given data
              matrix and returns it.  In particular, we return a matrix D such that:
                - D.nr() == data.nr()
                - D.nc() == fftr_nc_size(data.nc())
                - D(0,0) == the DC term of the Fourier transform.
                - starting with D(0,0), D contains progressively higher frequency components
                  of the input data.
                - ifftr(D) == data
    !*/

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    matrix<remove_complex_t<typename EXP::type>> ifftr (
        const matrix_exp<EXP>& data
    );  
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - Computes the 1 or 2 dimensional inverse real discrete Fourier transform of the
              given data vector and returns it.  In particular, we return a matrix D such
              that:
                - D.nr() == data.nr()
                - D.nc() == ifftr_nc_size(data.nc())
                - fftr(D) == data
    !*/

// ----------------------------------------------------------------------------------------
    
    template < 
        typename T, 
        long NR,
        long NC,
        typename MM,
        typename L 
        >
    void fft_inplace (
        matrix<std::complex<T>,NR,NC,MM,L>& data
    );
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - This function is identical to fft() except that it does the FFT in-place.
              That is, after this function executes we will have:
                - #data == fft(data)
    !*/
    
// ----------------------------------------------------------------------------------------

    template < 
        typename T, 
        long NR,
        long NC,
        typename MM,
        typename L 
        >
    void ifft_inplace (
        matrix<std::complex<T>,NR,NC,MM,L>& data
    );
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - This function is identical to ifft() except that it does the inverse FFT
              in-place.  That is, after this function executes we will have:
                - #data == ifft(data)*data.size()
                - Note that the output needs to be divided by data.size() to complete the 
                  inverse transformation.  
    !*/

// ----------------------------------------------------------------------------------------

    // These return function objects with signature double(size_t i, size_t wlen)
    // defining PERIODIC window functions suitable for passing to STFT functions

    inline function_object make_hann();
    inline function_object make_blackman();
    inline function_object make_blackman_nuttall();
    inline function_object make_blackman_harris();
    inline function_object make_blackman_harris7();
    inline function_object make_kaiser(beta_t beta);

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    matrix<complex_type> stft (
        const matrix_exp<EXP>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        requires
            - is_vector(signal) == true, i.e. signal has rank 1
            - signal.size() >= wlen
            - w is a function object with signature double(size_t i, size_t wlen) that defines a PERIODIC window,
              e.g. the output of make_hann().
            - fftsize >= wlen
            - wlen >= hoplen
            - EXP::type is a floating point type (float, double or long double), real or complex
        ensures
            - Performs a Short-Time-Fourier-Transform (STFT) on 1D data.
            - Returns a matrix D where first dimension correponds to time and second dimension corresponds to frequency.
            - Dimensions of D are:
                - D.nr() == (signal.size() + hoplen) / hoplen and corresponds to the number of time frames.
                - D.nc() == fftsize
            - The type of D is add_complex_t<EXP::type>
            - Each time frame t (equivalently, each row t) is centered on signal(t*hoplen)
            - This is equivalent to calling the following in python
              (provided w is converted into a string representation which scipy can interpret)
                win     = scipy.signal.get_window(w, wlen)
                scale   = win.sum()
                _, _, Z = scipy.signal.stft(signal, nfft=fftsize, nperseg=wlen, noverlap=(wlen-hoplen), window=win, return_onesided=False)
                Z       *= scale

    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename Alloc, typename WINDOW>
    matrix<complex_type> stft (
        const std::vector<T, Alloc>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        ensures
            - This is a shortcut to calling stft(dlib::mat(signal), w, fftsize, wlen, hoplen)
    !*/

// ----------------------------------------------------------------------------------------

    template <typename EXP,typename WINDOW>
    matrix<complex_type> istft (
        const matrix_exp<EXP>& stft,
        const WINDOW& w,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        requires
            - m has rank 2 where 1st dimension corresponds to time and second dimension corresponds to frequency
            - w is a function object with signature double(size_t i, size_t wlen) that defines a PERIODIC window,
              e.g. the output of make_hann().
            - wlen >= hoplen
            - EXP::type is a complex floating point type (complex<float>, complex<double> or complex<long double>)
            - If you wish to satisfy istft(stft(x, ...), ...) == x then:
                - w is the same as what was used with stft()
                - wlen is the same as what was used with stft()
                - hoplen is the same as what was used with stft()
        ensures
            - Performs an inverse Short-Time-Fourier-Transform (STFT)
            - istft(stft(x, w, wlen, wlen, hoplen), w, wlen, hoplen)) == x
            - istft(stft(x, w, fftsize, wlen, hoplen), w, wlen, hoplen)) == x
    !*/

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    matrix<complex_type> stftr (
        const matrix_exp<EXP>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        requires
            - is_vector(signal) == true, i.e. signal has rank 1
            - signal.size() >= wlen
            - w is a function object with signature double(size_t i, size_t wlen) that defines a PERIODIC window,
              e.g. the output of make_hann().
            - fftsize >= wlen
            - wlen >= hoplen
            - EXP::type is a floating point type (float, double or long double) and must be real
        ensures
            - Performs a real Short-Time-Fourier-Transform (STFTr) on 1D data.
            - Returns a matrix D where first dimension correponds to time and second dimension corresponds to frequency.
            - Dimensions of D are:
                - D.nr() == (signal.size() + hoplen) / hoplen and corresponds to the number of time frames.
                - D.nc() == fftsize/2 + 1
            - The type of D is add_complex_t<EXP::type>
            - Each time frame t (equivalently each row t) is centered on signal(t*hoplen)
            - This is equivalent to calling the follwoing in python
              (provided w is converted into a string representation which scipy can interpret)
                win     = scipy.signal.get_window(w, wlen)
                scale   = win.sum()
                _, _, Z = scipy.signal.stft(signal, nfft=fftsize, nperseg=wlen, noverlap=(wlen-hoplen), window=win, return_onesided=True)
                Z       *= scale

    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename Alloc, typename WINDOW>
    matrix<complex_type> stftr (
        const std::vector<T, Alloc>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        ensures
            - This is a shortcut to calling istft(dlib::mat(signal), w, wlen, hoplen)
    !*/

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    matrix<real_type> istftr (
        const matrix_exp<EXP>& stft,
        const WINDOW& w,
        std::size_t wlen,
        std::size_t hoplen
    );
    /*!
        requires
            - m has rank 2 where 1st dimension corresponds to time and second dimension corresponds to frequency
            - w is a function object with signature double(size_t i, size_t wlen) that defines a PERIODIC window,
              e.g. the output of make_hann().
            - wlen >= hoplen
            - EXP::type is a complex floating point type (complex<float>, complex<double> or complex<long double>)
            - If you wish to satisfy istftr(stftr(x, ...), ...) == x then:
                - w is the same as what was used with stftr()
                - wlen is the same as what was used with stftr()
                - hoplen is the same as what was used with stftr()
        ensures
            - Performs an inverse Short-Time-Fourier-Transform (STFT)
            - istftr(stftr(x, w, wlen, wlen, hoplen), w, wlen, hoplen)) == x
            - istftr(stftr(x, w, fftsize, wlen, hoplen), w, wlen, hoplen)) == x
    !*/
// ----------------------------------------------------------------------------------------
}

#endif // DLIB_FFt_ABSTRACT_Hh_

