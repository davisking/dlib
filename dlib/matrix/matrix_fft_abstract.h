// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FFt_ABSTRACT_Hh_
#ifdef DLIB_FFt_ABSTRACT_Hh_

#include "matrix_abstract.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    constexpr bool is_power_of_two (
        const unsigned long value
    );
    /*!
        ensures
            - returns true if value contains a power of two and false otherwise.  As a
              special case, we also consider 0 to be a power of two.
    !*/

// ----------------------------------------------------------------------------------------
   
    constexpr long fftr_nc_size(
        long nc
    );
    /*!
        ensures
            - returns the output dimension of a 1D real FFT
    !*/

// ----------------------------------------------------------------------------------------
    
    constexpr long ifftr_nc_size(
        long nc
    );
    /*!
        ensures
            - returns the output dimension of an inverse 1D real FFT
    !*/
    
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
    
    template < typename T, typename Alloc >
    void fft_inplace (
        std::vector<std::complex<T>, Alloc>& data
    )
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

    template < typename T, typename Alloc >
    void ifft_inplace (
        std::vector<std::complex<T>, Alloc>& data
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
}

#endif // DLIB_FFt_ABSTRACT_Hh_

