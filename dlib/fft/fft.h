// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FFT_DETAILS_Hh_
#define DLIB_FFT_DETAILS_Hh_

#include <complex>
#include "fft_size.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------

    constexpr bool is_power_of_two (
        const unsigned long n
    )
    /*!
        ensures
            - returns true if value contains a power of two and false otherwise.  As a
              special case, we also consider 0 to be a power of two.
    !*/
    {
        return n == 0 ? true : (n & (n - 1)) == 0;
    }

// ----------------------------------------------------------------------------------------
   
    constexpr long fftr_nc_size(
        long nc
    )
    /*!
        ensures
            - returns the output dimension of a 1D real FFT
    !*/
    {
        return nc == 0 ? 0 : nc/2+1;
    }

// ----------------------------------------------------------------------------------------
    
    constexpr long ifftr_nc_size(
        long nc
    )
    /*!
        ensures
            - returns the output dimension of an inverse 1D real FFT
    !*/
    {
        return nc == 0 ? 0 : 2*(nc-1);
    }
    
// ----------------------------------------------------------------------------------------

    template<typename T>
    void fft(const fft_size& dims, const std::complex<T>* in, std::complex<T>* out, bool is_inverse);
    /*!
        requires
            - T must be either float or double
            - dims represents the dimensions of both `in` and `out`
            - dims.num_dims() > 0
        ensures
            - performs an FFT on `in` and stores the result in `out`.
            - if `is_inverse` is true, a backward FFT is performed, 
              otherwise a forward FFT is performed.
    !*/

// ----------------------------------------------------------------------------------------

    template<typename T>
    void fftr(const fft_size& dims, const T* in, std::complex<T>* out);
    /*!
        requires
            - T must be either float or double
            - dims represent the dimensions of `in`
            - `in`  has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]}
            - `out` has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]/2+1}
            - dims.num_dims() > 0
            - dims.back() must be even
        ensures
            - performs a real FFT on `in` and stores the result in `out`.
    !*/

// ----------------------------------------------------------------------------------------

    template<typename T>
    void ifftr(const fft_size& dims, const std::complex<T>* in, T* out);
    /*!
        requires
            - T must be either float or double
            - dims represent the dimensions of `out`
            - `in`  has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]/2+1}
            - `out` has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]}
            - dims.num_dims() > 0
            - dims.back() must be even
        ensures
            - performs an inverse real FFT on `in` and stores the result in `out`.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif //DLIB_FFT_DETAILS_Hh_