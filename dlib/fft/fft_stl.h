// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FFT_STL_Hh_
#define DLIB_FFT_STL_Hh_

#include <vector>
#include "fft.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    void fft_inplace (std::vector<std::complex<T>, Alloc>& data)
    /*!
        requires
            - data contains elements of type std::complex<> that itself contains double, float, or long double.
        ensures
            - This function is identical to fft() except that it does the FFT in-place.
              That is, after this function executes we will have:
                - #data == fft(data)
    !*/
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
            fft({(long)data.size()}, &data[0], &data[0], false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, typename Alloc >
    void ifft_inplace (std::vector<std::complex<T>, Alloc>& data)
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
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
            fft({(long)data.size()}, &data[0], &data[0], true);
    }
    
// ----------------------------------------------------------------------------------------

}

#endif //DLIB_FFT_STL_Hh_