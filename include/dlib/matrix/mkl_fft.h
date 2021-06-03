#ifndef DLIB_MKL_FFT_H
#define DLIB_MKL_FFT_H

#include <type_traits>
#include <mkl_dfti.h>
#include "fft_size.h"

#define DLIB_DFTI_CHECK_STATUS(s) \
    if((s) != 0 && !DftiErrorClass((s), DFTI_NO_ERROR)) \
    { \
        throw dlib::error(DftiErrorMessage((s))); \
    }

namespace dlib
{
    template<typename T>
    void mkl_fft(const fft_size& dims, const std::complex<T>* in, std::complex<T>* out, bool is_inverse)
    /*!
        requires
            - T must be either float or double
            - dims represents the dimensions of both `in` and `out`
            - dims.num_dims() > 0
            - dims.num_dims() < 3
        ensures
            - performs an FFT on `in` and stores the result in `out`.
            - if `is_inverse` is true, a backward FFT is performed, 
              otherwise a forward FFT is performed.
    !*/
    {
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");

        constexpr DFTI_CONFIG_VALUE dfti_type = std::is_same<T,float>::value ? DFTI_SINGLE : DFTI_DOUBLE;

        DFTI_DESCRIPTOR_HANDLE h;
        MKL_LONG status;

        if (dims.num_dims() == 1)
        {
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_COMPLEX, 1, dims[0]);
            DLIB_DFTI_CHECK_STATUS(status);
        }
        else
        {
            MKL_LONG size[] = {dims[0], dims[1]};
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_COMPLEX, 2, size);
            DLIB_DFTI_CHECK_STATUS(status);

            MKL_LONG strides[3];
            strides[0] = 0;
            strides[1] = size[1];
            strides[2] = 1;

            status = DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
            DLIB_DFTI_CHECK_STATUS(status);
            status = DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
            DLIB_DFTI_CHECK_STATUS(status);
        }

        const DFTI_CONFIG_VALUE inplacefft = in == out ? DFTI_INPLACE : DFTI_NOT_INPLACE;
        status = DftiSetValue(h, DFTI_PLACEMENT, inplacefft);
        DLIB_DFTI_CHECK_STATUS(status);

        // Unless we use sequential mode, the fft results are not correct.
        status = DftiSetValue(h, DFTI_THREAD_LIMIT, 1);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiCommitDescriptor(h);
        DLIB_DFTI_CHECK_STATUS(status);

        if (is_inverse)
            status = DftiComputeBackward(h, (void*)in, (void*)out);
        else
            status = DftiComputeForward(h, (void*)in, (void*)out);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiFreeDescriptor(&h);
        DLIB_DFTI_CHECK_STATUS(status);
    }

    /*
     *  in  has dims[0] * dims[1] * ... * dims[-2] * dims[-1] points
     *  out has dims[0] * dims[1] * ... * dims[-2] * (dims[-1]/2+1) points
     */
    template<typename T>
    void mkl_fftr(const fft_size& dims, const T* in, std::complex<T>* out)
    /*!
        requires
            - T must be either float or double
            - dims represent the dimensions of `in`
            - `out` has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]/2+1}
            - dims.num_dims() > 0
            - dims.num_dims() <= 3
            - dims.back() must be even
        ensures
            - performs a real FFT on `in` and stores the result in `out`.
    !*/
    {
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");
        DLIB_ASSERT(dims.back() % 2 == 0, "last dimension needs to be even");
        
        constexpr DFTI_CONFIG_VALUE dfti_type = std::is_same<T,float>::value ? DFTI_SINGLE : DFTI_DOUBLE;

        DFTI_DESCRIPTOR_HANDLE h;
        MKL_LONG status;

        if (dims.num_dims() == 1)
        {
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_REAL, 1, dims[0]);
            DLIB_DFTI_CHECK_STATUS(status);
        }
        else
        {
            const long lastdim  = dims[1]/2+1;
            MKL_LONG size[] = {dims[0], dims[1]};
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_REAL, 2, size);
            DLIB_DFTI_CHECK_STATUS(status);

            {
                MKL_LONG strides[3];
                strides[0] = 0;
                strides[1] = size[1];
                strides[2] = 1;

                status = DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
                DLIB_DFTI_CHECK_STATUS(status);
            }
            {
                MKL_LONG strides[3];
                strides[0] = 0;
                strides[1] = lastdim;
                strides[2] = 1;
                status = DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
                DLIB_DFTI_CHECK_STATUS(status);
            }   
        }

        const DFTI_CONFIG_VALUE inplacefft = (void*)in == (void*)out ? DFTI_INPLACE : DFTI_NOT_INPLACE;
        status = DftiSetValue(h, DFTI_PLACEMENT, inplacefft);
        DLIB_DFTI_CHECK_STATUS(status);
        
        status = DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DLIB_DFTI_CHECK_STATUS(status);

        // Unless we use sequential mode, the fft results are not correct.
        status = DftiSetValue(h, DFTI_THREAD_LIMIT, 1);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiCommitDescriptor(h);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiComputeForward(h, (void*)in, (void*)out);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiFreeDescriptor(&h);
        DLIB_DFTI_CHECK_STATUS(status);
    }

    /*
     *  in  has dims[0] * dims[1] * ... * dims[-2] * (dims[-1]/2+1) points
     *  out has dims[0] * dims[1] * ... * dims[-2] * dims[-1] points
     */
    template<typename T>
    void mkl_ifftr(const fft_size& dims, const std::complex<T>* in, T* out)
    /*!
        requires
            - T must be either float or double
            - dims represent the dimensions of `out`
            - `in` has dimensions {dims[0], dims[1], ..., dims[-2], dims[-1]/2+1}
            - dims.num_dims() > 0
            - dims.num_dims() <= 3
            - dims.back() must be even
        ensures
            - performs an inverse real FFT on `in` and stores the result in `out`.
    !*/
    {
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");
        DLIB_ASSERT(dims.back() % 2 == 0, "last dimension needs to be even");

        constexpr DFTI_CONFIG_VALUE dfti_type = std::is_same<T,float>::value ? DFTI_SINGLE : DFTI_DOUBLE;

        DFTI_DESCRIPTOR_HANDLE h;
        MKL_LONG status;

        if (dims.num_dims() == 1)
        {
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_REAL, 1, dims[0]);
            DLIB_DFTI_CHECK_STATUS(status);
        }
        else
        {
            const long lastdim  = dims[1]/2+1;
            MKL_LONG size[] = {dims[0], dims[1]};
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_REAL, 2, size);
            DLIB_DFTI_CHECK_STATUS(status);

            {
                MKL_LONG strides[3];
                strides[0] = 0;
                strides[1] = lastdim;
                strides[2] = 1;

                status = DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
                DLIB_DFTI_CHECK_STATUS(status);
            }
            {
                MKL_LONG strides[3];
                strides[0] = 0;
                strides[1] = dims[1];
                strides[2] = 1;
                status = DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
                DLIB_DFTI_CHECK_STATUS(status);
            }   
        }

        const DFTI_CONFIG_VALUE inplacefft = (void*)in == (void*)out ? DFTI_INPLACE : DFTI_NOT_INPLACE;
        status = DftiSetValue(h, DFTI_PLACEMENT, inplacefft);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DLIB_DFTI_CHECK_STATUS(status);
        
        // Unless we use sequential mode, the fft results are not correct.
        status = DftiSetValue(h, DFTI_THREAD_LIMIT, 1);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiCommitDescriptor(h);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiComputeBackward(h, (void*)in, (void*)out);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiFreeDescriptor(&h);
        DLIB_DFTI_CHECK_STATUS(status);
    }
}

#endif // DLIB_MKL_FFT_H
