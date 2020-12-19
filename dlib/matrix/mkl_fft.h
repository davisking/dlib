#ifndef DLIB_MKL_FFT_H
#define DLIB_MKL_FFT_H

#include <type_traits>
#include <mkl_dfti.h>

#define DLIB_DFTI_CHECK_STATUS(s) \
    if((s) != 0 && !DftiErrorClass((s), DFTI_NO_ERROR)) \
    { \
        throw dlib::error(DftiErrorMessage((s))); \
    }

namespace dlib
{
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    void mkl_fft(const std::vector<long>& dims, const std::complex<T>* in, std::complex<T>* out, bool is_inverse)
    {
        static constexpr DFTI_CONFIG_VALUE dfti_type = std::is_same<T,float>::value ? DFTI_SINGLE : DFTI_DOUBLE;

        DFTI_DESCRIPTOR_HANDLE h;
        MKL_LONG status;

        if (dims.size() == 1)
        {
            status = DftiCreateDescriptor(&h, dfti_type, DFTI_COMPLEX, 1, dims[0]);
            DLIB_DFTI_CHECK_STATUS(status);
        }
        else if (dims.size() == 2)
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
        else
        {
            throw dlib::error("Need to implement MKL 3D FFT");
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
            status = DftiComputeBackward(h, in, out);
        else
            status = DftiComputeForward(h, in, out);
        DLIB_DFTI_CHECK_STATUS(status);

        status = DftiFreeDescriptor(&h);
        DLIB_DFTI_CHECK_STATUS(status);
    }
//    
//    /*
//     *  in  has dims[0] * dims[1] * ... * dims[-2] * dims[-1] points
//     *  out has dims[0] * dims[1] * ... * dims[-2] * (dims[-1]/2+1) points
//     */
//    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
//    void mkl_fftr(std::vector<long> dims, const T* in, std::complex<T>* out)
//    {
//        static constexpr DFTI_CONFIG_VALUE dfti_type = std::is_same<T,float>::value ? DFTI_SINGLE : DFTI_DOUBLE;
//
//        DFTI_DESCRIPTOR_HANDLE h;
//        MKL_LONG status;
//
//        if (dims.size() == 1)
//        {
//            status = DftiCreateDescriptor(&h, dfti_type, DFTI_REAL, 1, dims[0]);
//            DLIB_DFTI_CHECK_STATUS(status);
//        }
//        else if (dims.size() == 2)
//        {
//            MKL_LONG size[] = {dims[0], dims[1]};
//            status = DftiCreateDescriptor(&h, dfti_type, DFTI_COMPLEX, 2, size);
//            DLIB_DFTI_CHECK_STATUS(status);
//
//            MKL_LONG strides[3];
//            strides[0] = 0;
//            strides[1] = size[1];
//            strides[2] = 1;
//
//            status = DftiSetValue(h, DFTI_INPUT_STRIDES, strides);
//            DLIB_DFTI_CHECK_STATUS(status);
//            status = DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides);
//            DLIB_DFTI_CHECK_STATUS(status);
//        }
//        else
//        {
//            throw dlib::error("Need to implement MKL 3D FFT");
//        }
//
//        const DFTI_CONFIG_VALUE inplacefft = in == out ? DFTI_INPLACE : DFTI_NOT_INPLACE;
//        status = DftiSetValue(h, DFTI_PLACEMENT, inplacefft);
//        DLIB_DFTI_CHECK_STATUS(status);
//
//        // Unless we use sequential mode, the fft results are not correct.
//        status = DftiSetValue(h, DFTI_THREAD_LIMIT, 1);
//        DLIB_DFTI_CHECK_STATUS(status);
//
//        status = DftiCommitDescriptor(h);
//        DLIB_DFTI_CHECK_STATUS(status);
//
//        if (is_inverse)
//            status = DftiComputeBackward(h, in, out);
//        else
//            status = DftiComputeForward(h, in, out);
//        DLIB_DFTI_CHECK_STATUS(status);
//
//        status = DftiFreeDescriptor(&h);
//        DLIB_DFTI_CHECK_STATUS(status);
//    }
}

#endif // DLIB_MKL_FFT_H