#include "fft.h"

#ifdef DLIB_USE_MKL_FFT
#include "mkl_fft.h"
#else
#include "kiss_fft.h"
#endif

namespace dlib
{
    template<typename T>
    void fft(const fft_size& dims, const std::complex<T>* in, std::complex<T>* out, bool is_inverse)
    {
#ifdef DLIB_USE_MKL_FFT
        mkl_fft(dims, in, out, is_inverse);
#else
        kiss_fft(dims, in, out, is_inverse);
#endif
    }

    template<typename T>
    void fftr(const fft_size& dims, const T* in, std::complex<T>* out)
    {
#ifdef DLIB_USE_MKL_FFT
        mkl_fftr(dims, in, out);
#else
        kiss_fftr(dims, in, out);
#endif
    }

    template<typename T>
    void ifftr(const fft_size& dims, const std::complex<T>* in, T* out)
    {
#ifdef DLIB_USE_MKL_FFT
        mkl_ifftr(dims, in, out);
#else
        kiss_ifftr(dims, in, out);
#endif
    }

    template void fft<float>(const fft_size& dims, const std::complex<float>* in, std::complex<float>* out, bool is_inverse);
    template void fft<double>(const fft_size& dims, const std::complex<double>* in, std::complex<double>* out, bool is_inverse);

    template void fftr<float>(const fft_size& dims, const float* in, std::complex<float>* out);
    template void fftr<double>(const fft_size& dims, const double* in, std::complex<double>* out);

    template void ifftr<float>(const fft_size& dims, const std::complex<float>* in, float* out);
    template void ifftr<double>(const fft_size& dims, const std::complex<double>* in, double* out);
}