#ifndef DLIB_MKL_FFT_H
#define DLIB_MKL_FFT_H

#include <type_traits>
#include <memory>
#include <unordered_map>
#include <mkl_dfti.h>
#include "fft_size.h"

#ifdef DLIB_USE_MKL_WITH_TBB
// This is a workaround to make libdlib link to libtbb.so explicitly and adding its path to the build tree RPATH.
// This is because libmkl_tbb_threads.so depends on libtbb.so but dlib doesn't normally have any explicit symbols to libtbb.
// If you don't do this, the runtime path of libtbb.so is stripped after build time.
// Without this, at runtime you would get an error like "can't find libtbb.so"
// And you would have to manually set it via LD_LIBRARY_PATH or something.
// The better way to get around this is force cmake to add libtbb.so to RPATH. 
// But i can't find a way to do it cleanly. Hopefully there isn't much performance impact here.
extern "C" const char* TBB_runtime_version();
#endif

namespace dlib
{
    namespace mkl_details
    {

//----------------------------------------------------------------------------------------------------------------

        inline void check_status(MKL_LONG s)
        {
            if(s != 0 && !DftiErrorClass(s, DFTI_NO_ERROR))
                throw std::runtime_error(DftiErrorMessage(s));
        }

//----------------------------------------------------------------------------------------------------------------

        struct mkl_deleter { void operator()(DFTI_DESCRIPTOR* h) {DftiFreeDescriptor(&h);} };
        using mkl_ptr = std::unique_ptr<DFTI_DESCRIPTOR, mkl_deleter>;

//----------------------------------------------------------------------------------------------------------------

        struct plan_key
        {
            fft_size    size;
            bool        is_inplace{};
            bool        is_single_precision{};
            bool        is_complex{};
            bool        is_inverse{};
        };

//----------------------------------------------------------------------------------------------------------------

        inline bool operator==(const plan_key& a, const plan_key& b)
        {
            return  a.size                   == b.size                   && 
                    a.is_inplace             == b.is_inplace             && 
                    a.is_single_precision    == b.is_single_precision    && 
                    a.is_complex             == b.is_complex             &&
                    a.is_inverse             == b.is_inverse;
        }

//----------------------------------------------------------------------------------------------------------------
        
        struct hasher
        {
            uint32_t operator()(const plan_key& s) const noexcept
            {
                uint32_t hash = 0;
                hash = dlib::hash(s.size,                           hash);
                hash = dlib::hash((uint32_t)s.is_inplace,           hash);
                hash = dlib::hash((uint32_t)s.is_single_precision,  hash);
                hash = dlib::hash((uint32_t)s.is_complex,           hash);
                hash = dlib::hash((uint32_t)s.is_inverse,           hash);
                return hash;
            }
        };

//----------------------------------------------------------------------------------------------------------------

        inline const auto& get_handle(const plan_key& key)
        {
            thread_local std::unordered_map<plan_key, mkl_ptr, hasher> plans;

            if (plans.find(key) == plans.end())
            {
#ifdef DLIB_USE_MKL_WITH_TBB
                (void)TBB_runtime_version();
#endif

                const DFTI_CONFIG_VALUE dfti_type   = key.is_single_precision   ? DFTI_SINGLE   : DFTI_DOUBLE;
                const DFTI_CONFIG_VALUE inplacefft  = key.is_inplace            ? DFTI_INPLACE  : DFTI_NOT_INPLACE;
                const DFTI_CONFIG_VALUE domain      = key.is_complex            ? DFTI_COMPLEX  : DFTI_REAL;
                DFTI_DESCRIPTOR_HANDLE h;

                if (key.size.num_dims() == 1)
                {
                    check_status(DftiCreateDescriptor(&h, dfti_type, domain, 1, key.size[0]));
                }
                else if (key.is_complex)
                {
                    MKL_LONG size[]     = {key.size[0], key.size[1]};
                    MKL_LONG strides[]  = {0, size[1], 1};
                    check_status(DftiCreateDescriptor(&h, dfti_type, domain, 2, size));
                    check_status(DftiSetValue(h, DFTI_INPUT_STRIDES, strides));
                    check_status(DftiSetValue(h, DFTI_OUTPUT_STRIDES, strides));
                }
                else
                {
                    const long lastdim          = key.size[1]/2+1;
                    MKL_LONG size[]             = {key.size[0], key.size[1]};
                    MKL_LONG input_strides[]    = {0, size[1], 1};
                    MKL_LONG output_strides[]   = {0, lastdim, 1};

                    check_status(DftiCreateDescriptor(&h, dfti_type, domain, 2, size));
                    check_status(DftiSetValue(h, DFTI_INPUT_STRIDES,  !key.is_inverse ? input_strides  : output_strides));
                    check_status(DftiSetValue(h, DFTI_OUTPUT_STRIDES, !key.is_inverse ? output_strides : input_strides));
                    check_status(DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
                }

                check_status(DftiSetValue(h, DFTI_PLACEMENT, inplacefft));
                check_status(DftiSetValue(h, DFTI_THREAD_LIMIT, 1)); // Unless we use sequential mode, the fft results are not correct. 
                check_status(DftiCommitDescriptor(h));
                plans[key].reset(h);
            }
            
            return plans.at(key);
        }

//----------------------------------------------------------------------------------------------------------------

    }

//----------------------------------------------------------------------------------------------------------------

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
        using namespace mkl_details;
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");

        const auto& h = get_handle(plan_key{dims, in == out, std::is_same<T,float>::value, true, is_inverse});
        if (is_inverse) check_status(DftiComputeBackward(h.get(), (void*)in, (void*)out));
        else            check_status(DftiComputeForward(h.get(), (void*)in, (void*)out));
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
        using namespace mkl_details;
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");
        DLIB_ASSERT(dims.back() % 2 == 0, "last dimension needs to be even");

        const auto& h = get_handle(plan_key{dims, (void*)in == (void*)out, std::is_same<T,float>::value, false, false});
        check_status(DftiComputeForward(h.get(), (void*)in, (void*)out));
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
        using namespace mkl_details;
        static_assert(std::is_floating_point<T>::value, "template parameter needs to be a floatint point type");
        DLIB_ASSERT(dims.num_dims() > 0, "dims can't be empty");
        DLIB_ASSERT(dims.num_dims() < 3, "we currently only support up to 2D FFT. Please submit an issue on github if 3D or above is required.");
        DLIB_ASSERT(dims.back() % 2 == 0, "last dimension needs to be even");

        const auto& h = get_handle(plan_key{dims, (void*)in == (void*)out, std::is_same<T,float>::value, false, true});
        check_status(DftiComputeBackward(h.get(), (void*)in, (void*)out));
    }
}

#endif // DLIB_MKL_FFT_H
