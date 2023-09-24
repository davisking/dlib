// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FFt_Hh_
#define DLIB_FFt_Hh_

#include "matrix_fft_abstract.h"
#include "matrix_utilities.h"
#include "../hash.h"
#include "../algs.h"
#include "../math.h"
#include "../fft/fft.h"
#include "../fft/fft_stl.h"

namespace dlib
{     
  
// ----------------------------------------------------------------------------------------
    
    template < typename T, typename Alloc >
    matrix<std::complex<T>,0,1> fft (const std::vector<std::complex<T>, Alloc>& in)
    {
        //complex FFT
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        matrix<std::complex<T>,0,1> out(in.size());
        if (in.size() != 0)
            fft({(long)in.size()}, &in[0], &out(0,0), false);
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
            fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), false);
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
            fft({(long)in.size()}, &in[0], &out(0,0), true);
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
            fft({in.nr(),in.nc()}, &in(0,0), &out(0,0), true);
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
            fftr({in.nr(),in.nc()}, &in(0,0), &out(0,0));
        return out;
    }
    
// ----------------------------------------------------------------------------------------
    
    template <typename EXP>
    matrix<add_complex_t<typename EXP::type>> fftr (const matrix_exp<EXP>& data)
    {
        //real FFT for expression template
        static_assert(std::is_floating_point<typename EXP::type>::value, "input should be real");
        matrix<typename EXP::type> in(data);
        return fftr(in);
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
            ifftr({out.nr(),out.nc()}, &in(0,0), &out(0,0));
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
    
    template < typename T, long NR, long NC, typename MM, typename L >
    void fft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
            fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), false);
    }

// ----------------------------------------------------------------------------------------

    template < typename T, long NR, long NC, typename MM, typename L >
    void ifft_inplace (matrix<std::complex<T>,NR,NC,MM,L>& data)
    {
        static_assert(std::is_floating_point<T>::value, "only support floating point types");
        if (data.size() != 0)
            fft({data.nr(),data.nc()}, &data(0,0), &data(0,0), true);
    }

// ----------------------------------------------------------------------------------------

    namespace details
    {
        struct fft_func
        {
            template<typename MAT, typename T = typename MAT::type, typename std::enable_if<is_complex<T>::value, bool>::type = true>
            auto operator()(const MAT& mat) const { return dlib::fft(mat); }

            template<typename MAT, typename T = typename MAT::type, typename std::enable_if<!is_complex<T>::value, bool>::type = true>
            auto operator()(const MAT& mat) const { return dlib::fft(dlib::complex_matrix(mat)); }

            static constexpr std::size_t freqsize(std::size_t fftsize) { return fftsize; }
        };

        struct fftr_func
        {
            template<typename MAT>
            auto operator()(const MAT& mat) const { return dlib::fftr(mat); }

            static constexpr std::size_t freqsize(std::size_t fftsize) { return dlib::fftr_nc_size(fftsize); }
        };

        struct ifft_func
        {
            template<typename MAT>
            auto operator()(const MAT& mat) const { return dlib::ifft(mat); }
        };

        struct ifftr_func
        {
            template<typename MAT>
            auto operator()(const MAT& mat) const { return dlib::ifftr(mat); }
        };

        template <
            typename EXP,
            typename WINDOW,
            typename FFT_FUNC
        >
        auto stft_impl (
            const matrix_exp<EXP>& signal,
            const WINDOW& w,
            std::size_t fftsize,
            std::size_t wlen,
            std::size_t hoplen,
            const FFT_FUNC& fft_obj
        )
        {
            using T = typename EXP::type;
            using R = remove_complex_t<T>;
            using C = add_complex_t<T>;

            static_assert(std::is_floating_point<R>::value, "underlying type must be real or complex floating point type");
            DLIB_ASSERT(is_vector(signal), "input must be a vector type");
            DLIB_ASSERT(signal.size() >= (long)wlen, "signal.size() >= wlen not satisfied");
            DLIB_ASSERT(fftsize >= wlen, "fftsize >= wlen not satisfied");
            DLIB_ASSERT(wlen >= hoplen, "wlen >= hoplen not satisfied");

            // Input is left-padded by wlen/2 and right-padded wlen/2
            const std::size_t total_padding = wlen;
            const std::size_t overlap       = wlen - hoplen;
            const std::size_t nframes       = (signal.size() + total_padding - overlap) / hoplen;
            matrix<C> stft = zeros_matrix<C>(nframes,
                                             FFT_FUNC::freqsize(fftsize));
            matrix<R> win(1,wlen);
            for (std::size_t i = 0 ; i < wlen ; ++i)
                win(0, i) = w(i, wlen);

            // TODO: reduce extra buffers, e.g. padded
            matrix<T> padded;

            if (is_row_vector(signal))
                padded = join_rows(join_rows(zeros_matrix<T>(1, wlen/2), signal), zeros_matrix<T>(1, wlen/2));
            else
                padded = join_rows(join_rows(zeros_matrix<T>(1, wlen/2), trans(signal)), zeros_matrix<T>(1, wlen/2));

            for (long i = 0 ; i < stft.nr() ; ++i)
            {
                set_rowm(stft, i) = fft_obj(join_rows(pointwise_multiply(win, subm(padded, 0, i*hoplen, 1, wlen)),
                                                      zeros_matrix<T>(1, fftsize - wlen)));
            }

            return stft;
        }

        template <
            typename ReturnType,
            typename EXP,
            typename WINDOW,
            typename IFFT_FUNC
        >
        auto istft_impl (
            const matrix_exp<EXP>& stft,
            const WINDOW& w,
            std::size_t wlen,
            std::size_t hoplen,
            const IFFT_FUNC& ifft_obj
        )
        {
            using T = typename EXP::type;
            using R = remove_complex_t<T>;

            static_assert(is_complex<T>::value, "matrix type must be complex");
            static_assert(std::is_floating_point<R>::value, "underlying type must be complex floating point type");
            DLIB_ASSERT(stft.nc() > 0 && stft.nr() > 0, "stft must be non-empty");
            DLIB_ASSERT(ifftr_nc_size(stft.nc()) >= (long)wlen, "fftsize >= wlen not satisfied");
            DLIB_ASSERT(wlen >= hoplen, "wlen >= hoplen not satisfied");

            const size_t ntime = (stft.nr() - 1) * hoplen + wlen;
            matrix<ReturnType> signal = zeros_matrix<ReturnType>(1, ntime);
            matrix<R> norm = zeros_matrix<R>(1, ntime);
            matrix<R> win(1, wlen);
            for (std::size_t i = 0 ; i < wlen ; ++i)
                win(0, i) = w(i, wlen);
            matrix<R> win2 = squared(win);

            for (long t = 0 ; t < stft.nr() ; ++t)
            {
                set_subm(signal, 0, t*hoplen, 1, wlen) += pointwise_multiply(win, subm(ifft_obj(rowm(stft, t)), 0, 0, 1, wlen));
                set_subm(norm,   0, t*hoplen, 1, wlen) += win2;
            }

            // Remove padding of wlen/2 and wlen/2 on either end
            DLIB_ASSERT(sum(subm(norm, 0, wlen/2, 1, ntime - wlen) < 1e-13) == 0, "NOLA constraint not satisfied");
            signal = pointwise_divide(subm(signal, 0, wlen/2, 1, ntime - wlen),
                                      subm(norm,   0, wlen/2, 1, ntime - wlen));

            return signal;
        }
    }

// ----------------------------------------------------------------------------------------

    inline auto make_hann()
    {
        return [](std::size_t i, std::size_t N) {return hann(i, N, PERIODIC); };
    }

    inline auto make_blackman()
    {
        return [](std::size_t i, std::size_t N) {return blackman(i, N, PERIODIC);};
    }

    inline auto make_blackman_nuttall()
    {
        return [](std::size_t i, std::size_t N) {return blackman_nuttall(i, N, PERIODIC);};
    }

    inline auto make_blackman_harris()
    {
        return [](std::size_t i, std::size_t N) { return blackman_harris(i, N, PERIODIC); };
    }

    inline auto make_blackman_harris7()
    {
        return [](std::size_t i, std::size_t N) { return blackman_harris7(i, N, PERIODIC); };
    }

    inline auto make_kaiser(beta_t beta)
    {
        return [=](std::size_t i, std::size_t N){return kaiser(i, N, beta, PERIODIC);};
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    auto stft (
        const matrix_exp<EXP>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        return details::stft_impl(signal, w, fftsize, wlen, hoplen, details::fft_func{});
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename Alloc, typename WINDOW>
    auto stft (
        const std::vector<T, Alloc>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        return stft(dlib::mat(signal), w, fftsize, wlen, hoplen);
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP,typename WINDOW>
    auto istft (
        const matrix_exp<EXP>& stft,
        const WINDOW& w,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        using T = typename EXP::type;
        return details::istft_impl<T>(stft, w, wlen, hoplen, details::ifft_func{});
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    auto stftr (
        const matrix_exp<EXP>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        return details::stft_impl(signal, w, fftsize, wlen, hoplen, details::fftr_func{});
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename Alloc, typename WINDOW>
    auto stftr (
        const std::vector<T, Alloc>& signal,
        const WINDOW& w,
        std::size_t fftsize,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        return stftr(dlib::mat(signal), w, fftsize, wlen, hoplen);
    }

// ----------------------------------------------------------------------------------------

    template <typename EXP, typename WINDOW>
    auto istftr (
        const matrix_exp<EXP>& stft,
        const WINDOW& w,
        std::size_t wlen,
        std::size_t hoplen
    )
    {
        using R = remove_complex_t<typename EXP::type>;
        return details::istft_impl<R>(stft, w, wlen, hoplen, details::ifftr_func{});
    }

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_FFt_Hh_

