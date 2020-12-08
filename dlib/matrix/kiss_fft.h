/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef DLIB_KISS_FFT_H
#define DLIB_KISS_FFT_H

#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <numeric>
#include "../hash.h"

#define C_FIXDIV(x,y) /*noop*/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dlib
{
    namespace kiss_details
    {
        struct plan_key
        {
            std::vector<int> dims;
            bool is_inverse;

            plan_key(const std::vector<int>& dims, bool is_inverse)
            {
                this->dims = dims;
                this->is_inverse = is_inverse;
            }
            
            bool operator==(const plan_key& other) const
            {
                return std::tie(dims, is_inverse) == std::tie(other.dims, other.is_inverse);
            }

            uint32_t hash() const
            {
                uint32_t ret = 0;
                ret = dlib::hash(dims, ret);
                ret = dlib::hash((uint32_t)is_inverse, ret);
                return ret;
            }
        };

        template<typename T>
        struct kiss_fft_state
        {
            int nfft;
            bool inverse;
            std::vector<int> factors;
            std::vector<std::complex<T>> twiddles;
            
            kiss_fft_state() = default;
            kiss_fft_state(const plan_key& key);
        };

        template<typename T>
        struct kiss_fftnd_state
        {
            std::vector<int> dims;
            std::vector<kiss_fft_state<T>> plans;
            
            int dimprod() const {return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());}
            kiss_fftnd_state() = default;
            kiss_fftnd_state(const plan_key& key);
        };

        template<typename T>
        struct kiss_fftr_state
        {
            kiss_fft_state<T> substate;
            std::vector<std::complex<T>> super_twiddles;
            
            kiss_fftr_state() = default;
            kiss_fftr_state(const plan_key& key);
        };

        template<typename T>
        struct kiss_fftndr_state
        {
            kiss_fftr_state<T> cfg_r;
            kiss_fftnd_state<T> cfg_nd;
            
            kiss_fftndr_state() = default;
            kiss_fftndr_state(const plan_key& key);
        };

        template<typename T>
        inline void kf_bfly2(
            std::complex<T> * Fout,
            const size_t fstride,
            const kiss_fft_state<T>& cfg,
            int m
        )
        {
            const std::complex<T> * tw1 = &cfg.twiddles[0];
            std::complex<T> t;
            std::complex<T> * Fout2 = Fout + m;

            for (int i = 0 ; i < m ; i++)
            {
                t = Fout2[i] * tw1[i*fstride];
                Fout2[i] = Fout[i] - t;
                Fout[i] += t;
            }
        }

        template<typename T>
        inline std::complex<T> rot_PI_2(std::complex<T> z)
        {
            return std::complex<T>(z.imag(), -z.real());
        }

        template<typename T>
        inline void kf_bfly3 (
            std::complex<T> * Fout,
            const size_t fstride,
            const kiss_fft_state<T>& cfg,
            size_t m
        )
        {
            const size_t m2 = 2*m;
            const std::complex<T> *tw1,*tw2;
            std::complex<T> scratch[5];
            const std::complex<T> epi3 = cfg.twiddles[fstride*m];

            tw1=tw2=&cfg.twiddles[0];

            for (size_t k = 0 ; k < m ; k++)
            {
                C_FIXDIV(Fout[k],3); C_FIXDIV(Fout[k+m],3); C_FIXDIV(Fout[m2+k],3); //noop for float and double

                scratch[1] = Fout[k+m]  * tw1[k*fstride];
                scratch[2] = Fout[k+m2] * tw2[k*fstride*2];

                scratch[3] = scratch[1] + scratch[2];
                scratch[0] = scratch[1] - scratch[2];

                Fout[m+k] = Fout[k] - T(0.5) * scratch[3];

                scratch[0] *= epi3.imag();

                Fout[k] += scratch[3];

                Fout[k+m2] = Fout[k+m] + rot_PI_2(scratch[0]);
                Fout[k+m] -= rot_PI_2(scratch[0]);
            }
        }

        template<typename T>
        inline void kf_bfly4(
            std::complex<T> * Fout,
            const size_t fstride,
            const kiss_fft_state<T>& cfg,
            const size_t m
        )
        {
            const std::complex<T> *tw1,*tw2,*tw3;
            std::complex<T> scratch[6];
            const size_t m2=2*m;
            const size_t m3=3*m;

            tw3 = tw2 = tw1 = &cfg.twiddles[0];

            for (size_t k = 0 ; k < m ; k++)
            {
                C_FIXDIV(Fout[k],4); C_FIXDIV(Fout[m],4); C_FIXDIV(Fout[m2+k],4); C_FIXDIV(Fout[m3+k],4);

                scratch[0] = Fout[m+k]  * tw1[k*fstride];
                scratch[1] = Fout[m2+k] * tw2[k*fstride*2];
                scratch[2] = Fout[m3+k] * tw3[k*fstride*3];

                scratch[5] = Fout[k] - scratch[1];
                Fout[k]  += scratch[1];
                scratch[3] = scratch[0] + scratch[2];
                scratch[4] = scratch[0] - scratch[2];
                Fout[m2+k] = Fout[k] - scratch[3];

                Fout[k] += scratch[3];

                if(cfg.inverse) {
                    Fout[m+k]  = scratch[5] - rot_PI_2(scratch[4]);
                    Fout[m3+k] = scratch[5] + rot_PI_2(scratch[4]);
                }else {
                    Fout[m+k] =  scratch[5] + rot_PI_2(scratch[4]);
                    Fout[m3+k] = scratch[5] - rot_PI_2(scratch[4]);
                }
            }
        }

        template<typename T>
        inline void kf_bfly5(
            std::complex<T> * Fout,
            const size_t fstride,
            const kiss_fft_state<T>& cfg,
            int m
        )
        {
            std::complex<T> *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
            int u;
            std::complex<T> scratch[13];
            const std::complex<T> * twiddles = &cfg.twiddles[0];
            const std::complex<T> *tw;
            std::complex<T> ya,yb;
            ya = twiddles[fstride*m];
            yb = twiddles[fstride*2*m];

            Fout0=Fout;
            Fout1=Fout0+m;
            Fout2=Fout0+2*m;
            Fout3=Fout0+3*m;
            Fout4=Fout0+4*m;

            tw=&cfg.twiddles[0];
            for ( u=0; u<m; ++u ) {
                C_FIXDIV( *Fout0,5); C_FIXDIV( *Fout1,5); C_FIXDIV( *Fout2,5); C_FIXDIV( *Fout3,5); C_FIXDIV( *Fout4,5);
                scratch[0] = *Fout0;

                scratch[1] = *Fout1 * tw[u*fstride]; //C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
                scratch[2] = *Fout2 * tw[2*u*fstride]; //C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
                scratch[3] = *Fout3 * tw[3*u*fstride]; //C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
                scratch[4] = *Fout4 * tw[4*u*fstride]; //C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

                scratch[7]  = scratch[1] + scratch[4]; //C_ADD( scratch[7],scratch[1],scratch[4]);
                scratch[10] = scratch[1] - scratch[4]; //C_SUB( scratch[10],scratch[1],scratch[4]);
                scratch[8]  = scratch[2] + scratch[3]; //C_ADD( scratch[8],scratch[2],scratch[3]);
                scratch[9]  = scratch[2] - scratch[3]; //C_SUB( scratch[9],scratch[2],scratch[3]);

                *Fout0 += scratch[7] + scratch[8];

                scratch[5].real(scratch[0].real() + scratch[7].real() * ya.real() + scratch[8].real() * yb.real());
                scratch[5].imag(scratch[0].imag() + scratch[7].imag() * ya.real() + scratch[8].imag() * yb.real());

                scratch[6].real(scratch[10].imag() * ya.imag() + scratch[9].imag() * yb.imag());
                scratch[6].imag(-scratch[10].real() * ya.imag() - scratch[9].real() * yb.imag());

                *Fout1 = scratch[5] - scratch[6]; //C_SUB(*Fout1,scratch[5],scratch[6]);
                *Fout4 = scratch[5] + scratch[6]; //C_ADD(*Fout4,scratch[5],scratch[6]);

                scratch[11].real(scratch[0].real() + scratch[7].real()*yb.real() + scratch[8].real()*ya.real());
                scratch[11].imag(scratch[0].imag() + scratch[7].imag()*yb.real() + scratch[8].imag()*ya.real());
                scratch[12].real(- scratch[10].imag()*yb.imag() + scratch[9].imag()*ya.imag());
                scratch[12].imag(scratch[10].real()*yb.imag() - scratch[9].real()*ya.imag());

                *Fout2 = scratch[11] + scratch[12];
                *Fout3 = scratch[11] - scratch[12];

                ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
            }
        }

        /* perform the butterfly for one stage of a mixed radix FFT */
        template<typename T> 
        inline void kf_bfly_generic(
            std::complex<T> * Fout,
            const size_t fstride,
            const kiss_fft_state<T>& cfg,
            int m,
            int p
        )
        {
            int u,k,q1,q;
            const std::complex<T> * twiddles = &cfg.twiddles[0];
            std::complex<T> t;
            int Norig = cfg.nfft;

            std::vector<std::complex<T>> scratch(p);

            for ( u=0; u<m; ++u ) {
                k=u;
                for ( q1=0 ; q1<p ; ++q1 ) {
                    scratch[q1] = Fout[ k  ];
                    C_FIXDIV(scratch[q1],p);
                    k += m;
                }

                k=u;
                for ( q1=0 ; q1<p ; ++q1 ) {
                    int twidx=0;
                    Fout[ k ] = scratch[0];
                    for (q=1;q<p;++q ) {
                        twidx += fstride * k;
                        if (twidx>=Norig) twidx-=Norig;
                        t = scratch[q] * twiddles[twidx];
                        Fout[ k ] += t;
                    }
                    k += m;
                }
            }
        }

        template<typename T>
        inline void kf_work(
            const kiss_fft_state<T>& cfg,
            const int* factors,
            std::complex<T>* Fout,
            const std::complex<T>* f,
            const size_t fstride,
            int in_stride
        )
        {
            std::complex<T> * Fout_beg = Fout;
            const int p=*factors++; /* the radix  */
            const int m=*factors++; /* stage's fft length/p */
            const std::complex<T> * Fout_end = Fout + p*m;

            if (m==1) {
                do{
                    *Fout = *f;
                    f += fstride*in_stride;
                }while(++Fout != Fout_end );
            }else{
                do{
                    // recursive call:
                    // DFT of size m*p performed by doing
                    // p instances of smaller DFTs of size m,
                    // each one takes a decimated version of the input
                    kf_work(cfg, factors, Fout , f, fstride*p, in_stride);
                    f += fstride*in_stride;
                }while( (Fout += m) != Fout_end );
            }

            Fout=Fout_beg;

            // recombine the p smaller DFTs
            switch (p) {
                case 2: kf_bfly2(Fout,fstride,cfg,m); break;
                case 3: kf_bfly3(Fout,fstride,cfg,m); break;
                case 4: kf_bfly4(Fout,fstride,cfg,m); break;
                case 5: kf_bfly5(Fout,fstride,cfg,m); break;
                default: kf_bfly_generic(Fout,fstride,cfg,m,p); break;
            }
        }

        /*  facbuf is populated by p1,m1,p2,m2, ...
            where
            p[i] * m[i] = m[i-1]
            m0 = n                  */
        inline void kf_factor(int n, std::vector<int>& facbuf)
        {
            int p=4;
            double floor_sqrt;
            floor_sqrt = std::floor( std::sqrt((double)n) );

            /*factor out powers of 4, powers of 2, then any remaining primes */
            do {
                while (n % p) {
                    switch (p) {
                        case 4: p = 2; break;
                        case 2: p = 3; break;
                        default: p += 2; break;
                    }
                    if (p > floor_sqrt)
                        p = n;          /* no more factors, skip to end */
                }
                n /= p;
                facbuf.push_back(p);
                facbuf.push_back(n);
            } while (n > 1);
        }

        template<typename T>
        inline kiss_fft_state<T>::kiss_fft_state(const plan_key& key)
        {
            nfft       = key.dims[0];
            inverse    = key.is_inverse;
            twiddles.resize(nfft);

            for (int i = 0 ; i < nfft ; ++i) 
            {
                double phase = -2*M_PI*i / nfft;
                if (inverse)
                    phase *= -1;
                twiddles[i] = std::polar(1.0, phase);
            }

            kf_factor(nfft,factors);
        }

        template<typename T>
        void kiss_fft_stride(const kiss_fft_state<T>& cfg, const std::complex<T>* fin, std::complex<T>* fout,int fin_stride)
        {
            if (fin == fout) 
            {
                if (fout == nullptr)
                    throw std::runtime_error("fout buffer NULL.");

                std::vector<std::complex<T>> tmpbuf(cfg.nfft);
                kiss_fft_stride(cfg, fin, &tmpbuf[0], fin_stride);
                std::copy(tmpbuf.begin(), tmpbuf.end(), fout);
            }
            else
            {
                kf_work(cfg, &cfg.factors[0], fout, fin, 1, fin_stride);
            }
        }

        template<typename T>
        inline kiss_fftnd_state<T>::kiss_fftnd_state(const plan_key& key)
        {
            dims = key.dims;
            for (size_t i = 0 ; i < dims.size() ; i++)
                plans.push_back(std::move(kiss_fft_state<T>(plan_key({dims[i]}, key.is_inverse))));
        }

        template<typename T>
        void kiss_fftnd(const kiss_fftnd_state<T>& cfg, const std::complex<T>* fin, std::complex<T>* fout)
        {
            const std::complex<T>* bufin=fin;
            std::complex<T>* bufout;
            std::vector<std::complex<T>> tmpbuf(cfg.dimprod());

            /*arrange it so the last bufout == fout*/
            if ( cfg.dims.size() & 1 )
            {
                bufout = fout;
                if (fin==fout) {
                    std::copy(fin, fin + cfg.dimprod(), tmpbuf.begin());
                    bufin = &tmpbuf[0];
                }
            }
            else
                bufout = &tmpbuf[0];

            for (size_t k=0; k < cfg.dims.size(); ++k) 
            {
                int curdim = cfg.dims[k];
                int stride = cfg.dimprod() / curdim;

                for (int i=0 ; i<stride ; ++i ) 
                    kiss_fft_stride(cfg.plans[k], bufin+i , bufout+i*curdim, stride );

                /*toggle back and forth between the two buffers*/
                if (bufout == &tmpbuf[0])
                {
                    bufout = fout;
                    bufin = &tmpbuf[0];
                }
                else
                {
                    bufout = &tmpbuf[0];
                    bufin = fout;
                }
            }
        }

        template<typename T>
        inline kiss_fftr_state<T>::kiss_fftr_state(const plan_key& key)
        {
            const int nfft = key.dims[0] / 2;
            substate = kiss_fft_state<T>(plan_key({nfft}, key.is_inverse));
            super_twiddles.resize(nfft/2);

            for (size_t i = 0 ; i < super_twiddles.size() ; ++i) 
            {
                double phase = -M_PI * ((i+1) * 1.0 / nfft + 0.5);
                if (key.is_inverse)
                    phase *= -1;
                super_twiddles[i] = std::polar(1.0, phase);
            }
        }

        template<typename T>
        void kiss_fftr(const kiss_fftr_state<T>& plan, const T* timedata, std::complex<T>* freqdata)
        {
            if (plan.substate.inverse)
                throw std::runtime_error("bad fftr plan : need a forward plan. This is an inverse plan");

            const int nfft_h = plan.substate.nfft; //recall that the FFT size is actually half the original requested FFT size, i.e. the size of timedata

            /*perform the parallel fft of two real signals packed in real,imag*/
            std::vector<std::complex<T>> tmpbuf(nfft_h);
            kiss_fft_stride(plan.substate, (const std::complex<T>*)timedata, &tmpbuf[0], 1);
            /* The real part of the DC element of the frequency spectrum in st->tmpbuf
             * contains the sum of the even-numbered elements of the input time sequence
             * The imag part is the sum of the odd-numbered elements
             *
             * The sum of tdc.r and tdc.i is the sum of the input time sequence.
             *      yielding DC of input time sequence
             * The difference of tdc.r - tdc.i is the sum of the input (dot product) [1,-1,1,-1...
             *      yielding Nyquist bin of input time sequence
             */

            freqdata[0]         = std::complex<T>(tmpbuf[0].real() + tmpbuf[0].imag(), 0);
            freqdata[nfft_h]    = std::complex<T>(tmpbuf[0].real() - tmpbuf[0].imag(), 0);

            for (int k = 1 ; k <= nfft_h / 2 ; k++)
            {
                auto fpk  = tmpbuf[0];
                auto fpnk = std::complex<T>(tmpbuf[nfft_h-k].real(), -tmpbuf[nfft_h-k].imag());
                auto f1k = fpk + fpnk;
                auto f2k = fpk - fpnk;
                auto tw  = f2k * plan.super_twiddles[k-1];
                freqdata[k]         = 0.5 * (f1k + tw);
                freqdata[nfft_h-k]  = 0.5 * std::complex<T>(f1k.real() - tw.real(), tw.imag() - f1k.imag());
            }
        }

        template<typename T>
        void kiss_fftri(const kiss_fftr_state<T>& plan, const std::complex<T>* freqdata, T* timedata)
        {
            if (!plan.substate.inverse)
                throw std::runtime_error("bad fftr plan : need an inverse plan. This is a forward plan");

            const int nfft_h = plan.substate.nfft; //recall that the FFT size is actually half the original requested FFT size, i.e. the size of timedata

            std::vector<std::complex<T>> tmpbuf(nfft_h);

            tmpbuf[0] = std::complex<T>(freqdata[0].real() + freqdata[nfft_h].real(),
                                        freqdata[0].real() - freqdata[nfft_h].real());

            for (int k = 1; k <= nfft_h / 2; ++k)
            {
                std::complex<T> fk      = freqdata[k];
                std::complex<T> fnkc    = std::complex<T>(freqdata[nfft_h - k].real(), 
                                                          -freqdata[nfft_h - k].imag());
                auto fek = fk + fnkc;
                auto tmp = fk - fnkc;
                auto fok = tmp * plan.super_twiddles[k-1];
                tmpbuf[k] = fek + fok;
                tmpbuf[nfft_h - k] = fek - fok;
                tmpbuf[nfft_h - k].imag(-tmpbuf[nfft_h - k].imag());
            }

            kiss_fft_stride (plan.substate, &tmpbuf[0], (std::complex<T>*)timedata);
        }

        template<typename T>
        inline kiss_fftndr_state<T>::kiss_fftndr_state(const plan_key& key)
        {
            std::vector<int> frontdims = key.dims;
            frontdims.pop_back();
            cfg_r  = kiss_fftr_state<T>(key.dims.back(), key.is_inverse);
            cfg_nd = kiss_fftnd_state<T>(frontdims, key.is_inverse);
        }

        template<typename T>
        void kiss_fftndr(const kiss_fftndr_state<T>& plan, const T* timedata, std::complex<T>* freqdata)
        {
            int dimReal  = plan.cfg_r.substate.nfft*2; //recall the real fft size is half the length of the input
            int dimOther = plan.cfg_nd.dimprod();
            int nrbins   = dimReal/2+1;

            std::vector<std::complex<T>> tmp1(std::max<int>(nrbins, dimOther));
            std::vector<std::complex<T>> tmp2(plan.cfg_nd.dimprod()*dimReal);

            // take a real chunk of data, fft it and place the output at correct intervals
            for (int k1 = 0; k1 < dimOther; ++k1) 
            {
                kiss_fftr(plan.cfg_r, timedata + k1*dimReal , &tmp1[0]); // tmp1 now holds nrbins complex points
                for (int k2 = 0; k2 < nrbins; ++k2)
                   tmp2[k2*dimOther+k1] = tmp1[k2];
            }

            for (int k2 = 0; k2 < nrbins; ++k2) 
            {
                kiss_fftnd(plan.cfg_nd, &tmp2[k2*dimOther], &tmp1[0]);  // tmp1 now holds dimOther complex points
                for (int k1 = 0; k1 < dimOther; ++k1) 
                    freqdata[ k1*(nrbins) + k2] = tmp1[k1];
            }
        }

        template<typename T>
        void kiss_fftndri(const kiss_fftndr_state<T>& plan, const std::complex<T>* freqdata, T* timedata)
        {
            int dimReal  = plan.cfg_r.substate.nfft*2; //recall the real fft size is half the length of the input
            int dimOther = plan.cfg_nd.dimprod();
            int nrbins   = dimReal/2+1;

            std::vector<std::complex<T>> tmp1(std::max<int>(nrbins, dimOther));
            std::vector<std::complex<T>> tmp2(plan.cfg_nd.dimprod()*dimReal);

            for (int k2 = 0; k2 < nrbins; ++k2) 
            {
                for (int k1 = 0; k1 < dimOther; ++k1) 
                    tmp1[k1] = freqdata[ k1*(nrbins) + k2 ];
                kiss_fftnd(plan.cfg_nd, &tmp1[0], &tmp2[k2*dimOther]);
            }

            for (int k1 = 0; k1 < dimOther; ++k1) 
            {
                for (int k2 = 0; k2 < nrbins; ++k2)
                    tmp1[k2] = tmp2[ k2*dimOther+k1 ];
                kiss_fftri(plan.cfg_r, &tmp1[0], timedata + k1*dimReal);
            }
        }

        struct hasher
        {
            size_t operator()(const plan_key& key) const {return key.hash();}
        };

        template<typename plan_type>
        class cache
        {
        public:
            static cache& get()
            {
                static cache singleton;
                return singleton;
            }

            void get_plan(const plan_key& key, plan_type& plan)
            {
                std::lock_guard<std::mutex> l(m);
                auto it = plans.find(key);
                if (it != plans.end())
                {
                    plan = it->second;
                }
                else
                {
                    plans[key] = plan_type(key);
                    plan = plans[key];
                }
            }

        private:
            cache() = default;
            cache(const cache& orig) = delete;
            cache& operator=(const cache& orig) = delete;

            std::unordered_map<plan_key, plan_type, hasher> plans;
            std::mutex m;
        };
    }

    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    void kiss_fft(const std::vector<int>& dims, const std::complex<T>* fin, std::complex<T>* fout, bool is_inverse)
    {
        using namespace kiss_details;

        if (dims.size() == 1)
        {
            kiss_fft_state<T> plan;
            cache<kiss_fft_state<T>>::get().get_plan({dims, is_inverse}, plan);
            kiss_fft_stride(plan, fin, fout, 1);
        }
        else
        {
            kiss_fftnd_state<T> plan;
            cache<kiss_fftnd_state<T>>::get().get_plan({dims,is_inverse}, plan);
            kiss_fftnd(plan, fin, fout);
        }
    }

    /*
     *  fin  has dims[0] * dims[1] * ... * dims[-2] * dims[-1] points
     *  fout has dims[0] * dims[1] * ... * dims[-2] * (dims[-1]/2+1) points
     */
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    void kiss_fftr(const std::vector<int>& dims, const T* fin, std::complex<T>* fout)
    {
        using namespace kiss_details;

        if (dims.size() == 1)
        {
            kiss_fftr_state<T> plan;
            cache<kiss_fftr_state<T>>::get().get_plan({dims,false}, plan);
            kiss_fftr(plan, fin, fout);
        }
        else
        {
            kiss_fftndr_state<T> plan;
            cache<kiss_fftndr_state<T>>::get().get_plan({dims,false}, plan);
            kiss_fftndr(plan, fin, fout);
        }
    }

    /*
     *  fin  has dims[0] * dims[1] * ... * dims[-2] * (dims[-1]/2+1) points
     *  fout has dims[0] * dims[1] * ... * dims[-2] * dims[-1] points
     */
    template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
    void kiss_fftri(const std::vector<int>& dims, const std::complex<T>* fin, T* fout)
    {
        using namespace kiss_details;

        if (dims.size() == 1)
        {
            kiss_fftr_state<T> plan;
            cache<kiss_fftr_state<T>>::get().get_plan({dims,true}, plan);
            kiss_fftri(plan, fin, fout);
        }
        else
        {
            kiss_fftndr_state<T> plan;
            cache<kiss_fftndr_state<T>>::get().get_plan({dims,true}, plan);
            kiss_fftndri(plan, fin, fout);
        }
    }

    inline int kiss_fft_next_fast_size(int n)
    {
        while(1) {
            int m=n;
            while ( (m%2) == 0 ) m/=2;
            while ( (m%3) == 0 ) m/=3;
            while ( (m%5) == 0 ) m/=5;
            if (m<=1)
                break; /* n is completely factorable by twos, threes, and fives */
            n++;
        }
        return n;
    }

    inline int kiss_fftr_next_fast_size_real(int n)
    {
        return kiss_fft_next_fast_size((n+1)>>1) << 1;
    }
}

#endif // DLIB_KISS_FFT_H
