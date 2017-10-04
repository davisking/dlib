// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_sIMD4I_Hh_
#define DLIB_sIMD4I_Hh_

#include "simd_check.h"
#include "../uintn.h"

namespace dlib
{

#ifdef DLIB_HAVE_SSE2
    class simd4i
    {
    public:
        typedef int32 type;

        inline simd4i() {}
        inline simd4i(int32 f) { x = _mm_set1_epi32(f); }
        inline simd4i(int32 r0, int32 r1, int32 r2, int32 r3) { x = _mm_setr_epi32(r0,r1,r2,r3); }
        inline simd4i(const __m128i& val):x(val) {}

        inline simd4i& operator=(const __m128i& val)
        {
            x = val;
            return *this;
        }

        inline operator __m128i() const { return x; }

        inline void load_aligned(const type* ptr)  { x = _mm_load_si128((const __m128i*)ptr); }
        inline void store_aligned(type* ptr) const { _mm_store_si128((__m128i*)ptr, x); }
        inline void load(const type* ptr)          { x = _mm_loadu_si128((const __m128i*)ptr); }
        inline void store(type* ptr)         const { _mm_storeu_si128((__m128i*)ptr, x); }

        inline unsigned int size() const { return 4; }
        inline int32 operator[](unsigned int idx) const 
        {
            int32 temp[4];
            store(temp);
            return temp[idx];
        }

    private:
        __m128i x;
    };

#elif defined(DLIB_HAVE_VSX)

    class simd4i
    {    
        typedef union {
            vector signed int v;
            vector bool int b;
            signed int x[4];
        } v4i;

        v4i x;

    public:
        inline simd4i() : x{0,0,0,0} { }
        inline simd4i(const simd4i& v) : x(v.x) { }
        inline simd4i(const vector int& v) : x{v} { }
        inline simd4i(const vector bool int& b) { x.b=b; }

        inline simd4i(int32 f) : x{f,f,f,f} { }
        inline simd4i(int32 r0, int32 r1, int32 r2, int32 r3)
             : x{r0,r1,r2,r3} { }		

        inline simd4i& operator=(const simd4i& v) { x = v.x; return *this; }
        inline simd4i& operator=(const int32& v) { *this = simd4i(v); return *this; }

        inline vector signed int operator() () const { return x.v; }
        inline int32 operator[](unsigned int idx) const { return x.x[idx]; }
        
        inline vector bool int to_bool() const { return x.b; }
        
        // intrinsics now seem to use xxpermdi automatically now
        inline void load_aligned(const int32* ptr)  { x.v = vec_ld(0, ptr); }
        inline void store_aligned(int32* ptr) const { vec_st(x.v, 0, ptr); }
        inline void load(const int32* ptr) { x.v = vec_vsx_ld(0, ptr); }
        inline void store(int32* ptr) const { vec_vsx_st(x.v, 0, ptr); }
        

        struct rawarray
        {
            v4i v;
        };
        inline simd4i(const rawarray& a) : x{a.v} { }

    };

#elif defined(DLIB_HAVE_NEON)

    class simd4i
    {
    public:
        typedef int32 type;

        inline simd4i() {}
        inline simd4i(int32 f) { x = vdupq_n_s32(f); }
        inline simd4i(int32 r0, int32 r1, int32 r2, int32 r3)
        {
            int32 __attribute__((aligned(16))) data[4] = { r0, r1, r2, r3 };
            x = vld1q_s32(data);
        }
        inline simd4i(const int32x4_t& val):x(val) {}

        inline simd4i& operator=(const int32x4_t& val)
        {
            x = val;
            return *this;
        }

        inline operator int32x4_t() const { return x; }
        inline operator uint32x4_t() const { return (uint32x4_t)x; }

        inline void load_aligned(const type* ptr)  { x = vld1q_s32(ptr); }
        inline void store_aligned(type* ptr) const { vst1q_s32(ptr, x); }
        inline void load(const type* ptr)          { x = vld1q_s32(ptr); }
        inline void store(type* ptr)         const { vst1q_s32(ptr, x); }

        inline unsigned int size() const { return 4; }
        inline int32 operator[](unsigned int idx) const
        {
            int32 temp[4];
            store(temp);
            return temp[idx];
        }

    private:
        int32x4_t x;
    };

#else

    class simd4i
    {
    public:
        typedef int32 type;

        inline simd4i() {}
        inline simd4i(int32 f) { x[0]=f; x[1]=f; x[2]=f; x[3]=f; }
        inline simd4i(int32 r0, int32 r1, int32 r2, int32 r3) { x[0]=r0; x[1]=r1; x[2]=r2; x[3]=r3;}

        struct rawarray
        {
            int32 a[4];
        };
        inline simd4i(const rawarray& a) { x[0]=a.a[0]; x[1]=a.a[1]; x[2]=a.a[2]; x[3]=a.a[3]; }

        inline void load_aligned(const type* ptr)
        {
            x[0] = ptr[0];
            x[1] = ptr[1];
            x[2] = ptr[2];
            x[3] = ptr[3];
        }

        inline void store_aligned(type* ptr) const
        {
            ptr[0] = x[0];
            ptr[1] = x[1];
            ptr[2] = x[2];
            ptr[3] = x[3];
        }

        inline void load(const type* ptr)
        {
            x[0] = ptr[0];
            x[1] = ptr[1];
            x[2] = ptr[2];
            x[3] = ptr[3];
        }

        inline void store(type* ptr) const
        {
            ptr[0] = x[0];
            ptr[1] = x[1];
            ptr[2] = x[2];
            ptr[3] = x[3];
        }

        inline unsigned int size() const { return 4; }
        inline int32 operator[](unsigned int idx) const { return x[idx]; }

    private:
        int32 x[4];
    };
#endif

// ----------------------------------------------------------------------------------------

    inline std::ostream& operator<<(std::ostream& out, const simd4i& item)
    {
        int32 temp[4];
        item.store(temp);
        out << "(" << temp[0] << ", " << temp[1] << ", " << temp[2] << ", " << temp[3] << ")";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator+ (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_add_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_add(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return vaddq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]+rhs[0],
                      lhs[1]+rhs[1],
                      lhs[2]+rhs[2],
                      lhs[3]+rhs[3]);
#endif
    }
    inline simd4i& operator+= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs + rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator- (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_sub_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_sub(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return vsubq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]-rhs[0],
                      lhs[1]-rhs[1],
                      lhs[2]-rhs[2],
                      lhs[3]-rhs[3]);
#endif
    }
    inline simd4i& operator-= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs - rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator* (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE41
        return _mm_mullo_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_SSE2)
        int32 _lhs[4]; lhs.store(_lhs);
        int32 _rhs[4]; rhs.store(_rhs);
        return simd4i(_lhs[0]*_rhs[0],
                      _lhs[1]*_rhs[1],
                      _lhs[2]*_rhs[2],
                      _lhs[3]*_rhs[3]);
#elif defined(DLIB_HAVE_VSX)
        vector int a = lhs(), b = rhs();
        asm("vmuluwm %0, %0, %1\n\t" : "+&v" (a) : "v" (b) );
        return simd4i(a);
#elif defined(DLIB_HAVE_NEON)
        return vmulq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]*rhs[0],
                      lhs[1]*rhs[1],
                      lhs[2]*rhs[2],
                      lhs[3]*rhs[3]);
#endif
    }
    inline simd4i& operator*= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs * rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator& (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_and_si128(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_and(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return vandq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]&rhs[0],
                      lhs[1]&rhs[1],
                      lhs[2]&rhs[2],
                      lhs[3]&rhs[3]);
#endif
    }
    inline simd4i& operator&= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs & rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator| (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_or_si128(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_or(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return vorrq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]|rhs[0],
                      lhs[1]|rhs[1],
                      lhs[2]|rhs[2],
                      lhs[3]|rhs[3]);
#endif
    }
    inline simd4i& operator|= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs | rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator^ (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_xor_si128(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_xor(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return veorq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]^rhs[0],
                      lhs[1]^rhs[1],
                      lhs[2]^rhs[2],
                      lhs[3]^rhs[3]);
#endif
    }
    inline simd4i& operator^= (simd4i& lhs, const simd4i& rhs) 
    { return lhs = lhs ^ rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator~ (const simd4i& lhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_xor_si128(lhs, _mm_set1_epi32(0xFFFFFFFF)); 
#elif defined(DLIB_HAVE_VSX)
        return vec_xor(lhs(), vec_splats(~0));
#elif defined(DLIB_HAVE_NEON)
        return vmvnq_s32(lhs);
#else
        return simd4i(~lhs[0],
                      ~lhs[1],
                      ~lhs[2],
                      ~lhs[3]);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator<< (const simd4i& lhs, const int& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_sll_epi32(lhs,_mm_cvtsi32_si128(rhs));
#elif defined(DLIB_HAVE_VSX)
        return vec_sl(lhs(), vec_splats((uint32_t)rhs));         
#elif defined(DLIB_HAVE_NEON)
        return vshlq_s32(lhs, simd4i(rhs));
#else
        return simd4i(lhs[0]<<rhs,
                      lhs[1]<<rhs,
                      lhs[2]<<rhs,
                      lhs[3]<<rhs);
#endif
    }
    inline simd4i& operator<<= (simd4i& lhs, const int& rhs) 
    { return lhs = lhs << rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator>> (const simd4i& lhs, const int& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_sra_epi32(lhs,_mm_cvtsi32_si128(rhs));
#elif defined(DLIB_HAVE_VSX)
        return vec_sr(lhs(), vec_splats((uint32_t)rhs)); 
#elif defined(DLIB_HAVE_NEON)
        int32 _lhs[4]; lhs.store(_lhs);
        return simd4i(_lhs[0]>>rhs,
                      _lhs[1]>>rhs,
                      _lhs[2]>>rhs,
                      _lhs[3]>>rhs);
#else
        return simd4i(lhs[0]>>rhs,
                      lhs[1]>>rhs,
                      lhs[2]>>rhs,
                      lhs[3]>>rhs);
#endif
    }
    inline simd4i& operator>>= (simd4i& lhs, const int& rhs) 
    { return lhs = lhs >> rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd4i operator== (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmpeq_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_cmpeq(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return (int32x4_t)vceqq_s32(lhs,rhs);
#else
        return simd4i(lhs[0]==rhs[0] ? 0xFFFFFFFF : 0,
                      lhs[1]==rhs[1] ? 0xFFFFFFFF : 0,
                      lhs[2]==rhs[2] ? 0xFFFFFFFF : 0,
                      lhs[3]==rhs[3] ? 0xFFFFFFFF : 0);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator!= (const simd4i& lhs, const simd4i& rhs) 
    { 
#if defined(DLIB_HAVE_SSE2) || defined(DLIB_HAVE_VSX) || defined(DLIB_HAVE_NEON)
        return ~(lhs==rhs);
#else
        return simd4i(lhs[0]!=rhs[0] ? 0xFFFFFFFF : 0,
                      lhs[1]!=rhs[1] ? 0xFFFFFFFF : 0,
                      lhs[2]!=rhs[2] ? 0xFFFFFFFF : 0,
                      lhs[3]!=rhs[3] ? 0xFFFFFFFF : 0);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator< (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return _mm_cmplt_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_VSX)
        return vec_cmplt(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return (int32x4_t)vcltq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]<rhs[0] ? 0xFFFFFFFF : 0,
                      lhs[1]<rhs[1] ? 0xFFFFFFFF : 0,
                      lhs[2]<rhs[2] ? 0xFFFFFFFF : 0,
                      lhs[3]<rhs[3] ? 0xFFFFFFFF : 0);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator> (const simd4i& lhs, const simd4i& rhs) 
    { 
        return rhs < lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator<= (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE2
        return ~(lhs > rhs); 
#elif defined(DLIB_HAVE_NEON)
        return (int32x4_t)vcleq_s32(lhs, rhs);
#else
        return simd4i(lhs[0]<=rhs[0] ? 0xFFFFFFFF : 0,
                      lhs[1]<=rhs[1] ? 0xFFFFFFFF : 0,
                      lhs[2]<=rhs[2] ? 0xFFFFFFFF : 0,
                      lhs[3]<=rhs[3] ? 0xFFFFFFFF : 0);
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i operator>= (const simd4i& lhs, const simd4i& rhs) 
    { 
        return rhs <= lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd4i min (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE41
        return _mm_min_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_SSE2)
        int32 _lhs[4]; lhs.store(_lhs);
        int32 _rhs[4]; rhs.store(_rhs);
        return simd4i(std::min(_lhs[0],_rhs[0]),
                      std::min(_lhs[1],_rhs[1]),
                      std::min(_lhs[2],_rhs[2]),
                      std::min(_lhs[3],_rhs[3]));
#elif defined(DLIB_HAVE_VSX)
        return vec_min(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return (int32x4_t)vminq_s32(lhs, rhs);
#else
        return simd4i(std::min(lhs[0],rhs[0]),
                      std::min(lhs[1],rhs[1]),
                      std::min(lhs[2],rhs[2]),
                      std::min(lhs[3],rhs[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline simd4i max (const simd4i& lhs, const simd4i& rhs) 
    { 
#ifdef DLIB_HAVE_SSE41
        return _mm_max_epi32(lhs, rhs); 
#elif defined(DLIB_HAVE_SSE2)
        int32 _lhs[4]; lhs.store(_lhs);
        int32 _rhs[4]; rhs.store(_rhs);
        return simd4i(std::max(_lhs[0],_rhs[0]),
                      std::max(_lhs[1],_rhs[1]),
                      std::max(_lhs[2],_rhs[2]),
                      std::max(_lhs[3],_rhs[3]));
#elif defined(DLIB_HAVE_VSX)
        return vec_max(lhs(), rhs());
#elif defined(DLIB_HAVE_NEON)
        return vmaxq_s32(lhs, rhs);
#else
        return simd4i(std::max(lhs[0],rhs[0]),
                      std::max(lhs[1],rhs[1]),
                      std::max(lhs[2],rhs[2]),
                      std::max(lhs[3],rhs[3]));
#endif
    }

// ----------------------------------------------------------------------------------------

    inline int32 sum(const simd4i& item)
    {
#ifdef DLIB_HAVE_SSE3
        simd4i temp = _mm_hadd_epi32(item,item);
        temp = _mm_hadd_epi32(temp,temp);
        return _mm_cvtsi128_si32(temp);
#elif defined(DLIB_HAVE_SSE2)
        int32 temp[4];
        item.store(temp);
        return temp[0]+temp[1]+temp[2]+temp[3];
#elif defined(DLIB_HAVE_NEON)
        int32x2_t r = vadd_s32(vget_high_s32(item), vget_low_s32(item));
        return vget_lane_s32(vpadd_s32(r, r), 0);
#else
        return item[0]+item[1]+item[2]+item[3];
#endif
    }

// ----------------------------------------------------------------------------------------

    // perform cmp ? a : b
    inline simd4i select(const simd4i& cmp, const simd4i& a, const simd4i& b)
    {
#ifdef DLIB_HAVE_SSE41
        return _mm_blendv_epi8(b,a,cmp);
#elif defined(DLIB_HAVE_SSE2)
        return ((cmp&a) | _mm_andnot_si128(cmp,b));
#elif defined(DLIB_HAVE_VSX)
        return vec_sel(b(), a(), cmp.to_bool());
#elif defined(DLIB_HAVE_NEON)
        return vbslq_s32(cmp, a, b);
#else
        return ((cmp&a) | (~cmp&b));
#endif
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_sIMD4I_Hh_

