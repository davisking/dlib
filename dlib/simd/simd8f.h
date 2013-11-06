// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_sIMD8F_H__
#define DLIB_sIMD8F_H__

#include "simd_check.h"
#include "simd4f.h"


namespace dlib
{
#ifdef DLIB_HAVE_AVX
    class simd8f
    {
    public:
        typedef float type;

        simd8f() {}
        simd8f(const simd4f& low, const simd4f& high)
        {
            x = _mm256_insertf128_ps(_mm256_castps128_ps256(low),high,1);
        }
        simd8f(float f) { x = _mm256_set1_ps(f); }
        inline simd8f(float r0, float r1, float r2, float r3, float r4, float r5, float r6, float r7) 
        { x = _mm256_setr_ps(r0,r1,r2,r3,r4,r5,r6,r7); }

        simd8f(const __m256& val):x(val) {}
        simd8f& operator=(const __m256& val)
        {
            x = val;
            return *this;
        }
        inline operator __m256() const { return x; }

        void load_aligned(const type* ptr)  { x = _mm256_load_ps(ptr); }
        void store_aligned(type* ptr) const { _mm256_store_ps(ptr, x); }
        void load(const type* ptr)          { x = _mm256_loadu_ps(ptr); }
        void store(type* ptr)         const { _mm256_storeu_ps(ptr, x); }

        unsigned int size() const { return 8; }
        float operator[](unsigned int idx) const 
        {
            float temp[8];
            store(temp);
            return temp[idx];
        }

        simd4f low() const { return _mm256_castps256_ps128(x); }
        simd4f high() const { return _mm256_extractf128_ps(x,1); }

    private:
        __m256 x;
    };
#else
    class simd8f
    {
    public:
        typedef float type;

        simd8f() {}
        simd8f(const simd4f& low_, const simd4f& high_): _low(low_),_high(high_){}
        simd8f(float f) :_low(f),_high(f) {}
        simd8f(float r0, float r1, float r2, float r3, float r4, float r5, float r6, float r7) :
            _low(r0,r1,r2,r3), _high(r4,r5,r6,r7) {}

        void load_aligned(const type* ptr)  { _low.load_aligned(ptr); _high.load_aligned(ptr+4); }
        void store_aligned(type* ptr) const { _low.store_aligned(ptr); _high.store_aligned(ptr+4); }
        void load(const type* ptr)          { _low.load(ptr); _high.load(ptr+4); }
        void store(type* ptr)         const { _low.store(ptr); _high.store(ptr+4); }

        unsigned int size() const { return 8; }
        float operator[](unsigned int idx) const 
        {
            if (idx < 4)
                return _low[idx];
            else
                return _high[idx-4];
        }

        simd4f low() const { return _low; }
        simd4f high() const { return _high; }

    private:
        simd4f _low, _high;
    };
#endif

// ----------------------------------------------------------------------------------------

    inline std::ostream& operator<<(std::ostream& out, const simd8f& item)
    {
        float temp[8];
        item.store(temp);
        out << "(" << temp[0] << ", " << temp[1] << ", " << temp[2] << ", " << temp[3] << ", "
                   << temp[4] << ", " << temp[5] << ", " << temp[6] << ", " << temp[7] << ")";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline simd8f operator+ (const simd8f& lhs, const simd8f& rhs) 
    { 
#ifdef DLIB_HAVE_AVX
        return _mm256_add_ps(lhs, rhs); 
#else
        return simd8f(lhs.low()+rhs.low(),
                      lhs.high()+rhs.high());
#endif
    }
    inline simd8f& operator+= (simd8f& lhs, const simd8f& rhs) 
    { return lhs = lhs + rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8f operator* (const simd8f& lhs, const simd8f& rhs) 
    { 
#ifdef DLIB_HAVE_AVX
        return _mm256_mul_ps(lhs, rhs); 
#else
        return simd8f(lhs.low()*rhs.low(),
                      lhs.high()*rhs.high());
#endif
    }
    inline simd8f& operator*= (simd8f& lhs, const simd8f& rhs) 
    { return lhs = lhs * rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline float sum(const simd8f& item)
    {
#ifdef DLIB_HAVE_AVX
        simd8f temp = _mm256_hadd_ps(item,item);
        simd8f temp2 = _mm256_hadd_ps(temp,temp);
        return _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(temp2),_mm256_extractf128_ps(temp2,1)));
#else
        return sum(item.low()+item.high()); 
#endif
    }

// ----------------------------------------------------------------------------------------

    inline float dot(const simd8f& lhs, const simd8f& rhs)
    {
        return sum(lhs*rhs);
    }
   
// ----------------------------------------------------------------------------------------

}

#endif // DLIB_sIMD8F_H__

