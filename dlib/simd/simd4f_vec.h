// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD4F_VEC_Hh_
#define DLIB_SIMD4F_VEC_Hh_

#include "../uintn.h"
#include "simd4i_vec.h"

namespace dlib
{
    typedef simd4i simd4f_bool;

    class simd4f
    {    
        typedef union {
            vector float v;
            float x[4];
        } v4f;
        
        v4f x;
        
    public:
        inline simd4f() : x{0,0,0,0} {}
        inline simd4f(const simd4f& v) : x(v.x) { }
        inline simd4f(const vector float& v) : x{v} { }

        inline simd4f(const simd4i& v) {
            x.x[0]=v[0]; x.x[1]=v[1]; x.x[2]=v[2]; x.x[3]=v[3];
        }
        
        
        inline simd4f(float f) : x{f,f,f,f} { }
        inline simd4f(float r0, float r1, float r2, float r3)
             : x{r0,r1,r2,r3} { }

        inline simd4f& operator=(const simd4f& v) { x = v.x; return *this; }
        inline simd4f& operator=(const float& v) { *this = simd4f(v); return *this; }

        inline vector float operator() () const { return x.v; }
        inline float operator[](unsigned int idx) const { return x.x[idx]; }
        
        inline void load_aligned(const float* ptr)  { x.v = vec_ld(0, ptr); }
        inline void store_aligned(float* ptr) const { vec_st(x.v, 0, ptr); }
        inline void load(const float* ptr) { x.v = vec_vsx_ld(0, ptr); }
        inline void store(float* ptr) const { vec_vsx_st(x.v, 0, ptr); }
        
        
        // truncate to 32bit integers
        inline operator simd4i::rawarray() const 
        { 
            simd4i::rawarray temp;
            temp.v.x[0] = x.x[0];
            temp.v.x[1] = x.x[1];
            temp.v.x[2] = x.x[2];
            temp.v.x[3] = x.x[3];
            return temp;
        }
    };

    inline std::ostream& operator<<(std::ostream& out, const simd4f& item) {
        out << "(" << item[0] << ", " << item[1] << ", " << item[2] << ", " << item[3] << ")";
        return out;
    }

    inline simd4f operator+ (const simd4f& lhs, const simd4f& rhs) { return vec_add(lhs(), rhs()); }
    inline simd4f operator- (const simd4f& lhs, const simd4f& rhs) { return vec_sub(lhs(), rhs()); }
    inline simd4f operator* (const simd4f& lhs, const simd4f& rhs) { return vec_mul(lhs(), rhs()); }
    inline simd4f operator/ (const simd4f& lhs, const simd4f& rhs) { return vec_div(lhs(), rhs()); }
    

    inline simd4f operator+=(simd4f& lhs, const simd4f& rhs) { lhs = lhs + rhs; return lhs; }
    inline simd4f operator-=(simd4f& lhs, const simd4f& rhs) { lhs = lhs - rhs; return lhs; }
    inline simd4f operator*=(simd4f& lhs, const simd4f& rhs) { lhs = lhs * rhs; return lhs; }
    

    inline simd4i operator==(const simd4f& lhs, const simd4f& rhs) { return vec_cmpeq(lhs(), rhs()); }
    inline simd4i operator!=(const simd4f& lhs, const simd4f& rhs) { return ~(lhs==rhs); }
    inline simd4i operator< (const simd4f& lhs, const simd4f& rhs) { return vec_cmplt(lhs(), rhs()); }
    inline simd4i operator> (const simd4f& lhs, const simd4f& rhs) { return vec_cmpgt(lhs(), rhs()); }
    inline simd4i operator<=(const simd4f& lhs, const simd4f& rhs) { return vec_cmple(lhs(), rhs()); }
    inline simd4i operator>=(const simd4f& lhs, const simd4f& rhs) { return vec_cmpge(lhs(), rhs()); }
    
    inline simd4f min (const simd4f& lhs, const simd4f& rhs) { return vec_min(lhs(), rhs()); }
    inline simd4f max (const simd4f& lhs, const simd4f& rhs) { return vec_max(lhs(), rhs()); }
    inline simd4f select(const simd4i& cmp, const simd4f& a, const simd4f& b) {
        return vec_sel(b(), a(), cmp.to_bool());
    }
    
    inline float sum(const simd4f& item) { return item[0]+item[1]+item[2]+item[2]; }
    inline float dot(const simd4f& lhs, const simd4f& rhs) { return sum(lhs*rhs); }
    
    inline simd4f sqrt(const simd4f& item) { return vec_sqrt(item()); }
    inline simd4f reciprocal (const simd4f& item) { return vec_re(item()); }
    inline simd4f reciprocal_sqrt (const simd4f& item) { return vec_rsqrt(item()); }
    
    
    inline simd4f ceil(const simd4f& item)  { return vec_ceil(item());  }
    inline simd4f floor(const simd4f& item) { return vec_floor(item()); }
    

} // namespace dlib

#endif // DLIB_SIMD4F_VEC_Hh_

