// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD8F_VEC_Hh_
#define DLIB_SIMD8F_VEC_Hh_

#include "../uintn.h"
#include "simd8i_vec.h"

namespace dlib
{
    typedef simd8i simd8f_bool;		// For existing code

    class simd8f
    {
        typedef float v8f __attribute__ ((vector_size (32)));
        
        v8f x;
        
    public:
        
        inline simd8f() {}
        inline simd8f(const v8f& v) : x(v) { }
        inline simd8f(const simd8f& v) : x(v.x) { }
        inline simd8f(const simd8i& v) {
            x[0]=v[0]; x[1]=v[1]; x[2]=v[2]; x[3]=v[3]; x[4]=v[4]; x[5]=v[5]; x[6]=v[6]; x[7]=v[7];
        }
            
        inline simd8f(float f) : x(((v8f){f,f,f,f,f,f,f,f})) { }
        inline simd8f(float r0, float r1, float r2, float r3, float r4, float r5, float r6, float r7)
             : x(((v8f){r0,r1,r2,r3,r4,r5,r6,r7})) { }
        
        inline simd8f& operator=(const simd8f& v) { x = v.x; return *this; }
        inline simd8f& operator=(const float& v) { *this = simd8f(v); return *this; }


        inline v8f operator() () const { return x; }
        inline float operator[](unsigned int idx) const { return x[idx]; }
        

        inline simd8f operator- () { return -x; }
        inline simd8f operator! () { return !x; }
 
       
        // These will always return integer type of same length //
        inline simd8i operator== (const simd8f& rhs) { return simd8i(x==rhs.x); }
        inline simd8i operator!= (const simd8f& rhs) { return simd8i(x!=rhs.x); }
        inline simd8i operator<  (const simd8f& rhs) { return simd8i(x< rhs.x); }
        inline simd8i operator>  (const simd8f& rhs) { return simd8i(x> rhs.x); }
        inline simd8i operator<= (const simd8f& rhs) { return simd8i(x<=rhs.x); }
        inline simd8i operator>= (const simd8f& rhs) { return simd8i(x>=rhs.x); }


        inline void load(const float* p)  { x[0]=p[0]; x[1]=p[1]; x[2]=p[2]; x[3]=p[3]; x[4]=p[4]; x[5]=p[5]; x[6]=p[6]; x[7]=p[7]; }
        inline void store(float* p) const { p[0]=x[0]; p[1]=x[1]; p[2]=x[2]; p[3]=x[3]; p[4]=x[4]; p[5]=x[5]; p[6]=x[6]; p[7]=x[7]; }
        
        // truncate to 32bit integers
        inline operator simd8i::rawarray() const 
        { 
            simd8i::rawarray temp;
            temp.a[0] = (int32)x[0]; temp.a[1] = (int32)x[1];
            temp.a[2] = (int32)x[2]; temp.a[3] = (int32)x[3];
            temp.a[4] = (int32)x[4]; temp.a[5] = (int32)x[5];
            temp.a[6] = (int32)x[6]; temp.a[7] = (int32)x[7];
            return temp;
        }
    };
    
    inline std::ostream& operator<<(std::ostream& out, const simd8f& item)
    {
        out << "(" << item[0] << ", " << item[1] << ", " << item[2] << ", " << item[3] << ", "
                   << item[4] << ", " << item[5] << ", " << item[6] << ", " << item[7] << ")";
        return out;
    }
    
    inline simd8f operator+ (const simd8f& lhs, const simd8f& rhs) {	return (lhs() + rhs()) ;	}
    inline simd8f operator- (const simd8f& lhs, const simd8f& rhs) {	return (lhs() - rhs()) ;	}
    inline simd8f operator* (const simd8f& lhs, const simd8f& rhs) {	return (lhs() * rhs()) ;	}
    inline simd8f operator/ (const simd8f& lhs, const simd8f& rhs) {	return (lhs() / rhs()) ;	}
    
    inline simd8f operator+=(simd8f& lhs, const simd8f& rhs) {	lhs = lhs + rhs; return lhs;	}
    inline simd8f operator-=(simd8f& lhs, const simd8f& rhs) {	lhs = lhs - rhs; return lhs;	}
    inline simd8f operator*=(simd8f& lhs, const simd8f& rhs) {	lhs = lhs * rhs; return lhs;	}
    inline simd8f operator/=(simd8f& lhs, const simd8f& rhs) {  lhs = lhs / rhs; return lhs;	}
    
    
    inline float sum(const simd8f& v) { return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7]; }
    inline float sum(const simd8f& a, const simd8f& b) { return sum(a*b); }
    
    inline simd8f min (const simd8f& lhs, const simd8f& rhs) { return ((lhs()<rhs()) ? lhs() : rhs()); }
    inline simd8f max (const simd8f& lhs, const simd8f& rhs) { return ((lhs()>rhs()) ? lhs() : rhs()); }

    inline simd8f select(const simd8i& cmp, const simd8f& a, const simd8f& b) { return cmp() ? a() : b(); }


    
    inline simd8f sqrt (const simd8f& v) { return simd8f(
        std::sqrt(v[0]), std::sqrt(v[1]), std::sqrt(v[2]), std::sqrt(v[3]),
        std::sqrt(v[4]), std::sqrt(v[5]), std::sqrt(v[6]), std::sqrt(v[7]));
    }
    inline simd8f ceil (const simd8f& v) { return simd8f(
        std::ceil(v[0]), std::ceil(v[1]), std::ceil(v[2]), std::ceil(v[3]),
        std::ceil(v[4]), std::ceil(v[5]), std::ceil(v[6]), std::ceil(v[7]));
    }
    inline simd8f floor(const simd8f& v) { return simd8f(
        std::floor(v[0]), std::floor(v[1]), std::floor(v[2]), std::floor(v[3]),
        std::floor(v[4]), std::floor(v[5]), std::floor(v[6]), std::floor(v[7]));
    }
    
    inline simd8f reciprocal (const simd8f& item)  { return 1.f / item(); }
    inline simd8f reciprocal_sqrt (const simd8f& item) { return 1.f / sqrt(item); }
    





} // namespace dlib

#endif // DLIB_SIMD8F_VEC_Hh_

