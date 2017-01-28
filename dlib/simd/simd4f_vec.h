// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD4F_VEC_Hh_
#define DLIB_SIMD4F_VEC_Hh_

#include "../uintn.h"
#include "simd4i_vec.h"

namespace dlib
{
    typedef simd4i simd4f_bool;		// For existing code
    
    class simd4f
    {
        typedef float v4f __attribute__ ((vector_size (16)));
        
        v4f x;
        
    public:
        
        inline simd4f() {}
        inline simd4f(const v4f& v) : x(v) { }
        inline simd4f(const simd4f& v) : x(v.x) { }
        inline simd4f(const simd4i& v) {
            x[0]=v[0]; x[1]=v[1]; x[2]=v[2]; x[3]=v[3];
        }
    
        inline simd4f(float f) : x(((v4f){f,f,f,f})) { }
        inline simd4f(float r0, float r1, float r2, float r3)
             : x(((v4f){r0,r1,r2,r3})) { }
        
        inline simd4f& operator=(const simd4f& v) { x = v.x; return *this; }
        inline simd4f& operator=(const float& v) { *this = simd4f(v); return *this; }
        

        inline v4f operator() () const { return x; }
        inline float operator[](unsigned int idx) const { return x[idx]; }
        

        inline simd4f operator- () { return -x; }
        inline simd4f operator! () { return !x; }
/*
        inline simd4f operator+ (const simd4f& rhs) {	return (x + rhs.x) ;	}
        inline simd4f operator- (const simd4f& rhs) {	return (x - rhs.x) ;	}
        inline simd4f operator* (const simd4f& rhs) {	return (x * rhs.x) ;	}
        inline simd4f operator/ (const simd4f& rhs) {	return (x / rhs.x) ;	}

        inline simd4f& operator+= (const simd4f& rhs) { x = x + rhs.x; return *this; }
        inline simd4f& operator-= (const simd4f& rhs) { x = x - rhs.x; return *this; }
        inline simd4f& operator*= (const simd4f& rhs) { x = x * rhs.x; return *this; }
        inline simd4f& operator/= (const simd4f& rhs) { x = x / rhs.x; return *this; }
*/
        // These will always return integer type of same length //
        inline simd4i operator== (const simd4f& rhs) { return simd4i(x==rhs.x); }
        inline simd4i operator!= (const simd4f& rhs) { return simd4i(x!=rhs.x); }
        inline simd4i operator<  (const simd4f& rhs) { return simd4i(x< rhs.x); }
        inline simd4i operator>  (const simd4f& rhs) { return simd4i(x> rhs.x); }
        inline simd4i operator<= (const simd4f& rhs) { return simd4i(x<=rhs.x); }
        inline simd4i operator>= (const simd4f& rhs) { return simd4i(x>=rhs.x); }

        
        
        inline void load(const float* p)  { x[0]=p[0]; x[1]=p[1]; x[2]=p[2]; x[3]=p[3]; }
        inline void store(float* p) const { p[0]=x[0]; p[1]=x[1]; p[2]=x[2]; p[3]=x[3]; }
        
        
        // truncate to 32bit integers
        inline operator simd4i::rawarray() const 
        { 
            simd4i::rawarray temp;
            temp.a[0] = (int32)x[0];
            temp.a[1] = (int32)x[1];
            temp.a[2] = (int32)x[2];
            temp.a[3] = (int32)x[3];
            return temp;
        }
    };
    
    inline std::ostream& operator<<(std::ostream& out, const simd4f& item)
    {
        out << "(" << item[0] << ", " << item[1] << ", " << item[2] << ", " << item[3] << ")";
        return out;
    }
    
    inline simd4f operator+ (const simd4f& lhs, const simd4f& rhs) {	return (lhs() + rhs()) ;	}
    inline simd4f operator- (const simd4f& lhs, const simd4f& rhs) {	return (lhs() - rhs()) ;	}
    inline simd4f operator* (const simd4f& lhs, const simd4f& rhs) {	return (lhs() * rhs()) ;	}
    inline simd4f operator/ (const simd4f& lhs, const simd4f& rhs) {	return (lhs() / rhs()) ;	}
    
    inline simd4f operator+=(simd4f& lhs, const simd4f& rhs) {	lhs = lhs + rhs; return lhs;	}
    inline simd4f operator-=(simd4f& lhs, const simd4f& rhs) {	lhs = lhs - rhs; return lhs;	}
    inline simd4f operator*=(simd4f& lhs, const simd4f& rhs) {	lhs = lhs * rhs; return lhs;	}
    inline simd4f operator/=(simd4f& lhs, const simd4f& rhs) {  lhs = lhs / rhs; return lhs;	}
    

    
    inline float sum(const simd4f& v) { return v[0]+v[1]+v[2]+v[3]; }
    inline float dot(const simd4f& a, const simd4f& b) { return sum(a*b); }
    
    inline simd4f min (const simd4f& lhs, const simd4f& rhs) { return ((lhs()<rhs()) ? lhs() : rhs()); }
    inline simd4f max (const simd4f& lhs, const simd4f& rhs) { return ((lhs()>rhs()) ? lhs() : rhs()); }

    inline simd4f select(const simd4i& cmp, const simd4f& a, const simd4f& b) { return cmp() ? a() : b(); }



    // TODO Check compilers results for these,  vs SIMD instructions
    
    inline simd4f sqrt (const simd4f& v) { 
        return simd4f(std::sqrt(v[0]), std::sqrt(v[1]), std::sqrt(v[2]), std::sqrt(v[3]));
    }
    inline simd4f ceil (const simd4f& v) { 
        return simd4f(std::ceil(v[0]), std::ceil(v[1]), std::ceil(v[2]), std::ceil(v[3]));
    }
    inline simd4f floor(const simd4f& v) {
        return simd4f(std::floor(v[0]), std::floor(v[1]), std::floor(v[2]), std::floor(v[3]));
    }
    
    inline simd4f reciprocal (const simd4f& item)  { return 1.f / item(); }
    inline simd4f reciprocal_sqrt (const simd4f& item) { return 1.f / sqrt(item); }
    
    



} // namespace dlib

#endif // DLIB_SIMD4F_VEC_Hh_

