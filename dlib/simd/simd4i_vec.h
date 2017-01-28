// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD4I_VEC_Hh_
#define DLIB_SIMD4I_VEC_Hh_

#include "../uintn.h"


namespace dlib
{
    class simd4i
    {
        typedef int32 v4i __attribute__ ((vector_size (16)));
        
        v4i x;
        
    public:	
        
        inline simd4i() {}
        inline simd4i(const v4i& v) : x(v) { }
        inline simd4i(const simd4i& v) : x(v.x) { }

        inline simd4i(int32 f) : x(((v4i){f,f,f,f})) { }
        inline simd4i(int32 r0, int32 r1, int32 r2, int32 r3)
             : x(((v4i){r0,r1,r2,r3})) { }		
    
        inline simd4i& operator=(const simd4i& v) { x = v.x; return *this; }
        inline simd4i& operator=(const int32& v) { *this = simd4i(v); return *this; }

        inline v4i operator() () const { return x; }
        inline int32 operator[](unsigned int idx) const { return x[idx]; }

        
        // unary negate
        inline simd4i operator- () { return -x; }
        inline simd4i operator! () { return !x; }
        
/*
        inline simd4i operator+ (const simd4i& rhs) {	return (x + rhs.x) ;	}
        inline simd4i operator- (const simd4i& rhs) {	return (x - rhs.x) ;	}
        inline simd4i operator* (const simd4i& rhs) {	return (x * rhs.x) ;	}
        inline simd4i operator/ (const simd4i& rhs) {	return (x / rhs.x) ;	}

        inline simd4i& operator+= (const simd4i& rhs) { x = x + rhs.x; return *this; }
        inline simd4i& operator-= (const simd4i& rhs) { x = x - rhs.x; return *this; }
        inline simd4i& operator*= (const simd4i& rhs) { x = x * rhs.x; return *this; }
        inline simd4i& operator/= (const simd4i& rhs) { x = x / rhs.x; return *this; }
*/
        // These will always return integer type of same length //
        inline simd4i operator== (const simd4i& rhs) { return (x==rhs.x); }
        inline simd4i operator!= (const simd4i& rhs) { return (x!=rhs.x); }
        inline simd4i operator<  (const simd4i& rhs) { return (x< rhs.x); }
        inline simd4i operator>  (const simd4i& rhs) { return (x> rhs.x); }
        inline simd4i operator<= (const simd4i& rhs) { return (x<=rhs.x); }
        inline simd4i operator>= (const simd4i& rhs) { return (x>=rhs.x); }


        // Binary / Integer
        inline simd4i operator~  () { return ~x; }
        inline simd4i operator&  (const simd4i& rhs) { return x & rhs.x; }
        inline simd4i operator|  (const simd4i& rhs) { return x | rhs.x; }
        inline simd4i operator^  (const simd4i& rhs) { return x ^ rhs.x; }
        inline simd4i operator<< (const simd4i& rhs) { return x << rhs.x; } 
        inline simd4i operator>> (const simd4i& rhs) { return x >> rhs.x; } 
        

        
        
        inline void load(const int* p)  { x[0]=p[0]; x[1]=p[1]; x[2]=p[2]; x[3]=p[3]; }
        inline void store(int* p) const { p[0]=x[0]; p[1]=x[1]; p[2]=x[2]; p[3]=x[3]; }
        
        // kludge to handle simd4i( simd4f_v );
        struct rawarray
        {
            int32 a[4];
        };
        inline simd4i(const rawarray& a) { x[0]=a.a[0]; x[1]=a.a[1]; x[2]=a.a[2]; x[3]=a.a[3]; }

    };
    
    inline std::ostream& operator<<(std::ostream& out, const simd4i& item)
    {
        out << "(" << item[0] << ", " << item[1] << ", " << item[2] << ", " << item[3] << ")";
        return out;
    }
    
    inline simd4i operator+ (const simd4i& lhs, const simd4i& rhs) {	return (lhs() + rhs()) ;	}
    inline simd4i operator- (const simd4i& lhs, const simd4i& rhs) {	return (lhs() - rhs()) ;	}
    inline simd4i operator* (const simd4i& lhs, const simd4i& rhs) {	return (lhs() * rhs()) ;	}
    inline simd4i operator/ (const simd4i& lhs, const simd4i& rhs) {	return (lhs() / rhs()) ;	}
    
    inline simd4i operator+=(simd4i& lhs, const simd4i& rhs) {	lhs = lhs + rhs; return lhs;	}
    inline simd4i operator-=(simd4i& lhs, const simd4i& rhs) {	lhs = lhs - rhs; return lhs;	}
    inline simd4i operator*=(simd4i& lhs, const simd4i& rhs) {	lhs = lhs * rhs; return lhs;	}
    inline simd4i operator/=(simd4i& lhs, const simd4i& rhs) {  lhs = lhs / rhs; return lhs;	}
    
    inline int sum(const simd4i& v) { return v[0]+v[1]+v[2]+v[3]; }

    inline simd4i min (const simd4i& lhs, const simd4i& rhs) { return ((lhs()<rhs()) ? lhs() : rhs()); }
    inline simd4i max (const simd4i& lhs, const simd4i& rhs) { return ((lhs()>rhs()) ? lhs() : rhs()); }

    inline simd4i select(const simd4i& cmp, const simd4i& a, const simd4i& b) { return cmp() ? a() : b(); }



} // namespace dlib

#endif // DLIB_SIMD4I_VEC_Hh_

