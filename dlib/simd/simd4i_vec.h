// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD4I_VEC_Hh_
#define DLIB_SIMD4I_VEC_Hh_

#include "../uintn.h"


namespace dlib
{

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

    inline std::ostream& operator<<(std::ostream& out, const simd4i& item) {
        out << "(" << item[0] << ", " << item[1] << ", " << item[2] << ", " << item[3] << ")";
        return out;
    }

    inline simd4i operator+ (const simd4i& lhs, const simd4i& rhs) { return vec_add(lhs(), rhs());	}
    inline simd4i operator- (const simd4i& lhs, const simd4i& rhs) { return vec_sub(lhs(), rhs());	}
    inline simd4i operator* (const simd4i& lhs, const simd4i& rhs) {
        vector int a = lhs(), b = rhs();
        asm("vmuluwm %0, %0, %1\n\t" : "+&v" (a) : "v" (b) );
        return simd4i(a);	
    }
    
    inline simd4i operator& (const simd4i& lhs, const simd4i& rhs) { return vec_and(lhs(), rhs()); }
    inline simd4i operator| (const simd4i& lhs, const simd4i& rhs) { return vec_or(lhs(), rhs());  }
    inline simd4i operator^ (const simd4i& lhs, const simd4i& rhs) { return vec_xor(lhs(), rhs()); }


    inline simd4i operator~ (const simd4i& lhs) {
        simd4i n(~0); return vec_xor(lhs(), n());
    }

    inline simd4i operator<< (const simd4i& lhs, const unsigned int& rhs) {
        return vec_sl(lhs(), vec_splats(rhs)); 
    }
    inline simd4i operator>> (const simd4i& lhs, const unsigned int& rhs) {
        return vec_sr(lhs(), vec_splats(rhs)); 
    }

    inline simd4i& operator+=(simd4i& lhs, const simd4i& rhs) { lhs = lhs + rhs; return lhs; }
    inline simd4i& operator-=(simd4i& lhs, const simd4i& rhs) { lhs = lhs - rhs; return lhs; }
    inline simd4i& operator*=(simd4i& lhs, const simd4i& rhs) { lhs = lhs * rhs; return lhs; }
    
    inline simd4i& operator&= (simd4i& lhs, const simd4i& rhs) { lhs = lhs & rhs; return lhs; }
    inline simd4i& operator|= (simd4i& lhs, const simd4i& rhs) { lhs = lhs | rhs; return lhs; }
    inline simd4i& operator^= (simd4i& lhs, const simd4i& rhs) { lhs = lhs ^ rhs; return lhs; }
        
    inline simd4i& operator<<= (simd4i& lhs, const unsigned int& rhs) { return lhs = lhs << rhs; return lhs; }
    inline simd4i& operator>>= (simd4i& lhs, const unsigned int& rhs) { return lhs = lhs >> rhs; return lhs; }
    
    inline simd4i operator==(const simd4i& lhs, const simd4i& rhs) { return vec_cmpeq(lhs(), rhs()); }
    inline simd4i operator!=(const simd4i& lhs, const simd4i& rhs) { return ~(lhs==rhs); }
    inline simd4i operator< (const simd4i& lhs, const simd4i& rhs) { return vec_cmplt(lhs(), rhs()); }
    inline simd4i operator> (const simd4i& lhs, const simd4i& rhs) { return vec_cmpgt(lhs(), rhs()); }
    inline simd4i operator<=(const simd4i& lhs, const simd4i& rhs) { return vec_cmple(lhs(), rhs()); }
    inline simd4i operator>=(const simd4i& lhs, const simd4i& rhs) { return vec_cmpge(lhs(), rhs()); }
    
    inline simd4i min (const simd4i& lhs, const simd4i& rhs) { return vec_min(lhs(), rhs()); }
    inline simd4i max (const simd4i& lhs, const simd4i& rhs) { return vec_max(lhs(), rhs()); }
    inline simd4i select(const simd4i& cmp, const simd4i& a, const simd4i& b) {
        return vec_sel(b(), a(), cmp.to_bool());
    }
    
    inline int32 sum(const simd4i& item) { return item[0]+item[1]+item[2]+item[2]; }
    

} // namespace dlib

#endif // DLIB_SIMD4I_VEC_Hh_

