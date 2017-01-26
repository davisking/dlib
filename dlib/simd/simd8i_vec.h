// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMD8I_VEC_Hh_
#define DLIB_SIMD8I_VEC_Hh_

#include "../uintn.h"

namespace dlib
{

	class simd8i
	{
		typedef int32 v8i __attribute__ ((vector_size (32)));
		
		v8i x;
		
    public:	
		
		inline simd8i() {}
		inline simd8i(const v8i& v) : x(v) { }
		inline simd8i(const simd8i& v) : x(v.x) { }
		
		inline simd8i(int32 f) : x(((v8i){f,f,f,f,f,f,f,f})) { }
		inline simd8i(int32 r0, int32 r1, int32 r2, int32 r3, int32 r4, int32 r5, int32 r6, int32 r7)
			 : x(((v8i){r0,r1,r2,r3,r4,r5,r6,r7})) { }
		
        inline simd8i& operator=(const simd8i& v) { x = v.x; return *this; }
        inline simd8i& operator=(const int32& v) { *this = simd8i(v); return *this; }

		inline v8i operator() () const { return x; }
        inline int32 operator[](unsigned int idx) const { return x[idx]; }

		
		// unary negate
		inline simd8i operator- () { return -x; }
		inline simd8i operator! () { return !x; }
		
		
		inline simd8i operator+ (const simd8i& rhs) {	return (x + rhs.x) ;	}
		inline simd8i operator- (const simd8i& rhs) {	return (x - rhs.x) ;	}
		inline simd8i operator* (const simd8i& rhs) {	return (x * rhs.x) ;	}
		inline simd8i operator/ (const simd8i& rhs) {	return (x / rhs.x) ;	}

		inline simd8i& operator+= (const simd8i& rhs) { x = x + rhs.x; return *this; }
		inline simd8i& operator-= (const simd8i& rhs) { x = x - rhs.x; return *this; }
		inline simd8i& operator*= (const simd8i& rhs) { x = x * rhs.x; return *this; }
		inline simd8i& operator/= (const simd8i& rhs) { x = x / rhs.x; return *this; }
		
		// These will always return integer type of same length //
		inline simd8i operator== (const simd8i& rhs) { return (x==rhs.x); }
		inline simd8i operator!= (const simd8i& rhs) { return (x!=rhs.x); }
		inline simd8i operator<  (const simd8i& rhs) { return (x< rhs.x); }
		inline simd8i operator>  (const simd8i& rhs) { return (x> rhs.x); }
		inline simd8i operator<= (const simd8i& rhs) { return (x<=rhs.x); }
		inline simd8i operator>= (const simd8i& rhs) { return (x>=rhs.x); }


		// Binary / Integer
		inline simd8i operator~  () { return ~x; }
		inline simd8i operator&  (const simd8i& rhs) { return x & rhs.x; }
		inline simd8i operator|  (const simd8i& rhs) { return x | rhs.x; }
		inline simd8i operator^  (const simd8i& rhs) { return x ^ rhs.x; }
		inline simd8i operator<< (const simd8i& rhs) { return x << rhs.x; } 
		inline simd8i operator>> (const simd8i& rhs) { return x >> rhs.x; } 

		
		
        inline void load(const int* p)  { x[0]=p[0]; x[1]=p[1]; x[2]=p[2]; x[3]=p[3]; x[4]=p[4]; x[5]=p[5]; x[6]=p[6]; x[7]=p[7]; }
        inline void store(int* p) const { p[0]=x[0]; p[1]=x[1]; p[2]=x[2]; p[3]=x[3]; p[4]=x[4]; p[5]=x[5]; p[6]=x[6]; p[7]=x[7]; }

	
		friend std::ostream& operator<<(std::ostream& out, const simd8i& s)
		{
			size_t n = sizeof(s.x) / 4;

			out << "(";
			for (int i=0; i<n; i++)
				out << s.x[i] << (((i+1)==n)?"":", ") ;
			out << ")";
			return out;
		}
	};
	
	inline simd8i operator+ (const simd8i& lhs, const simd8i& rhs) {	return (lhs() + rhs()) ;	}
	inline simd8i operator- (const simd8i& lhs, const simd8i& rhs) {	return (lhs() - rhs()) ;	}
	inline simd8i operator* (const simd8i& lhs, const simd8i& rhs) {	return (lhs() * rhs()) ;	}
	inline simd8i operator/ (const simd8i& lhs, const simd8i& rhs) {	return (lhs() / rhs()) ;	}
	
	inline int sum(const simd8i& v) { return v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7]; }

	inline simd8i min (const simd8i& lhs, const simd8i& rhs) { return ((lhs()<rhs()) ? lhs() : rhs()); }
	inline simd8i max (const simd8i& lhs, const simd8i& rhs) { return ((lhs()>rhs()) ? lhs() : rhs()); }

    inline simd8i select(const simd8i& cmp, const simd8i& a, const simd8i& b) { return cmp() ? a() : b(); }




} // namespace dlib

#endif // DLIB_SIMD8I_VEC_Hh_

