// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_sIMD8I_Hh_
#define DLIB_sIMD8I_Hh_

#include "../uintn.h"

namespace dlib
{

    class simd8i
    {
    public:
        typedef int32 type;

        inline simd8i() {}
        inline simd8i(const simd4i& low_, const simd4i& high_): _low(low_),_high(high_){}
        inline simd8i(int32 f) :_low(f),_high(f) {}
        inline simd8i(int32 r0, int32 r1, int32 r2, int32 r3, int32 r4, int32 r5, int32 r6, int32 r7) :
            _low(r0,r1,r2,r3), _high(r4,r5,r6,r7) {}

        struct rawarray
        {
            simd4i low, high;
        };
        inline simd8i(const rawarray& a) 
        { 
            _low = a.low;
            _high = a.high;
        }

        inline void load_aligned(const type* ptr)  { _low.load_aligned(ptr); _high.load_aligned(ptr+4); }
        inline void store_aligned(type* ptr) const { _low.store_aligned(ptr); _high.store_aligned(ptr+4); }
        inline void load(const type* ptr)          { _low.load(ptr); _high.load(ptr+4); }
        inline void store(type* ptr)         const { _low.store(ptr); _high.store(ptr+4); }

        inline unsigned int size() const { return 8; }
        inline int32 operator[](unsigned int idx) const 
        {
            if (idx < 4)
                return _low[idx];
            else
                return _high[idx-4];
        }

        inline simd4i low() const { return _low; }
        inline simd4i high() const { return _high; }

    private:
        simd4i _low, _high;
    };


// ----------------------------------------------------------------------------------------

    inline std::ostream& operator<<(std::ostream& out, const simd8i& item)
    {
        int32 temp[8];
        item.store(temp);
        out << "(" << temp[0] << ", " << temp[1] << ", " << temp[2] << ", " << temp[3] << ", "
                   << temp[4] << ", " << temp[5] << ", " << temp[6] << ", " << temp[7] << ")";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator+ (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()+rhs.low(),
                      lhs.high()+rhs.high());
    }
    inline simd8i& operator+= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs + rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator- (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()-rhs.low(),
                      lhs.high()-rhs.high());
    }
    inline simd8i& operator-= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs - rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator* (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()*rhs.low(),
                      lhs.high()*rhs.high());
    }
    inline simd8i& operator*= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs * rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator& (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()&rhs.low(),
                      lhs.high()&rhs.high());
    }
    inline simd8i& operator&= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs & rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator| (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()|rhs.low(),
                      lhs.high()|rhs.high());
    }
    inline simd8i& operator|= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs | rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator^ (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()^rhs.low(),
                      lhs.high()^rhs.high());
    }
    inline simd8i& operator^= (simd8i& lhs, const simd8i& rhs) 
    { return lhs = lhs ^ rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator~ (const simd8i& lhs) 
    { 
        return simd8i(~lhs.low(), ~lhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator<< (const simd8i& lhs, const int& rhs) 
    { 
        return simd8i(lhs.low()<<rhs,
                      lhs.high()<<rhs);
    }
    inline simd8i& operator<<= (simd8i& lhs, const int& rhs) 
    { return lhs = lhs << rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator>> (const simd8i& lhs, const int& rhs) 
    { 
        return simd8i(lhs.low()>>rhs,
                      lhs.high()>>rhs);
    }
    inline simd8i& operator>>= (simd8i& lhs, const int& rhs) 
    { return lhs = lhs >> rhs; return lhs;}

// ----------------------------------------------------------------------------------------

    inline simd8i operator== (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()==rhs.low(),
                      lhs.high()==rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator!= (const simd8i& lhs, const simd8i& rhs) 
    { 
        return ~(lhs==rhs);
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator> (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(lhs.low()>rhs.low(),
                      lhs.high()>rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator< (const simd8i& lhs, const simd8i& rhs) 
    { 
        return rhs > lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator<= (const simd8i& lhs, const simd8i& rhs) 
    { 
        return ~(lhs > rhs); 
    }

// ----------------------------------------------------------------------------------------

    inline simd8i operator>= (const simd8i& lhs, const simd8i& rhs) 
    { 
        return rhs <= lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd8i min (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(min(lhs.low(),rhs.low()),
                      min(lhs.high(),rhs.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8i max (const simd8i& lhs, const simd8i& rhs) 
    { 
        return simd8i(max(lhs.low(),rhs.low()),
                      max(lhs.high(),rhs.high()));
    }

// ----------------------------------------------------------------------------------------

    inline int32 sum(const simd8i& item)
    {
        return sum(item.low()+item.high());
    }

// ----------------------------------------------------------------------------------------

    // perform cmp ? a : b
    inline simd8i select(const simd8i& cmp, const simd8i& a, const simd8i& b)
    {
        return simd8i(select(cmp.low(),  a.low(),  b.low()),
                      select(cmp.high(), a.high(), b.high()));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_sIMD8I_Hh_


