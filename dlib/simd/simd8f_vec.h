// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_sIMD8F_Hh_
#define DLIB_sIMD8F_Hh_

#include "simd4f_vec.h"
#include "simd8i_vec.h"

namespace dlib
{
    class simd8f
    {
    public:
        typedef float type;

        inline simd8f() {}
        inline simd8f(const simd4f& low_, const simd4f& high_): _low(low_),_high(high_){}
        inline simd8f(float f) :_low(f),_high(f) {}
        inline simd8f(float r0, float r1, float r2, float r3, float r4, float r5, float r6, float r7) :
            _low(r0,r1,r2,r3), _high(r4,r5,r6,r7) {}
        inline simd8f(const simd8i& val) : _low(val.low()), _high(val.high()) { }

        // truncate to 32bit integers
        inline operator simd8i::rawarray() const 
        { 
            simd8i::rawarray temp;
            temp.low = simd4i(_low);
            temp.high = simd4i(_high);
            return temp;
        }

        inline void load_aligned(const type* ptr)  { _low.load_aligned(ptr); _high.load_aligned(ptr+4); }
        inline void store_aligned(type* ptr) const { _low.store_aligned(ptr); _high.store_aligned(ptr+4); }
        inline void load(const type* ptr)          { _low.load(ptr); _high.load(ptr+4); }
        inline void store(type* ptr)         const { _low.store(ptr); _high.store(ptr+4); }

        inline unsigned int size() const { return 8; }
        inline float operator[](unsigned int idx) const 
        {
            if (idx < 4)
                return _low[idx];
            else
                return _high[idx-4];
        }

        inline simd4f low() const { return _low; }
        inline simd4f high() const { return _high; }

    private:
        simd4f _low, _high;
    };

    class simd8f_bool
    {
    public:
        typedef float type;

        inline simd8f_bool() {}
        inline simd8f_bool(const simd4f_bool& low_, const simd4f_bool& high_): _low(low_),_high(high_){}


        inline simd4f_bool low() const { return _low; }
        inline simd4f_bool high() const { return _high; }
    private:
        simd4f_bool _low,_high;
    };


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
        return simd8f(lhs.low()+rhs.low(),
                      lhs.high()+rhs.high());
    }
    inline simd8f& operator+= (simd8f& lhs, const simd8f& rhs) 
    { lhs = lhs + rhs; return lhs; }

// ----------------------------------------------------------------------------------------

    inline simd8f operator- (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f(lhs.low()-rhs.low(),
                      lhs.high()-rhs.high());

    }
    inline simd8f& operator-= (simd8f& lhs, const simd8f& rhs) 
    { lhs = lhs - rhs; return lhs; }

// ----------------------------------------------------------------------------------------

    inline simd8f operator* (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f(lhs.low()*rhs.low(),
                      lhs.high()*rhs.high());
    }
    inline simd8f& operator*= (simd8f& lhs, const simd8f& rhs) 
    { lhs = lhs * rhs; return lhs; }

// ----------------------------------------------------------------------------------------

    inline simd8f operator/ (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f(lhs.low()/rhs.low(),
                      lhs.high()/rhs.high());
    }
    inline simd8f& operator/= (simd8f& lhs, const simd8f& rhs) 
    { lhs = lhs / rhs; return lhs; }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator== (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f_bool(lhs.low() ==rhs.low(),
                      lhs.high()==rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator!= (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f_bool(lhs.low() !=rhs.low(),
                      lhs.high()!=rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator< (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f_bool(lhs.low() <rhs.low(),
                      lhs.high()<rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator> (const simd8f& lhs, const simd8f& rhs) 
    { 
        return rhs < lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator<= (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f_bool(lhs.low() <=rhs.low(),
                      lhs.high()<=rhs.high());
    }

// ----------------------------------------------------------------------------------------

    inline simd8f_bool operator>= (const simd8f& lhs, const simd8f& rhs) 
    { 
        return rhs <= lhs;
    }

// ----------------------------------------------------------------------------------------

    inline simd8f min (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f(min(lhs.low(), rhs.low()),
                      min(lhs.high(),rhs.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8f max (const simd8f& lhs, const simd8f& rhs) 
    { 
        return simd8f(max(lhs.low(), rhs.low()),
                      max(lhs.high(),rhs.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8f reciprocal (const simd8f& item) 
    { 
        return simd8f(reciprocal(item.low()),
                      reciprocal(item.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8f reciprocal_sqrt (const simd8f& item) 
    { 
        return simd8f(reciprocal_sqrt(item.low()),
                      reciprocal_sqrt(item.high()));
    }

// ----------------------------------------------------------------------------------------

    inline float sum(const simd8f& item)
    {
        return sum(item.low()+item.high()); 
    }

// ----------------------------------------------------------------------------------------

    inline float dot(const simd8f& lhs, const simd8f& rhs)
    {
        return sum(lhs*rhs);
    }
   
// ----------------------------------------------------------------------------------------

    inline simd8f sqrt(const simd8f& item)
    {
        return simd8f(sqrt(item.low()),
                      sqrt(item.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8f ceil(const simd8f& item)
    {
        return simd8f(ceil(item.low()),
                      ceil(item.high()));
    }

// ----------------------------------------------------------------------------------------

    inline simd8f floor(const simd8f& item)
    {
        return simd8f(floor(item.low()),
                      floor(item.high()));
    }

// ----------------------------------------------------------------------------------------

    // perform cmp ? a : b
    inline simd8f select(const simd8f_bool& cmp, const simd8f& a, const simd8f& b)
    {
        return simd8f(select(cmp.low(),  a.low(),  b.low()),
                      select(cmp.high(), a.high(), b.high()));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_sIMD8F_Hh_

