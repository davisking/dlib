// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SAMPLE_PaIR_H__
#define DLIB_SAMPLE_PaIR_H__

#include "sample_pair_abstract.h"
#include <limits>
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class sample_pair 
    {
    public:
        sample_pair(
        ) : 
            _index1(0),
            _index2(0)
        {
            _distance = std::numeric_limits<float>::infinity();
        }

        sample_pair (
            const unsigned long idx1,
            const unsigned long idx2,
            const float dist
        )
        {
            _distance = dist;
            if (idx1 < idx2)
            {
                _index1 = idx1;
                _index2 = idx2;
            }
            else
            {
                _index1 = idx2;
                _index2 = idx1;
            }
        }

        const unsigned long& index1 (
        ) const { return _index1; }

        const unsigned long& index2 (
        ) const { return _index2; }

        const float& distance (
        ) const { return _distance; }

    private:
        unsigned long _index1;
        unsigned long _index2;
        float _distance;
    };

// ----------------------------------------------------------------------------------------

    inline bool order_by_index (
        const sample_pair& a,
        const sample_pair& b
    )
    {
        return a.index1() < b.index1() || (a.index1() == b.index1() && a.index2() < b.index2());
    }

    inline bool order_by_distance (
        const sample_pair& a,
        const sample_pair& b
    )
    {
        return a.distance() < b.distance();
    }

// ----------------------------------------------------------------------------------------

    inline bool operator == (
        const sample_pair& a,
        const sample_pair& b
    ) 
    {
        return a.index1() == b.index1() && a.index2() == b.index2();
    }

    inline bool operator != (
        const sample_pair& a,
        const sample_pair& b
    ) 
    {
        return !(a == b); 
    }

// ----------------------------------------------------------------------------------------

    inline void serialize (
        const sample_pair& item,
        std::ostream& out
    )
    {
        try
        {
            serialize(item.index1(),out);
            serialize(item.index2(),out);
            serialize(item.distance(),out);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while serializing object of type sample_pair"); 
        }
    }

    inline void deserialize (
        sample_pair& item,
        std::istream& in 
    )
    {
        try
        {
            unsigned long idx1, idx2;
            float dist;

            deserialize(idx1,in);
            deserialize(idx2,in);
            deserialize(dist,in);
            item = sample_pair(idx1, idx2, dist);
        }
        catch (serialization_error& e)
        { 
            throw serialization_error(e.info + "\n   while deserializing object of type sample_pair"); 
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SAMPLE_PaIR_H__

