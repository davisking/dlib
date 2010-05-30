// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SAMPLE_PaIR_ABSTRACT_H__
#ifdef DLIB_SAMPLE_PaIR_ABSTRACT_H__

#include <limits>
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class sample_pair 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is intended to represent an edge in an undirected graph 
                which has data samples at its vertices.  So it contains two integers
                (index1 and index2) which represent the identifying indices of 
                the samples at the ends of an edge.  Note that this object enforces
                the constraint that index1 <= index2.  This has the effect of 
                making the edges undirected since a sample_pair is incapable
                of representing a single edge in more than one way.  That is,
                sample_pair(i,j) == sample_pair(j,i) for any value of i and j.

                This object also contains a float which can be used for any purpose.
        !*/

    public:
        sample_pair(
        );
        /*!
            ensures
                - #index1() == 0
                - #index2() == 0
                - #distance() == std::numeric_limits<float>::infinity()
        !*/

        sample_pair (
            const unsigned long idx1,
            const unsigned long idx2,
            const float dist
        );
        /*!
            ensures
                - #index1() == min(idx1,idx2)
                - #index2() == max(idx1,idx2)
                - #distance() == dist
        !*/

        const unsigned long& index1 (
        ) const; 
        /*!
            ensures
                - returns the first index value stored in this object 
        !*/

        const unsigned long& index2 (
        ) const; 
        /*!
            ensures
                - returns the second index value stored in this object 
        !*/

        const float& distance (
        ) const;
        /*!
            ensures
                - returns the floating point number stored in this object
        !*/

    };

// ----------------------------------------------------------------------------------------

    inline bool order_by_index (
        const sample_pair& a,
        const sample_pair& b
    ) { return a.index1() < b.index1() || (a.index1() == b.index1() && a.index2() < b.index2()); }
    /*!
        ensures
            - provides a total ordering of sample_pair objects that will cause pairs that are 
              equal to be adjacent when sorted.  This function can be used with std::sort() to
              first sort sequences of sample_pair objects and then find duplicate edges.
    !*/

    inline bool order_by_distance (
        const sample_pair& a,
        const sample_pair& b
    ) { return a.distance() < b.distance(); }
    /*!
        ensures
            - provides a total ordering of sample_pair objects that causes pairs with 
              smallest distance to be the first in a sorted list.  This function can be
              used with std::sort()
    !*/

// ----------------------------------------------------------------------------------------

    inline bool operator == (
        const sample_pair& a,
        const sample_pair& b
    );
    /*!
        ensures
            - returns a.index1() == b.index1() && a.index2() == b.index2();
              I.e. returns true if a and b both represent the same pair and false otherwise.  
              Note that the distance field is not involved in this comparison.
    !*/

    inline bool operator != (
        const sample_pair& a,
        const sample_pair& b
    );
    /*!
        ensures
            - returns !(a == b)
    !*/

// ----------------------------------------------------------------------------------------

    inline void serialize (
        const sample_pair& item,
        std::ostream& out
    );
    /*!
        provides serialization support 
    !*/

    inline void deserialize (
        sample_pair& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SAMPLE_PaIR_ABSTRACT_H__


