// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_HAsH_ABSTRACT_H__
#ifdef DLIB_HAsH_ABSTRACT_H__ 

#include "murmur_hash3_abstract.h"
#include <vector>
#include <string>
#include <map>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    uint32 hash (
        const std::string& item,
        uint32 seed = 0
    );
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.  
            - Each value of seed results in a different hash function being used.  
              (e.g. hash(item,0) should generally not be equal to hash(item,1))
            - uses the murmur_hash3() routine to compute the actual hash.
    !*/

// ----------------------------------------------------------------------------------------

    uint32 hash (
        const std::wstring& item,
        uint32 seed = 0
    );
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.  
            - Each value of seed results in a different hash function being used.  
              (e.g. hash(item,0) should generally not be equal to hash(item,1))
            - uses the murmur_hash3() routine to compute the actual hash.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename alloc>
    uint32 hash (
        const std::vector<T,alloc>& item,
        uint32 seed = 0
    );
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.  
            - Each value of seed results in a different hash function being used.  
              (e.g. hash(item,0) should generally not be equal to hash(item,1))
            - uses the murmur_hash3() routine to compute the actual hash.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename U, typename alloc>
    uint32 hash (
        const std::vector<std::pair<T,U>,alloc>& item,
        uint32 seed = 0
    );
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.  
            - Each value of seed results in a different hash function being used.  
              (e.g. hash(item,0) should generally not be equal to hash(item,1))
            - uses the murmur_hash3() routine to compute the actual hash.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename T, typename U, typename comp, typename alloc>
    uint32 hash (
        const std::map<T,U,comp,alloc>& item,
        uint32 seed = 0
    );
    /*!
        ensures
            - returns a 32bit hash of the data stored in item.  
            - Each value of seed results in a different hash function being used.  
              (e.g. hash(item,0) should generally not be equal to hash(item,1))
            - uses the murmur_hash3() routine to compute the actual hash.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_HAsH_ABSTRACT_H__


