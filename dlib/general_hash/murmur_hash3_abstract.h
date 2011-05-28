// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MURMUR_HAsH_3_ABSTRACT_H__ 
#ifdef DLIB_MURMUR_HAsH_3_ABSTRACT_H__ 

#include "../uintn.h"

namespace dlib
{
    uint32 murmur_hash3 ( 
        const void* key, 
        int len, 
        uint32 seed = 0
    );
    /*!
        requires
            - key == a pointer to a block of memory len bytes long
        ensures
            - returns a 32bit hash of the len bytes pointed to by key.  
            - Each value of seed results in a different hash function being used.  
              (e.g. murmur_hash3(key,len,0) should generally not be equal to murmur_hash3(key,len,1))
            - This function is machine architecture agnostic and should always give the same
              hash value when presented with the same inputs.
            - This hashing algorithm is Austin Appleby's excellent MurmurHash3.  
              See: http://code.google.com/p/smhasher/
    !*/

}

#endif // DLIB_MURMUR_HAsH_3_ABSTRACT_H__


