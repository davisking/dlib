// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MURMUR_HAsH_3_H__ 
#define DLIB_MURMUR_HAsH_3_H__ 

#include "murmur_hash3_abstract.h"
#include "../uintn.h"

namespace dlib
{
    //-----------------------------------------------------------------------------
    // The original MurmurHash3 code was written by Austin Appleby, and is placed 
    // in the public domain. The author hereby disclaims copyright to this source code.
    // The code in this particular file was modified by Davis E. King.  In
    // particular, endian-swapping was added along with some other minor code
    // changes.


    //-----------------------------------------------------------------------------
    // Platform-specific functions and macros

    // Microsoft Visual Studio

#if defined(_MSC_VER)

#define DLIB_FORCE_INLINE	__forceinline

#include <stdlib.h>

#define DLIB_ROTL32(x,y)	_rotl(x,y)


    // Other compilers

#else	// defined(_MSC_VER)

#define	DLIB_FORCE_INLINE __attribute__((always_inline))

    inline uint32 murmur_rotl32 ( uint32 x, int8 r )
    {
        return (x << r) | (x >> (32 - r));
    }

#define	DLIB_ROTL32(x,y)	murmur_rotl32(x,y)


#endif // !defined(_MSC_VER)

// ----------------------------------------------------------------------------------------
    // Block read - if your platform needs to do endian-swapping or can only
    // handle aligned reads, do the conversion here

    DLIB_FORCE_INLINE uint32 murmur_getblock ( const uint32 * p, int i )
    {
        return p[i];
    }

    DLIB_FORCE_INLINE uint32 murmur_getblock_byte_swap ( const uint32 * p, int i )
    {
        union 
        {
            uint8 bytes[4];
            uint32 val;
        } temp;

        const uint8* pp = reinterpret_cast<const uint8*>(p + i);
        temp.bytes[0] = pp[3];
        temp.bytes[1] = pp[2];
        temp.bytes[2] = pp[1];
        temp.bytes[3] = pp[0];

        return temp.val;
    }

// ----------------------------------------------------------------------------------------
    // Finalization mix - force all bits of a hash block to avalanche

    DLIB_FORCE_INLINE uint32 fmix ( uint32 h )
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;

        return h;
    }

// ----------------------------------------------------------------------------------------

    inline uint32 murmur_hash3 ( 
        const void * key, 
        int len, 
        uint32 seed = 0
    )
    {
        const uint8 * data = (const uint8*)key;
        const int nblocks = len / 4;

        uint32 h1 = seed;

        uint32 c1 = 0xcc9e2d51;
        uint32 c2 = 0x1b873593;

        //----------
        // body

        const uint32 * blocks = (const uint32 *)(data + nblocks*4);

        bool is_little_endian = true;
        uint32 endian_test = 1;
        if (*reinterpret_cast<unsigned char*>(&endian_test) != 1)
            is_little_endian = false;


        if (is_little_endian)
        {
            for(int i = -nblocks; i; i++)
            {
                uint32 k1 = murmur_getblock(blocks,i);

                k1 *= c1;
                k1 = DLIB_ROTL32(k1,15);
                k1 *= c2;

                h1 ^= k1;
                h1 = DLIB_ROTL32(h1,13); 
                h1 = h1*5+0xe6546b64;
            }
        }
        else
        {
            for(int i = -nblocks; i; i++)
            {
                uint32 k1 = murmur_getblock_byte_swap(blocks,i);

                k1 *= c1;
                k1 = DLIB_ROTL32(k1,15);
                k1 *= c2;

                h1 ^= k1;
                h1 = DLIB_ROTL32(h1,13); 
                h1 = h1*5+0xe6546b64;
            }
        }

        //----------
        // tail

        const uint8 * tail = (const uint8*)(data + nblocks*4);

        uint32 k1 = 0;

        switch(len & 3)
        {
            case 3: k1 ^= tail[2] << 16;
            case 2: k1 ^= tail[1] << 8;
            case 1: k1 ^= tail[0];
                    k1 *= c1; k1 = DLIB_ROTL32(k1,15); k1 *= c2; h1 ^= k1;
        };

        //----------
        // finalization

        h1 ^= len;

        h1 = fmix(h1);

        return h1;
    } 

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MURMUR_HAsH_3_H__

