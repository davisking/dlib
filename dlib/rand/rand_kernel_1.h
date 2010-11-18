// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RAND_KERNEl_1_
#define DLIB_RAND_KERNEl_1_

#include <string>
#include "../algs.h"
#include "rand_kernel_abstract.h"
#include "mersenne_twister.h"
#include "../is_kind.h"
#include <iostream>
#include "../serialize.h"

namespace dlib
{


    class rand_kernel_1
    {

        /*!       
            INITIAL VALUE
                - seed == ""

            CONVENTION
                - the random numbers come from the boost mersenne_twister code
                - get_seed() == seed
        !*/
        
        public:

            rand_kernel_1(
            ) 
            {
                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();
            }

            virtual ~rand_kernel_1(
            )
            {}

            void clear(
            )
            {
                mt.seed();
                seed.clear();

                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();
            }
 
            const std::string& get_seed (
            )
            {
                return seed;
            }

            void set_seed (
                const std::string& value
            )
            {
                seed = value;

                // make sure we do the seeding so that using a seed of "" gives the same
                // state as calling this->clear()
                if (value.size() != 0)
                {
                    uint32 s = 0;
                    for (std::string::size_type i = 0; i < seed.size(); ++i)
                    {
                        s = (s*37) + static_cast<uint32>(seed[i]);
                    }
                    mt.seed(s);
                }
                else
                {
                    mt.seed();
                }

                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();
            }

            unsigned char get_random_8bit_number (
            )
            {
                return static_cast<unsigned char>(mt());
            }

            uint16 get_random_16bit_number (
            )
            {
                return static_cast<uint16>(mt());
            }

            inline uint32 get_random_32bit_number (
            )
            {
                return mt();
            }

            void swap (
                rand_kernel_1& item
            )
            {
                exchange(mt,item.mt);
                exchange(seed, item.seed);
            }
    
            friend void serialize(
                const rand_kernel_1& item, 
                std::ostream& out
            );

            friend void deserialize(
                rand_kernel_1& item, 
                std::istream& in 
            );

        private:
            mt19937 mt;

            std::string seed;
    };


    inline void swap (
        rand_kernel_1& a, 
        rand_kernel_1& b 
    ) { a.swap(b); }   


    template <>
    struct is_rand<rand_kernel_1>
    {
        static const bool value = true; 
    };

    inline void serialize(
        const rand_kernel_1& item, 
        std::ostream& out
    )
    {
        serialize(item.mt, out);
        serialize(item.seed, out);
    }

    inline void deserialize(
        rand_kernel_1& item, 
        std::istream& in 
    )
    {
        deserialize(item.mt, in);
        deserialize(item.seed, in);
    }
}

#endif // DLIB_RAND_KERNEl_1_


