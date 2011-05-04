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


    class rand
    {

        /*!       
            INITIAL VALUE
                - seed == ""

            CONVENTION
                - the random numbers come from the boost mersenne_twister code
                - get_seed() == seed
        !*/
        
        public:

            // These typedefs are here for backwards compatibility with older versions of dlib.
            typedef rand kernel_1a;
            typedef rand float_1a;

            rand(
            ) 
            {
                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();

                max_val =  0xFFFFFF;
                max_val *= 0x1000000;
                max_val += 0xFFFFFF;
                max_val += 0.01;
            }

            virtual ~rand(
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

            double get_random_double (
            )
            {
                uint32 temp;

                temp = rand::get_random_32bit_number();
                temp &= 0xFFFFFF;

                double val = static_cast<double>(temp);

                val *= 0x1000000;

                temp = rand::get_random_32bit_number();
                temp &= 0xFFFFFF;

                val += temp;

                val /= max_val;

                if (val < 1.0)
                {
                    return val;
                }
                else
                {
                    // return a value slightly less than 1.0
                    return 1.0 - std::numeric_limits<double>::epsilon();
                }
            }

            float get_random_float (
            )
            {
                uint32 temp;

                temp = rand::get_random_32bit_number();
                temp &= 0xFFFFFF;

                const float scale = 1.0/0x1000000;

                const float val = static_cast<float>(temp)*scale;
                if (val < 1.0f)
                {
                    return val;
                }
                else
                {
                    // return a value slightly less than 1.0
                    return 1.0f - std::numeric_limits<float>::epsilon();
                }
            }


            void swap (
                rand& item
            )
            {
                exchange(mt,item.mt);
                exchange(seed, item.seed);
            }
    
            friend void serialize(
                const rand& item, 
                std::ostream& out
            );

            friend void deserialize(
                rand& item, 
                std::istream& in 
            );

        private:
            mt19937 mt;

            std::string seed;


            double max_val;
    };


    inline void swap (
        rand& a, 
        rand& b 
    ) { a.swap(b); }   


    template <>
    struct is_rand<rand>
    {
        static const bool value = true; 
    };

    inline void serialize(
        const rand& item, 
        std::ostream& out
    )
    {
        serialize(item.mt, out);
        serialize(item.seed, out);
    }

    inline void deserialize(
        rand& item, 
        std::istream& in 
    )
    {
        deserialize(item.mt, in);
        deserialize(item.seed, in);
    }
}

#endif // DLIB_RAND_KERNEl_1_


