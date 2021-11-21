// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RDRAND_KERNEl_1_
#define DLIB_RDRAND_KERNEl_1_

#include "rdrand_kernel_abstract.h"
#include "../simd/simd_check.h"

namespace dlib
{

    class rdrand
    {

        /*!
            INITIAL VALUE
                - max_retries == 10

            CONVENTION
                - the digital random numbers come from RDRND instructions
        !*/
        
        public:
        
            rdrand (
            )
            {
                init();
            }

            rdrand (
                uint16 max_retries_value
            )
            {
                init();
                set_max_retries(max_retries_value);
            }

            virtual ~rdrand (
            )
            {}

            void clear(
            )
            {
                max_retries = 10;
            }

            const uint16& get_max_retries (
            )
            {
                return max_retries;
            }

            void set_max_retries (
                uint16 max_retries_value
            )
            {
                max_retries = max_retries_value;
            }

            unsigned char get_random_8bit_number (
            )
            {
                uint16 rand = get_random_16bit_number();

                return static_cast<unsigned char>(rand);
            }

            uint16 get_random_16bit_number (
            )
            {
#ifdef DLIB_HAVE_RDRND
                uint16 rand;
                uint16 retries = max_retries;

                while (retries--)
                {
                    if (_rdrand16_step(&rand))
                        return rand;
                }

                DLIB_ASSERT(false,
                    "\t dlib::rdrand failed to generate a digital random number after "
                    << max_retries << " tries"
                    );
#endif
                return 0;
            }

            inline uint32 get_random_32bit_number (
            )
            {
#ifdef DLIB_HAVE_RDRND
                uint32 rand;
                uint16 retries = max_retries;

                while (retries--)
                {
                    if (_rdrand32_step(&rand))
                        return rand;
                }

                DLIB_ASSERT(false,
                    "\t dlib::rdrand failed to generate a digital random number after "
                    << max_retries << " tries"
                    );
#endif
                return 0;
            }

            inline uint64 get_random_64bit_number (
            )
            {
#ifdef DLIB_HAVE_RDRND
    #ifdef __x86_64__
                uint64 rand;
                uint16 retries = max_retries;

                while (retries--)
                {
                    if (_rdrand64_step(&rand))
                        return rand;
                }

                DLIB_ASSERT(false,
                    "\t dlib::rdrand failed to generate a digital random number after "
                    << max_retries << " tries"
                    );
    #else
                const uint64 a = get_random_32bit_number();
                const uint64 b = get_random_32bit_number();
                return (a<<32)|b;
    #endif
#endif
                return 0;
            }

            void swap (
                rdrand& item
            )
            {
                exchange(max_retries, item.max_retries);
            }
    
            friend void serialize(
                const rdrand& item,
                std::ostream& out
            );

            friend void deserialize(
                rdrand& item,
                std::istream& in
            );

        private:

            void init()
            {
#ifndef DLIB_HAVE_RDRND
                DLIB_ASSERT(false,
                    "\t Dlib wasn't compiled to use RDRND instructions."
                    << "\n\t If you have a CPU that supports RDRND instructions then turn them on like this:"
                    << "\n\t\t mkdir build; cd build; cmake .. -DUSE_RDRND_INSTRUCTIONS=1; cmake --build ."
                    );
#endif
                max_retries = 10;
            }

            uint16 max_retries;
    };

    inline void swap (
        rdrand& a,
        rdrand& b
    ) { a.swap(b); }

    template <>
    struct is_rand<rdrand>
    {
        static const bool value = true;
    };

    inline void serialize(
        const rdrand& item,
        std::ostream& out
    )
    {
        int version = 1;
        serialize(version, out);

        serialize(item.max_retries, out);
    }

    inline void deserialize(
        rdrand& item,
        std::istream& in
    )
    {
        int version;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Error deserializing object of type rdrand: unexpected version.");

        deserialize(item.max_retries, in);
    }
}

#endif // DLIB_RDRAND_KERNEl_1_

