// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RDRAND_KERNEl_ABSTRACT_
#ifdef DLIB_RDRAND_KERNEl_ABSTRACT_

#include "../uintn.h"

namespace dlib
{

    class rdrand
    {

        /*!
            INITIAL VALUE
                get_max_retries() == 10


            WHAT THIS OBJECT REPRESENTS
                This object represents a digital random number generator.
        !*/
        
        public:

            rdrand(
            );
            /*!
                ensures 
                    - #*this is properly initialized
                throws
                    - std::bad_alloc
            !*/

            rdrand (
                uint16 max_retries_value
            );
            /*!
                ensures 
                    - #*this is properly initialized
                    - #get_max_retries() == max_retries_value
                    - This version of the constructor is equivalent to using
                      the default constructor and then calling set_max_retries(max_retries_value)
                throws
                    - std::bad_alloc
            !*/

            virtual ~rdrand(
            ); 
            /*!
                ensures
                    - all memory associated with *this has been released
            !*/

            void clear(
            );
            /*!
                ensures
                    - #*this has its initial value
                throws
                    - std::bad_alloc
                        if this exception is thrown then *this is unusable 
                        until clear() is called and succeeds
            !*/

            const uint16& get_max_retries (
            );
            /*!
                ensures
                    - returns the max_retries value currently being used to get a digital random number
            !*/

            void set_max_retries (
                uint16 max_retries_value
            );
            /*!
                ensures
                    - #get_max_retries() == max_retries_value
            !*/

            unsigned char get_random_8bit_number (
            );
            /*!
                ensures
                    - returns a digital random number in the range 0 to 255
            !*/

            uint16 get_random_16bit_number (
            );
            /*!
                ensures
                    - returns a digital random number in the range 0 to 2^16-1
            !*/

            uint32 get_random_32bit_number (
            );
            /*!
                ensures
                    - returns a digital random number in the range 0 to 2^32-1
            !*/

            uint64 get_random_64bit_number (
            );
            /*!
                ensures
                    - returns a digital random number in the range 0 to 2^64-1
            !*/

            void swap (
                rdrand& item
            );
            /*!
                ensures
                    - swaps *this and item
            !*/ 

    };

    inline void swap (
        rdrand& a,
        rdrand& b
    ) { a.swap(b); }
    /*!
        provides a global swap function
    !*/

    void serialize (
        const rdrand& item,
        std::ostream& out
    );   
    /*!
        provides serialization support
    !*/

    void deserialize (
        rdrand& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support
    !*/
}

#endif // DLIB_RDRAND_KERNEl_ABSTRACT_

