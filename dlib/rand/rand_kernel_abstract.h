// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RAND_KERNEl_ABSTRACT_
#ifdef DLIB_RAND_KERNEl_ABSTRACT_

#include <string>
#include <complex>
#include "../uintn.h"

namespace dlib
{


    class rand
    {

        /*!      
            INITIAL VALUE
                get_seed() == ""


            WHAT THIS OBJECT REPRESENTS
                This object represents a pseudorandom number generator.
        !*/
        
        public:


            rand(
            );
            /*!
                ensures 
                    - #*this is properly initialized
                throws
                    - std::bad_alloc
            !*/

            rand (
                time_t seed_value
            );
            /*!
                ensures 
                    - #*this is properly initialized
                    - #get_seed() == cast_to_string(seed_value) 
                    - This version of the constructor is equivalent to using
                      the default constructor and then calling set_seed(cast_to_string(seed_value))
                throws
                    - std::bad_alloc
            !*/

            rand (
                const std::string& seed_value
            );
            /*!
                ensures 
                    - #*this is properly initialized
                    - #get_seed() == seed_value
                    - This version of the constructor is equivalent to using
                      the default constructor and then calling set_seed(seed_value)
                throws
                    - std::bad_alloc
            !*/

            virtual ~rand(
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

            const std::string& get_seed (
            );
            /*!
                ensures
                    - returns the string currently being used as the random seed.
            !*/

            void set_seed (
                const std::string& value
            );
            /*!
                ensures
                    - #get_seed() == value
            !*/

            unsigned char get_random_8bit_number (
            );
            /*!
                ensures
                    - returns a pseudorandom number in the range 0 to 255
            !*/

            uint16 get_random_16bit_number (
            );
            /*!
                ensures
                    - returns a pseudorandom number in the range 0 to 2^16-1 
            !*/

            uint32 get_random_32bit_number (
            );
            /*!
                ensures
                    - returns a pseudorandom number in the range 0 to 2^32-1 
            !*/

            uint64 get_random_64bit_number (
            );
            /*!
                ensures
                    - returns a pseudorandom number in the range 0 to 2^64-1 
            !*/

            float get_random_float (
            );
            /*!
                ensures
                    - returns a random float number N where:  0.0 <= N < 1.0.
            !*/

            double get_random_double (
            );
            /*!
                ensures
                    - returns a random double number N where:  0.0 <= N < 1.0.
            !*/

            double get_double_in_range (
                double begin,
                double end
            );
            /*!
                requires
                    - begin <= end
                ensures
                    - if (begin < end) then
                        - returns a random double number N where:  begin <= N < end.
                    - else
                        - returns begin
            !*/

            long long get_integer_in_range(
                long long begin,
                long long end
            );
            /*!
                requires
                    - begin <= end
                ensures
                    - returns a random integer N selected from the range: begin <= N < end
                      The integer is selected uniformly at random.  If begin==end then
                      begin is returned.
            !*/

            long long get_integer(
                long long end
            );
            /*!
                requires
                    - 0 <= end
                ensures
                    - returns get_integer_in_range(0,end)
            !*/

            double get_random_gaussian (
            );
            /*!
                ensures
                    - returns a random number sampled from a Gaussian distribution 
                      with mean 0 and standard deviation 1. 
            !*/

            std::complex<double> get_random_complex_gaussian (
            );
            /*!
                ensures
                    - returns a random complex number sampled from a Gaussian distribution 
                      with mean 0 and standard deviation 1. 
            !*/
            
            double get_random_exponential (
                double lambda
            );
            /*!
                ensures
                    - returns a random number sampled from an exponential distribution
                      with rate parameter lambda
            !*/

            double get_random_weibull (
                double lambda,
                double k,
                double gamma
            );
            /*!
                ensures
                    - returns a random number sampled from a Weibull distribution
                      with shape parameter k, scale parameter lambda and 
                      threshold parameter gamma.
            !*/
            
            void swap (
                rand& item
            );
            /*!
                ensures
                    - swaps *this and item
            !*/ 

    };

    inline void swap (
        rand& a, 
        rand& b 
    ) { a.swap(b); }   
    /*!
        provides a global swap function
    !*/

    void serialize (
        const rand& item, 
        std::ostream& out 
    );   
    /*!
        provides serialization support 
    !*/

    void deserialize (
        rand& item, 
        std::istream& in
    );   
    /*!
        provides deserialization support 
    !*/
}

#endif // DLIB_RAND_KERNEl_ABSTRACT_

