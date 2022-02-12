// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RAND_KERNEl_1_
#define DLIB_RAND_KERNEl_1_

#include <string>
#include <complex>
#include "../algs.h"
#include "rand_kernel_abstract.h"
#include "mersenne_twister.h"
#include "../is_kind.h"
#include <iostream>
#include "../serialize.h"
#include "../string.h"

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
                init();
            }

            rand (
                time_t seed_value
            )
            {
                init();
                set_seed(cast_to_string(seed_value));
            }

            rand (
                const std::string& seed_value
            )
            {
                init();
                set_seed(seed_value);
            }

            virtual ~rand(
            )
            {}

            void clear(
            )
            {
                mt.seed();
                seed.clear();

                has_gaussian = false;
                next_gaussian = 0;

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


                has_gaussian = false;
                next_gaussian = 0;
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

            inline uint64 get_random_64bit_number (
            )
            {
                const uint64 a = get_random_32bit_number();
                const uint64 b = get_random_32bit_number();
                return (a<<32)|b;
            }

            double get_double_in_range (
                double begin,
                double end
            )
            {
                DLIB_ASSERT(begin <= end);
                return begin + get_random_double()*(end-begin);
            }

            long long get_integer_in_range(
                long long begin,
                long long end
            )
            {
                DLIB_ASSERT(begin <= end);
                if (begin == end)
                    return begin;

                auto r = get_random_64bit_number();
                const auto limit = std::numeric_limits<decltype(r)>::max();
                const auto range = end-begin;
                // Use rejection sampling to remove the biased sampling you would get with
                // the naive get_random_64bit_number()%range sampling. 
                while(r >= (limit/range)*range)
                    r = get_random_64bit_number();

                return begin + static_cast<long long>(r%range);
            }

            long long get_integer(
                long long end
            )
            {
                DLIB_ASSERT(end >= 0);

                return get_integer_in_range(0,end);
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
  
            std::complex<double> get_random_complex_gaussian (
            )
            {
                double x1, x2, w;

                const double rndmax = std::numeric_limits<dlib::uint32>::max();

                // Generate a pair of Gaussian random numbers using the Box-Muller transformation.
                do 
                {
                    const double rnd1 = get_random_32bit_number()/rndmax;
                    const double rnd2 = get_random_32bit_number()/rndmax;

                    x1 = 2.0 * rnd1 - 1.0;
                    x2 = 2.0 * rnd2 - 1.0;
                    w = x1 * x1 + x2 * x2;
                } while ( w >= 1.0 );

                w = std::sqrt( (-2.0 * std::log( w ) ) / w );
                return std::complex<double>(x1 * w, x2 * w);
            }

            double get_random_gaussian (
            )
            {
                if (has_gaussian)
                {
                    has_gaussian = false;
                    return next_gaussian;
                }
                
                std::complex<double> r = get_random_complex_gaussian();
                next_gaussian = r.imag();
                has_gaussian = true;
                return r.real();
            }

            double get_random_exponential (
                double lambda
            )
            {
                DLIB_ASSERT(lambda > 0, "lambda must be greater than zero");
                double u = 0.0;
                while (u == 0.0)
                    u = get_random_double();
                return -std::log( u ) / lambda;
            }

            double get_random_weibull (
                double lambda,
                double k,
                double gamma
            )
            {
                DLIB_ASSERT(k > 0, "k must be greater than zero");
                DLIB_ASSERT(lambda > 0, "lambda must be greater than zero");
                double u = 0.0;
                while (u == 0.0)
                    u = get_random_double();
                return gamma + lambda*std::pow(-std::log(u), 1.0 / k);
            }

            double get_random_beta (
                double alpha,
                double beta
            )
            {
                DLIB_CASSERT(alpha > 0, "alpha must be greater than zero")
                DLIB_CASSERT(beta > 0, "beta must be greater than zero");
                auto u = std::pow(get_random_double(), 1 / alpha);
                auto v = std::pow(get_random_double(), 1 / beta);
                while ((u + v) > 1 || (u == 0 && v == 0))
                {
                    u = std::pow(get_random_double(), 1 / alpha);
                    v = std::pow(get_random_double(), 1 / beta);
                }
                return u / (u + v);
            }

            void swap (
                rand& item
            )
            {
                exchange(mt,item.mt);
                exchange(seed, item.seed);
                exchange(has_gaussian, item.has_gaussian);
                exchange(next_gaussian, item.next_gaussian);
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

            void init()
            {
                // prime the generator a bit
                for (int i = 0; i < 10000; ++i)
                    mt();

                max_val =  0xFFFFFF;
                max_val *= 0x1000000;
                max_val += 0xFFFFFF;
                max_val += 0.05;


                has_gaussian = false;
                next_gaussian = 0;
            }

            mt19937 mt;

            std::string seed;


            double max_val;
            bool has_gaussian;
            double next_gaussian;
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
        int version = 1;
        serialize(version, out);

        serialize(item.mt, out);
        serialize(item.seed, out);
        serialize(item.has_gaussian, out);
        serialize(item.next_gaussian, out);
    }

    inline void deserialize(
        rand& item, 
        std::istream& in 
    )
    {
        int version;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Error deserializing object of type rand: unexpected version."); 

        deserialize(item.mt, in);
        deserialize(item.seed, in);
        deserialize(item.has_gaussian, in);
        deserialize(item.next_gaussian, in);
    }
}

#endif // DLIB_RAND_KERNEl_1_


