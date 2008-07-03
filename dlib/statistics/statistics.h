// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STATISTICs_
#define DLIB_STATISTICs_

#include "statistics_abstract.h"
#include <limits>
#include <cmath>
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    class running_stats
    {
    public:

        running_stats()
        {
            clear();

            COMPILE_TIME_ASSERT ((
                    is_same_type<float,T>::value ||
                    is_same_type<double,T>::value ||
                    is_same_type<long double,T>::value 
            ));
        }

        void clear()
        {
            sum = 0;
            sum_sqr = 0;
            n = 0;
            maximum_n = std::numeric_limits<T>::max();
            min_value = std::numeric_limits<T>::infinity();
            max_value = -std::numeric_limits<T>::infinity();
        }

        void set_max_n (
            const T& val
        )
        {
            maximum_n = val;
        }

        void add (
            const T& val
        )
        {
            const T div_n   = 1/(n+1);
            const T n_div_n = n*div_n;

            sum     = n_div_n*sum     + val*div_n;
            sum_sqr = n_div_n*sum_sqr + val*div_n*val;

            if (val < min_value)
                min_value = val;
            if (val > max_value)
                max_value = val;

            if (n < maximum_n)
                ++n;
        }

        T max_n (
        ) const
        {
            return max_n;
        }

        T current_n (
        ) const
        {
            return n;
        }

        T mean (
        ) const
        {
            return sum;
        }

        T max (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 1,
                "\tT running_stats::max"
                << "\n\tyou have to add some numbers to this object first"
                << "\n\tthis: " << this
                );

            return max_value;
        }

        T min (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 1,
                "\tT running_stats::min"
                << "\n\tyou have to add some numbers to this object first"
                << "\n\tthis: " << this
                );

            return min_value;
        }

        T variance (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 1,
                "\tT running_stats::variance"
                << "\n\tyou have to add some numbers to this object first"
                << "\n\tthis: " << this
                );

            T temp = n/(n-1);
            temp = temp*(sum_sqr - sum*sum);
            // make sure the variance is never negative.  This might
            // happen due to numerical errors.
            if (temp >= 0)
                return temp;
            else
                return 0;
        }

        T scale (
            const T& val
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 1,
                "\tT running_stats::variance"
                << "\n\tyou have to add some numbers to this object first"
                << "\n\tthis: " << this
                );
            return (val-mean())/std::sqrt(variance());
        }

    private:
        T sum;
        T sum_sqr;
        T n;
        T maximum_n;
        T min_value;
        T max_value;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STATISTICs_

