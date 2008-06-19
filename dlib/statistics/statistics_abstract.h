// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_STATISTICs_ABSTRACT_
#ifdef DLIB_STATISTICs_ABSTRACT_

#include <limits>
#include <cmath>

namespace dlib
{

    template <
        typename T
        >
    class running_stats
    {
        /*!
            REQUIREMENTS ON T
                - T must be a float, double, or long double type

            INITIAL VALUE
                - max_n() == std::numeric_limits<T>::max()
                - mean() == 0
                - current_n() == 0

            WHAT THIS OBJECT REPRESENTS
                This object represents something that can compute the running mean and
                variance of a stream of real numbers.  

                As this object accumulates more and more numbers it will be the case
                that each new number impacts the current mean and variance estimate
                less and less.  This may be what you want.  But it might not be. 

                For example, your stream of numbers might be non-stationary, that is,
                the mean and variance might change over time.  To enable you to use
                this object on such a stream of numbers this object provides the 
                ability to set a "max_n."  The meaning of the max_n() parameter
                is that after max_n() samples have been seen each new sample will
                have the same impact on the mean and variance estimates from then on.

                So if you have a highly non-stationary stream of data you might
                set the max_n to a small value while if you have a very stationary
                stream you might set it to a very large value.
        !*/
    public:

        running_stats(
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        void clear(
        );
        /*!
            ensures
                - this object has its initial value
                - clears all memory of any previous data points
        !*/

        void set_max_n (
            const T& val
        );
        /*!
            ensures
                - #max_n() == val
        !*/

        T max_n (
        ) const;
        /*!
            ensures
                - returns the max value that current_n() is allowed to take on
        !*/

        T current_n (
        ) const;
        /*!
            ensures
                - returns the number of points given to this object so far or
                  max_n(), whichever is smallest.
        !*/

        void add (
            const T& val
        );
        /*!
            ensures
                - updates the mean and variance stored in this object so that
                  the new value is factored into them
                - #mean() == mean()*current_n()/(current_n()+1) + val/(current_n()+1)
                - #variance() == the updated variance that takes this new value into account
                - if (current_n() < max_n()) then
                    - #current_n() == current_n() + 1
                - else
                    - #current_n() == current_n()
        !*/

        T mean (
        ) const;
        /*!
            ensures
                - returns the mean of all the values presented to this object 
                  so far.
        !*/

        T variance (
        ) const;
        /*!
            requires
                - current_n() > 1
            ensures
                - returns the variance of all the values presented to this
                  object so far.
        !*/

        T scale (
            const T& val
        ) const;
        /*!
            requires
                - current_n() > 1
            ensures
                - return (val-mean())/std::sqrt(variance());
        !*/
    };

}

#endif // DLIB_STATISTICs_ABSTRACT_

