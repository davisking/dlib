// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RuNNING_GRADIENT_ABSTRACT_Hh_
#ifdef DLIB_RuNNING_GRADIENT_ABSTRACT_Hh_


namespace dlib
{
    class running_gradient 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for estimating if a noisy sequence of numbers is
                trending up or down and by how much.  It does this by finding the least
                squares fit of a line to the data and then allows you to perform a
                statistical test on the slope of that line.
        !*/

    public:

        running_gradient (
        );
        /*!
            ensures
                - #current_n() == 0
        !*/

        void clear(
        );
        /*!
            ensures
                - #current_n() == 0
                - this object has its initial value
                - clears all memory of any previous data points
        !*/

        double current_n (
        ) const;
        /*!
            ensures
                - returns the number of values given to this object by add(). 
        !*/

        void add(
            double y
        );
        /*!
            ensures
                - Updates the gradient() and standard_error() estimates in this object
                  based on the new y value.
                - #current_n() == current_n() + 1
        !*/

        double gradient (
        ) const;
        /*!
            requires
                - current_n() > 1
            ensures
                - If we consider the values given to add() as time series data, we can
                  estimate the rate-of-change of those values.  That is, how much,
                  typically, do those values change from sample to sample?  The gradient()
                  function returns the current estimate.  It does this by finding the least
                  squares fit of a line to the data given to add() and returning the slope
                  of this line.
        !*/

        double standard_error ( 
        ) const;
        /*!
            requires
                - current_n() > 2
            ensures
                - returns the standard deviation of the estimate of gradient(). 
        !*/

        double probability_gradient_less_than (
            double thresh
        ) const;
        /*!
            requires
                - current_n() > 2
            ensures
                - If we can assume the values given to add() are linearly related to each
                  other and corrupted by Gaussian additive noise then our estimate of
                  gradient() is a random variable with a mean value of gradient() and a
                  standard deviation of standard_error().  This lets us compute the
                  probability that the true gradient of the data is less than thresh, which
                  is what this function returns.
        !*/

        double probability_gradient_greater_than (
            double thresh
        ) const;
        /*!
            requires
                - current_n() > 2
            ensures
                - returns 1-probability_gradient_less_than(thresh)
        !*/

    };

    void serialize (
        const running_gradient& item, 
        std::ostream& out 
    );
    /*!
        provides serialization support 
    !*/

    void deserialize (
        running_gradient& item, 
        std::istream& in
    );
    /*!
        provides serialization support 
    !*/
}

#endif // DLIB_RuNNING_GRADIENT_ABSTRACT_Hh_


