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

        double intercept (
        ) const;
        /*!
            requires
                - current_n() > 0
            ensures
                - This class fits a line to the time series data given to add().  This
                  function returns the intercept of that line while gradient() returns the
                  slope of that line.  This means that, for example, the next point that
                  add() will see, as predicted by this best fit line, is the value
                  intercept() + current_n()*gradient().
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

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    double probability_gradient_less_than (
        const T& container,
        double thresh
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with a
              range based for loop.
            - The container must contain more than 2 elements.
        ensures
            - Puts all the elements of container into a running_gradient object, R, and
              then returns R.probability_gradient_less_than(thresh). 
    !*/

    template <
        typename T
        >
    double probability_gradient_greater_than (
        const T& container,
        double thresh
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with a
              range based for loop.
            - The container must contain more than 2 elements.
        ensures
            - Puts all the elements of container into a running_gradient object, R, and
              then returns R.probability_gradient_greater_than(thresh).
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    size_t count_steps_without_decrease (
        const T& container,
        double probability_of_decrease = 0.51
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with
              .rbegin() and .rend().
            - 0.5 < probability_of_decrease < 1
        ensures
            - If you think of the contents of container as a potentially noisy time series,
              then this function returns a count of how long the time series has gone
              without noticeably decreasing in value.  It does this by adding the
              elements into a running_gradient object and counting how many elements,
              starting with container.back(), that you need to examine before you are
              confident that the series has been decreasing in value.  Here, "confident of
              decrease" means that the probability of decrease is >= probability_of_decrease.  
            - Setting probability_of_decrease to 0.51 means we count until we see even a
              small hint of decrease, whereas a larger value of 0.99 would return a larger
              count since it keeps going until it is nearly certain the time series is
              decreasing.
            - The max possible output from this function is container.size().
    !*/

    template <
        typename T
        > 
    size_t count_steps_without_decrease_robust (
        const T& container,
        double probability_of_decrease = 0.51,
        double quantile_discard = 0.10
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with
              .begin() and .end() as well as .rbegin() and .rend().
            - 0.5 < probability_of_decrease < 1
            - 0 <= quantile_discard <= 1
        ensures
            - This function behaves just like
              count_steps_without_decrease(container,probability_of_decrease) except that
              it ignores values in container that are in the upper quantile_discard
              quantile.  So for example, if the quantile discard is 0.1 then the 10%
              largest values in container are ignored.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    size_t count_steps_without_increase (
        const T& container,
        double probability_of_increase = 0.51
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with
              .rbegin() and .rend().
            - 0.5 < probability_of_increase < 1
        ensures
            - If you think of the contents of container as a potentially noisy time series,
              then this function returns a count of how long the time series has gone
              without noticeably increasing in value.  It does this by adding the
              elements into a running_gradient object and counting how many elements,
              starting with container.back(), that you need to examine before you are
              confident that the series has been increasing in value.  Here, "confident of
              increase" means that the probability of increase is >= probability_of_increase.  
            - Setting probability_of_increase to 0.51 means we count until we see even a
              small hint of increase, whereas a larger value of 0.99 would return a larger
              count since it keeps going until it is nearly certain the time series is
              increasing.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    double find_upper_quantile (
        const T& container,
        double quantile
    );
    /*!
        requires
            - container must be a container of double values that can be enumerated with
              .begin() and .end().
            - 0 <= quantile <= 1
            - container.size() > 0
        ensures
            - Finds and returns the value such that quantile percent of the values in
              container are greater than it.  For example, 0.5 would find the median value
              in container while 0.1 would find the value that lower bounded the 10%
              largest values in container.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RuNNING_GRADIENT_ABSTRACT_Hh_


