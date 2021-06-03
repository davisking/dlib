// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RuNNING_GRADIENT_Hh_
#define DLIB_RuNNING_GRADIENT_Hh_

#include "running_gradient_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include <cmath>
#include "../matrix.h"
#include <algorithm>


namespace dlib
{
    class running_gradient 
    {
    public:

        running_gradient (
        )
        {
            clear();
        }

        void clear(
        )
        {
            n = 0;
            R = identity_matrix<double>(2)*1e6;
            w = 0;
            residual_squared = 0;
        }

        double current_n (
        ) const
        {
            return n;
        }

        void add(
            double y
        )
        {
            matrix<double,2,1> x;
            x = n, 1;

            // Do recursive least squares computations
            const double temp = 1 + trans(x)*R*x;
            matrix<double,2,1> tmp = R*x;
            R = R - (tmp*trans(tmp))/temp;
            // R should always be symmetric.  This line improves numeric stability of this algorithm.
            R = 0.5*(R + trans(R));
            w = w + R*x*(y - trans(x)*w);

            // Also, recursively keep track of the residual error between the given value
            // and what our linear predictor outputs.
            residual_squared = residual_squared + std::pow((y - trans(x)*w),2.0)*temp;

            ++n;
        }

        double gradient (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 1,
                "\t double running_gradient::gradient()"
                << "\n\t You must add more values into this object before calling this function."
                << "\n\t this: " << this
                );

            return w(0);
        }

        double intercept (
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 0,
                "\t double running_gradient::intercept()"
                << "\n\t You must add more values into this object before calling this function."
                << "\n\t this: " << this
                );

            return w(1);
        }
        double standard_error ( 
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 2,
                "\t double running_gradient::standard_error()"
                << "\n\t You must add more values into this object before calling this function."
                << "\n\t this: " << this
                );


            const double s = residual_squared/(n-2);
            const double adjust = 12.0/(std::pow(current_n(),3.0) - current_n());
            return std::sqrt(s*adjust);
        }

        double probability_gradient_less_than (
            double thresh
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 2,
                "\t double running_gradient::probability_gradient_less_than()"
                << "\n\t You must add more values into this object before calling this function."
                << "\n\t this: " << this
                );

            return normal_cdf(thresh, gradient(), standard_error());
        }

        double probability_gradient_greater_than (
            double thresh
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(current_n() > 2,
                "\t double running_gradient::probability_gradient_greater_than()"
                << "\n\t You must add more values into this object before calling this function."
                << "\n\t this: " << this
                );

            return 1-probability_gradient_less_than(thresh);
        }

        friend void serialize (const running_gradient& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.n, out);
            serialize(item.R, out);
            serialize(item.w, out);
            serialize(item.residual_squared, out);
        }

        friend void deserialize (running_gradient& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::running_gradient.");
            deserialize(item.n, in);
            deserialize(item.R, in);
            deserialize(item.w, in);
            deserialize(item.residual_squared, in);
        }

    private:

        static double normal_cdf(double value, double mean, double stddev) 
        {
            if (stddev == 0)
            {
                if (value < mean)
                    return 0;
                else if (value > mean)
                    return 1;
                else
                    return 0.5;
            }
            value = (value-mean)/stddev;
            return 0.5 * std::erfc(-value / std::sqrt(2.0));
        }

        double n;
        matrix<double,2,2> R;
        matrix<double,2,1> w;
        double residual_squared;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    double probability_gradient_less_than (
        const T& container,
        double thresh
    )
    {
        running_gradient g;
        for(auto&& v : container)
            g.add(v);

        // make sure requires clause is not broken
        DLIB_ASSERT(g.current_n() > 2,
            "\t double probability_gradient_less_than()"
            << "\n\t You need more than 2 elements in the given container to call this function."
        );
        return g.probability_gradient_less_than(thresh);
    }

    template <
        typename T
        >
    double probability_gradient_greater_than (
        const T& container,
        double thresh
    )
    {
        running_gradient g;
        for(auto&& v : container)
            g.add(v);

        // make sure requires clause is not broken
        DLIB_ASSERT(g.current_n() > 2,
            "\t double probability_gradient_greater_than()"
            << "\n\t You need more than 2 elements in the given container to call this function."
        );
        return g.probability_gradient_greater_than(thresh);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    double find_upper_quantile (
        const T& container_,
        double quantile
    )
    {
        DLIB_CASSERT(0 <= quantile && quantile <= 1.0);

        // copy container into a std::vector
        std::vector<double> container(container_.begin(), container_.end());

        DLIB_CASSERT(container.size() > 0);

        size_t idx_upper = std::round((container.size()-1)*(1-quantile));

        std::nth_element(container.begin(), container.begin()+idx_upper, container.end());
        auto upper_q = *(container.begin()+idx_upper);
        return upper_q;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    double probability_values_are_increasing (
        const T& container
    )
    {
        running_gradient g;
        for (auto x : container)
        {
            g.add(x);
        }
        if (g.current_n() > 2)
            return g.probability_gradient_greater_than(0);
        else
            return 0.5;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    double probability_values_are_increasing_robust (
        const T& container,
        double quantile_discard = 0.10
    )
    {
        const auto quantile_thresh = find_upper_quantile(container, quantile_discard); 
        running_gradient g;
        for (auto x : container)
        {
            // Ignore values that are too large.
            if (x <= quantile_thresh)
                g.add(x);
        }
        if (g.current_n() > 2)
            return g.probability_gradient_greater_than(0);
        else
            return 0.5;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    size_t count_steps_without_decrease (
        const T& container,
        double probability_of_decrease = 0.51
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0.5 < probability_of_decrease && probability_of_decrease < 1,
            "\t size_t count_steps_without_decrease()"
            << "\n\t probability_of_decrease: "<< probability_of_decrease 
        );

        running_gradient g;
        size_t count = 0;
        size_t j = 0;
        for (auto i = container.rbegin(); i != container.rend(); ++i)
        {
            ++j;
            g.add(*i);
            if (g.current_n() > 2)
            {
                // Note that this only looks backwards because we are looping over the
                // container backwards.  So here we are really checking if the gradient isn't
                // decreasing.
                double prob_decreasing = g.probability_gradient_greater_than(0);
                // If we aren't confident things are decreasing.
                if (prob_decreasing < probability_of_decrease)
                    count = j;
            }
        }
        return count;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    size_t count_steps_without_decrease_robust (
        const T& container,
        double probability_of_decrease = 0.51,
        double quantile_discard = 0.10
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0 <= quantile_discard && quantile_discard <= 1);
        DLIB_ASSERT(0.5 < probability_of_decrease && probability_of_decrease < 1,
            "\t size_t count_steps_without_decrease_robust()"
            << "\n\t probability_of_decrease: "<< probability_of_decrease 
        );

        if (container.size() == 0)
            return 0;

        const auto quantile_thresh = find_upper_quantile(container, quantile_discard); 

        running_gradient g;
        size_t count = 0;
        size_t j = 0;
        for (auto i = container.rbegin(); i != container.rend(); ++i)
        {
            ++j;
            // ignore values that are too large
            if (*i <= quantile_thresh)
                g.add(*i);

            if (g.current_n() > 2)
            {
                // Note that this only looks backwards because we are looping over the
                // container backwards.  So here we are really checking if the gradient isn't
                // decreasing.
                double prob_decreasing = g.probability_gradient_greater_than(0);
                // If we aren't confident things are decreasing.
                if (prob_decreasing < probability_of_decrease)
                    count = j;
            }
        }
        return count;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    size_t count_steps_without_increase (
        const T& container,
        double probability_of_increase = 0.51
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(0.5 < probability_of_increase && probability_of_increase < 1,
            "\t size_t count_steps_without_increase()"
            << "\n\t probability_of_increase: "<< probability_of_increase 
        );

        running_gradient g;
        size_t count = 0;
        size_t j = 0;
        for (auto i = container.rbegin(); i != container.rend(); ++i)
        {
            ++j;
            g.add(*i);
            if (g.current_n() > 2)
            {
                // Note that this only looks backwards because we are looping over the
                // container backwards.  So here we are really checking if the gradient isn't
                // increasing.
                double prob_increasing = g.probability_gradient_less_than(0);
                // If we aren't confident things are increasing.
                if (prob_increasing < probability_of_increase)
                    count = j;
            }
        }
        return count;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RuNNING_GRADIENT_Hh_


