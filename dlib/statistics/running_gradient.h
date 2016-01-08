// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RuNNING_GRADIENT_Hh_
#define DLIB_RuNNING_GRADIENT_Hh_

#include "running_gradient_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include <cmath>
#include "../matrix.h"


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

            return normal_cfd(thresh, gradient(), standard_error());
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

        static double normal_cfd(double value, double mean, double stddev) 
        {
            value = (value-mean)/stddev;
            return 0.5 * erfc(-value / std::sqrt(2.0));
        }

        double n;
        matrix<double,2,2> R;
        matrix<double,2,1> w;
        double residual_squared;
    };
}

#endif // DLIB_RuNNING_GRADIENT_Hh_


