// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_UTILITIES_H_
#define DLIB_DNn_UTILITIES_H_

#include "../cuda/tensor.h"
#include "utilities_abstract.h"
#include "../geometry.h"
#include <fstream>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    )
    {
        for (auto& val : params)
        {
            // Draw a random number to initialize the layer according to formula (16)
            // from Understanding the difficulty of training deep feedforward neural
            // networks by Xavier Glorot and Yoshua Bengio.
            val = 2*rnd.get_random_float()-1;
            val *= std::sqrt(6.0/(num_inputs_and_outputs));
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        label_type label{};
        float weight = 1.f;
    };

// ----------------------------------------------------------------------------------------

    inline double log1pexp(double x)
    {
        using std::exp;
        using namespace std; // Do this instead of using std::log1p because some compilers
                             // error out otherwise (E.g. gcc 4.9 in cygwin)
        if (x <= -37)
            return exp(x);
        else if (-37 < x && x <= 18)
            return log1p(exp(x));
        else if (18 < x && x <= 33.3)
            return x + exp(-x);
        else
            return x;
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    T safe_log(T input, T epsilon = 1e-10)
    {
        // Prevent trying to calculate the logarithm of a very small number (let alone zero)
        return std::log(std::max(input, epsilon));
    }

// ----------------------------------------------------------------------------------------

    static size_t tensor_index(
        const tensor& t,
        const long sample,
        const long k,
        const long r,
        const long c
    )
    {
        return ((sample * t.k() + k) * t.nr() + r) * t.nc() + c;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_H_ 



