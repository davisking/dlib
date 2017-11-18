// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UPPER_bOUND_FUNCTION_Hh_
#define DLIB_UPPER_bOUND_FUNCTION_Hh_

#include "upper_bound_function_abstract.h"
#include "../svm/svm_c_linear_dcd_trainer.h"
#include "../statistics.h"
#include <limits>
#include <utility>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct function_evaluation
    {
        function_evaluation() = default;
        function_evaluation(const matrix<double,0,1>& x, double y) :x(x), y(y) {}

        matrix<double,0,1> x;
        double y = std::numeric_limits<double>::quiet_NaN();
    };

// ----------------------------------------------------------------------------------------

    class upper_bound_function
    {

    public:

        upper_bound_function(
        ) = default;

        explicit upper_bound_function(
            const std::vector<function_evaluation>& _points,
            const double relative_noise_magnitude = 0.001,
            const double solver_eps = 0.0001
        ) : points(_points)
        {
            DLIB_CASSERT(points.size() > 1);
            DLIB_CASSERT(points[0].x.size() > 0, "The vectors can't be empty.");
            DLIB_CASSERT(relative_noise_magnitude >= 0);
            DLIB_CASSERT(solver_eps > 0);

            const long dims = points[0].x.size();
            for (auto& p : points)
                DLIB_CASSERT(p.x.size() == dims, "All the vectors given to upper_bound_function must have the same dimensionality.");


            using sample_type = std::vector<std::pair<size_t,double>>;
            using kernel_type = sparse_linear_kernel<sample_type>;
            std::vector<sample_type> x;
            std::vector<double> y;

            // We are going to normalize the data so the values aren't extreme.  First, we
            // collect statistics on our data.
            std::vector<running_stats<double>> x_rs(dims);
            running_stats<double> y_rs;
            for (auto& v : points)
            {
                for (long i = 0; i < v.x.size(); ++i)
                    x_rs[i].add(v.x(i));
                y_rs.add(v.y);
            }

            x.reserve(points.size()*(points.size()-1)/2);
            y.reserve(points.size()*(points.size()-1)/2);

            // compute normalization vectors for the data.  The only reason we do this is
            // to make the optimization well conditioned.  In particular, scaling the y
            // values will prevent numerical errors in the 1-diff*diff computation below that
            // would otherwise result when diff is really big.  Also, scaling the xvalues
            // to be about 1 will similarly make the optimization more stable and it also
            // has the added benefit of keeping the relative_noise_magnitude's scale
            // constant regardless of the size of x values.
            const double yscale = 1.0/y_rs.stddev();
            std::vector<double> xscale(dims);
            for (size_t i = 0; i < xscale.size(); ++i)
                xscale[i] = 1.0/(x_rs[i].stddev()*yscale); // make it so that xscale[i]*yscale ==  1/x_rs[i].stddev()


            sample_type samp;
            for (size_t i = 0; i < points.size(); ++i)
            {
                for (size_t j = i+1; j < points.size(); ++j)
                {
                    samp.clear();
                    for (long k = 0; k < dims; ++k)
                    {
                        double temp = (points[i].x(k) - points[j].x(k))*xscale[k]*yscale;
                        samp.push_back(std::make_pair(k, temp*temp));
                    }

                    if (points[i].y > points[j].y)
                        samp.push_back(std::make_pair(dims + j, relative_noise_magnitude));
                    else
                        samp.push_back(std::make_pair(dims + i, relative_noise_magnitude));

                    const double diff = (points[i].y - points[j].y)*yscale;
                    samp.push_back(std::make_pair(dims + points.size(), 1-diff*diff));

                    x.push_back(samp);
                    y.push_back(1);
                }
            }

            svm_c_linear_dcd_trainer<kernel_type> trainer;
            trainer.set_c(std::numeric_limits<double>::infinity());
            //trainer.be_verbose();
            trainer.force_last_weight_to_1(true);
            trainer.set_epsilon(solver_eps);

            auto df = trainer.train(x,y);


            const auto& bv = df.basis_vectors(0);
            slopes.set_size(dims);
            for (long i = 0; i < dims; ++i)
                slopes(i) = bv[i].second*xscale[i]*xscale[i];

            //cout << "slopes:" << trans(slopes);

            offsets.resize(points.size());


            auto s = x.begin();
            for (size_t i = 0; i < points.size(); ++i)
            {
                for (size_t j = i+1; j < points.size(); ++j)
                {
                    double val = df(*s);
                    // If the constraint wasn't exactly satisfied then we need to adjust
                    // the offsets so that it is satisfied.  So we check for that here
                    if (points[i].y > points[j].y)
                    {
                        if (val + offsets[j] < 1)
                            offsets[j] = 1-val;
                    }
                    else
                    {
                        if (val + offsets[i] < 1)
                            offsets[i] = 1-val;
                    }
                    ++s;
                }
            }

            for (size_t i = 0; i < points.size(); ++i)
            {
                offsets[i] += bv[slopes.size()+i].second*relative_noise_magnitude;
            }
        }

        long num_points(
        ) const 
        { 
            return points.size(); 
        }

        long dimensionality(
        ) const
        { 
            if (points.size() == 0)
                return 0;
            else
                return points[0].x.size();
        }

        double operator() (
            matrix<double,0,1> x
        ) const
        {
            DLIB_CASSERT(num_points() > 0);
            DLIB_CASSERT(x.size() == dimensionality());



            double upper_bound = std::numeric_limits<double>::infinity();

            for (size_t i = 0; i < points.size(); ++i)
            {
                const double local_bound = points[i].y + std::sqrt(offsets[i] + dot(slopes, squared(x-points[i].x)));
                upper_bound = std::min(upper_bound, local_bound);
            }

            return upper_bound;
        }

    private:

        std::vector<function_evaluation> points;
        std::vector<double> offsets; // offsets.size() == points.size()
        matrix<double,0,1> slopes; // slopes.size() == points[0].first.size()
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UPPER_bOUND_FUNCTION_Hh_


