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

        upper_bound_function(
            const double relative_noise_magnitude,
            const double solver_eps 
        ) : relative_noise_magnitude(relative_noise_magnitude), solver_eps(solver_eps)
        {
            DLIB_CASSERT(relative_noise_magnitude >= 0);
            DLIB_CASSERT(solver_eps > 0);
        }

        explicit upper_bound_function(
            const std::vector<function_evaluation>& _points,
            const double relative_noise_magnitude = 0.001,
            const double solver_eps = 0.0001
        ) : relative_noise_magnitude(relative_noise_magnitude), solver_eps(solver_eps), points(_points)
        {
            DLIB_CASSERT(relative_noise_magnitude >= 0);
            DLIB_CASSERT(solver_eps > 0);

            if (points.size() > 1)
            {
                DLIB_CASSERT(points[0].x.size() > 0, "The vectors can't be empty.");

                const long dims = points[0].x.size();
                for (auto& p : points)
                    DLIB_CASSERT(p.x.size() == dims, "All the vectors given to upper_bound_function must have the same dimensionality.");

                learn_params();
            }

        }

        void add (
            const function_evaluation& point
        )
        {
            DLIB_CASSERT(point.x.size() != 0, "The vectors can't be empty.");
            if (points.size() == 0)
            {
                points.push_back(point);
                return;
            }

            DLIB_CASSERT(point.x.size() == dimensionality(), "All the vectors given to upper_bound_function must have the same dimensionality.");

            if (points.size() < 4)
            {
                points.push_back(point);
                *this = upper_bound_function(points, relative_noise_magnitude, solver_eps);
                return;
            }

            points.push_back(point);
            // add constraints between the new point and the old points
            for (size_t i = 0; i < points.size()-1; ++i)
                active_constraints.push_back(std::make_pair(i,points.size()-1));

            learn_params();
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

        const std::vector<function_evaluation>& get_points(
        ) const 
        { 
            return points; 
        }

        double operator() (
            const matrix<double,0,1>& x
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

        void learn_params (
        )
        {
            const long dims = points[0].x.size();

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
            auto add_constraint = [&](long i, long j) {
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
            };

            if (active_constraints.size() == 0)
            {
                x.reserve(points.size()*(points.size()-1)/2);
                y.reserve(points.size()*(points.size()-1)/2);
                for (size_t i = 0; i < points.size(); ++i)
                {
                    for (size_t j = i+1; j < points.size(); ++j)
                    {
                        add_constraint(i,j);
                    }
                }
            }
            else
            {
                for (auto& p : active_constraints)
                    add_constraint(p.first, p.second);
            }




            svm_c_linear_dcd_trainer<kernel_type> trainer;
            trainer.set_c(std::numeric_limits<double>::infinity());
            //trainer.be_verbose();
            trainer.force_last_weight_to_1(true);
            trainer.set_epsilon(solver_eps);

            svm_c_linear_dcd_trainer<kernel_type>::optimizer_state state;
            auto df = trainer.train(x,y, state);

            // save the active constraints for later so we can use them inside add() to add
            // new points efficiently.
            if (active_constraints.size() == 0)
            {
                long k = 0;
                for (size_t i = 0; i < points.size(); ++i)
                {
                    for (size_t j = i+1; j < points.size(); ++j)
                    {
                        if (state.get_alpha()[k++] != 0)
                            active_constraints.push_back(std::make_pair(i,j));
                    }
                }
            }
            else
            {
                DLIB_CASSERT(state.get_alpha().size() == active_constraints.size());
                new_active_constraints.clear();
                for (size_t i = 0; i < state.get_alpha().size(); ++i)
                {
                    if (state.get_alpha()[i] != 0)
                        new_active_constraints.push_back(active_constraints[i]);
                }
                active_constraints.swap(new_active_constraints);
            }

            //std::cout << "points.size(): " << points.size() << std::endl;
            //std::cout << "active_constraints.size(): " << active_constraints.size() << std::endl;


            const auto& bv = df.basis_vectors(0);
            slopes.set_size(dims);
            for (long i = 0; i < dims; ++i)
                slopes(i) = bv[i].second*xscale[i]*xscale[i];

            //std::cout << "slopes:" << trans(slopes);

            offsets.assign(points.size(),0);


            for (size_t i = 0; i < points.size(); ++i)
            {
                offsets[i] += bv[slopes.size()+i].second*relative_noise_magnitude;
            }
        }



        double relative_noise_magnitude = 0.001;
        double solver_eps = 0.0001; 
        std::vector<std::pair<size_t,size_t>> active_constraints, new_active_constraints;

        std::vector<function_evaluation> points;
        std::vector<double> offsets; // offsets.size() == points.size()
        matrix<double,0,1> slopes; // slopes.size() == points[0].first.size()
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_UPPER_bOUND_FUNCTION_Hh_


