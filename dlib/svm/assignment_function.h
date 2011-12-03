// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ASSIGNMENT_FuNCTION_H__
#define DLIB_ASSIGNMENT_FuNCTION_H__

#include "assignment_function_abstract.h"
#include "../matrix.h"
#include <vector>
#include "../optimization/max_cost_assignment.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor 
        >
    class assignment_function
    {
    public:

        typedef typename feature_extractor::lhs_type lhs_type;
        typedef typename feature_extractor::rhs_type rhs_type;


        typedef std::pair<std::vector<lhs_type>, std::vector<rhs_type> > sample_type;

        typedef std::vector<long> label_type;
        typedef label_type result_type;

        assignment_function()
        {
            weights.set_size(fe.num_features());
            weights = 0;
            force_assignment = false;
        }

        explicit assignment_function(
            const matrix<double,0,1>& weights_
        ) : 
            weights(weights_),
            force_assignment(false)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(fe.num_features() == static_cast<unsigned long>(weights_.size()),
                "\t assignment_function::assignment_function(weights_)"
                << "\n\t These sizes should match"
                << "\n\t fe.num_features(): " << fe.num_features() 
                << "\n\t weights_.size():   " << weights_.size() 
                << "\n\t this: " << this
                );

        }

        assignment_function(
            const feature_extractor& fe_,
            const matrix<double,0,1>& weights_
        ) :
            fe(fe_),
            weights(weights_),
            force_assignment(false)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(fe_.num_features() == static_cast<unsigned long>(weights_.size()),
                "\t assignment_function::assignment_function(fe_,weights_)"
                << "\n\t These sizes should match"
                << "\n\t fe_.num_features(): " << fe_.num_features() 
                << "\n\t weights_.size():    " << weights_.size() 
                << "\n\t this: " << this
                );
        }

        assignment_function(
            const feature_extractor& fe_,
            const matrix<double,0,1>& weights_,
            bool force_assignment_
        ) :
            fe(fe_),
            weights(weights_),
            force_assignment(force_assignment_)
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(fe_.num_features() == static_cast<unsigned long>(weights_.size()),
                "\t assignment_function::assignment_function(fe_,weights_,force_assignment_)"
                << "\n\t These sizes should match"
                << "\n\t fe_.num_features(): " << fe_.num_features() 
                << "\n\t weights_.size():    " << weights_.size() 
                << "\n\t this: " << this
                );
        }


        result_type operator()(
            const std::vector<lhs_type>& lhs,
            const std::vector<rhs_type>& rhs 
        ) const
        /*!
            ensures
                - returns a vector A such that:
                    - A.size() == lhs.size()
                    - if (A[i] != -1) then
                        - lhs[i] is predicted to associate to rhs[A[i]]
        !*/
        {
            using dlib::sparse_vector::dot;
            using dlib::dot;

            matrix<double> cost;
            unsigned long size;
            if (force_assignment)
            {
                size = std::max(lhs.size(), rhs.size());
            }
            else
            {
                size = rhs.size() + lhs.size();
            }
            cost.set_size(size, size);

            // now fill out the cost assignment matrix
            for (long r = 0; r < cost.nr(); ++r)
            {
                for (long c = 0; c < cost.nc(); ++c)
                {
                    if (r < (long)lhs.size() && c < (long)rhs.size())
                    {
                        cost(r,c) = dot(weights, fe(lhs[r], rhs[c]));
                    }
                    else
                    {
                        cost(r,c) = 0;
                    }
                }
            }

            std::vector<long> assignment;

            if (cost.size() != 0)
            {
                // max_cost_assignment() only works with integer matrices, so convert from
                // double to integer.
                const double scale = (std::numeric_limits<dlib::int64>::max()/1000)/max(abs(cost));
                matrix<dlib::int64> int_cost = matrix_cast<dlib::int64>(round(cost*scale));
                assignment = max_cost_assignment(int_cost);
                assignment.resize(lhs.size());
            }

            // adjust assignment so that non-assignments have a value of -1
            for (unsigned long i = 0; i < assignment.size(); ++i)
            {
                if (assignment[i] >= (long)rhs.size())
                    assignment[i] = -1;
            }

            return assignment;
        }


        result_type operator() (
            const sample_type& item
        ) const
        {
            return (*this)(item.first, item.second);
        }

    private:


        feature_extractor fe;
        matrix<double,0,1> weights;
        bool force_assignment;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ASSIGNMENT_FuNCTION_H__

