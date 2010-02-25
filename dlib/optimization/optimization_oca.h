// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATIoN_OCA_H__
#define DLIB_OPTIMIZATIoN_OCA_H__

#include "optimization_oca_abstract.h"

#include "../matrix.h"
#include "optimization_solve_qp_using_smo.h"
#include <list>

// ----------------------------------------------------------------------------------------

namespace dlib
{
    template <typename matrix_type>
    class oca_problem
    {
    public:
        typedef typename matrix_type::type scalar_type;

        virtual ~oca_problem() {}

        virtual void optimization_status (
            scalar_type ,
            scalar_type ,
            unsigned long 
        ) const {}

        virtual bool R_has_lower_bound (
            scalar_type& 
        ) const { return false; }

        virtual scalar_type get_C (
        ) const = 0;

        virtual long get_num_dimensions (
        ) const = 0;

        virtual void get_risk (
            matrix_type& current_solution,
            scalar_type& risk_value,
            matrix_type& risk_subgradient
        ) const = 0;

    };

// ----------------------------------------------------------------------------------------

    class oca
    {
    public:

        oca () 
        {
            eps = 0.001;
            max_iter = 1000000;

            sub_eps = 1e-5;
            sub_max_iter = 20000;

            inactive_thresh = 15;
        }

        void set_epsilon (
            double eps_
        ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(eps_ > 0,
                "\t void oca::set_epsilon"
                << "\n\t epsilon must be greater than 0"
                << "\n\t eps_: " << eps_ 
                << "\n\t this: " << this
                );

            eps = eps_; 
        }

        double get_epsilon (
        ) const { return eps; }

        void set_max_iterations (
            unsigned long max_iter_
        ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(max_iter_ > 0,
                "\t void oca::set_max_iterations"
                << "\n\t max iterations must be greater than 0"
                << "\n\t max_iter_: " << max_iter_
                << "\n\t this: " << this
                );

            max_iter = max_iter_; 
        }

        unsigned long get_max_iterations (
        ) const { return max_iter; }

        void set_subproblem_epsilon (
            double eps_
        ) { sub_eps = eps_; }

        double get_subproblem_epsilon (
        ) const { return sub_eps; }

        void set_subproblem_max_iterations (
            unsigned long sub_max_iter_
        ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(sub_max_iter_ > 0,
                "\t void oca::set_subproblem_max_iterations"
                << "\n\t max iterations must be greater than 0"
                << "\n\t sub_max_iter_: " << sub_max_iter_
                << "\n\t this: " << this
                );

            sub_max_iter = sub_max_iter_; 
        }

        unsigned long get_subproblem_max_iterations (
        ) const { return sub_max_iter; }

        void set_inactive_plane_threshold (
            unsigned long inactive_thresh_
        ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(inactive_thresh_ > 0,
                "\t void oca::set_inactive_plane_threshold"
                << "\n\t inactive threshold must be greater than 0"
                << "\n\t inactive_thresh_: " << inactive_thresh_
                << "\n\t this: " << this
                );

            inactive_thresh = inactive_thresh_; 
        }

        unsigned long get_inactive_plane_threshold (
        ) const { return inactive_thresh; }

        template <
            typename matrix_type
            >
        typename matrix_type::type operator() (
            const oca_problem<matrix_type>& problem,
            matrix_type& w
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(problem.get_C() > 0 &&
                        problem.get_num_dimensions() > 0,
                "\t void oca::operator()"
                << "\n\t The oca_problem is invalid"
                << "\n\t problem.get_C():              " << problem.get_C() 
                << "\n\t problem.get_num_dimensions(): " << problem.get_num_dimensions() 
                << "\n\t this: " << this
                );

            typedef typename matrix_type::type scalar_type;
            typedef typename matrix_type::layout_type layout_type;
            typedef typename matrix_type::mem_manager_type mem_manager_type;
            typedef matrix<scalar_type,0,1,mem_manager_type, layout_type> vect_type;

            const scalar_type C = problem.get_C();

            std::list<vect_type> planes;
            std::vector<scalar_type> bs, miss_count;

            vect_type temp, alpha, w_cur;

            w.set_size(problem.get_num_dimensions(), 1);
            w = 0;
            w_cur = w;

            // The best objective value seen so far.   Note also
            // that w always contains the best solution seen so far.
            scalar_type best_obj = std::numeric_limits<scalar_type>::max();

            // This will hold the cutting plane objective value.  This value is
            // a lower bound on the true optimal objective value.
            scalar_type cp_obj = 0;

            scalar_type R_lower_bound;
            if (problem.R_has_lower_bound(R_lower_bound))
            {
                // The flat lower bounding plane is always good to have if we know
                // what it is.
                bs.push_back(R_lower_bound);
                planes.push_back(zeros_matrix<scalar_type>(w.size(),1));
                miss_count.push_back(0);
            }

            matrix<scalar_type,0,0,mem_manager_type, layout_type> K;

            for (unsigned long iter = 0; iter < max_iter; ++iter)
            {

                // add the next cutting plane
                scalar_type cur_risk;
                planes.resize(planes.size()+1);
                problem.get_risk(w_cur, cur_risk, planes.back());
                bs.push_back(cur_risk - dot(w_cur,planes.back()));
                miss_count.push_back(0);

                // Check the objective value at w_cur and see if it is better than
                // the best seen so far.
                const scalar_type cur_obj = 0.5*trans(w_cur)*w_cur + C*cur_risk;
                if (cur_obj < best_obj)
                {
                    best_obj = cur_obj;
                    w = w_cur;
                }

                // check stopping condition and stop if we can
                if (best_obj - cp_obj <= eps)
                    break;


                // compute kernel matrix for all the planes
                K.set_size(planes.size(), planes.size());
                long rr = 0;
                for (typename std::list<vect_type>::iterator r = planes.begin(); r != planes.end(); ++r)
                {
                    long cc = rr;
                    for (typename std::list<vect_type>::iterator c = r; c != planes.end(); ++c)
                    {
                        K(rr,cc) = dot(*r, *c);
                        K(cc,rr) = K(rr,cc);
                        ++cc;
                    }
                    ++rr;
                }

                alpha = uniform_matrix<scalar_type>(planes.size(),1, C/planes.size());

                // solve the cutting plane subproblem for the next w_cur
                solve_qp_using_smo(K, vector_to_matrix(bs), alpha, static_cast<scalar_type>(sub_eps), sub_max_iter); 

                // construct the w_cur that minimized the subproblem.
                w_cur = 0;
                rr = 0;
                for (typename std::list<vect_type>::iterator i = planes.begin(); i != planes.end(); ++i)
                {
                    if (alpha(rr) != 0)
                    {
                        w_cur -= alpha(rr)*(*i);
                        miss_count[rr] = 0;
                    }
                    else
                    {
                        miss_count[rr] += 1;
                    }
                    ++rr;
                }

                // Compute the lower bound on the true objective given to us by the cutting 
                // plane subproblem.
                cp_obj = -0.5*trans(w_cur)*w_cur + trans(alpha)*vector_to_matrix(bs);

                // check stopping condition and stop if we can
                if (best_obj - cp_obj <= eps)
                    break;

                // report current status
                problem.optimization_status(best_obj, best_obj - cp_obj, planes.size());

                // If it has been a while since a cutting plane was an active constraint then
                // we should throw it away.
                while (max(vector_to_matrix(miss_count)) >= inactive_thresh)
                {
                    long idx = index_of_max(vector_to_matrix(miss_count));
                    typename std::list<vect_type>::iterator i0 = planes.begin();
                    advance(i0, idx);
                    planes.erase(i0);
                    bs.erase(bs.begin()+idx);
                    miss_count.erase(miss_count.begin()+idx);
                }

            }

            // report current status
            problem.optimization_status(best_obj, best_obj - cp_obj, planes.size());

            return best_obj;
        }

    private:

        double eps;
        double sub_eps;

        unsigned long max_iter;
        unsigned long sub_max_iter;

        unsigned long inactive_thresh;
    };
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_OPTIMIZATIoN_OCA_H__

