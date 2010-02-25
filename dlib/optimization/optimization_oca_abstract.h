// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OPTIMIZATION_OCA_ABsTRACT_H__
#ifdef DLIB_OPTIMIZATION_OCA_ABsTRACT_H__

// ----------------------------------------------------------------------------------------

namespace dlib
{
    template <typename matrix_type>
    class oca_problem
    {
        /*!
            REQUIREMENTS ON matrix_type
                - matrix_type == a dlib::matrix capable of storing column vectors

            WHAT THIS OBJECT REPRESENTS
                This object is the interface used to define the optimization 
                problems solved by the oca optimizer defined later in this file.

                OCA solves optimization problems with the following form:
                    Minimize: f(w) == 0.5*dot(w,w) + C*R(w)

                    Where R(w) is a convex function and C > 0
        !*/

    public:

        typedef typename matrix_type::type scalar_type;

        virtual ~oca_problem() {}

        virtual void optimization_status (
            scalar_type current_objective_value,
            scalar_type current_error_gap,
            unsigned long num_cutting_planes
        ) const {}
        /*!
            This function is called by the OCA optimizer each iteration.  It
            exists to allow the user to monitor the progress of the optimization.
        !*/

        virtual bool R_has_lower_bound (
            scalar_type& lower_bound
        ) const { return false; }
        /*!
            ensures
                - if (R(w) >= a constant for all values of w) then
                    - returns true
                    - #lower_bound == the constant that lower bounds R(w)
                - else
                    - returns false
        !*/

        virtual scalar_type get_C (
        ) const = 0;
        /*!
            ensures
                - returns the C parameter
        !*/

        virtual long num_dimensions (
        ) const = 0;
        /*!
            ensures
                - returns the number of free variables in this optimization problem
        !*/

        virtual void get_risk (
            matrix_type& current_solution,
            scalar_type& risk_value,
            matrix_type& risk_subgradient
        ) const = 0;
        /*!
            requires
                - is_col_vector(current_solution) == true
                - current_solution.size() == num_dimensions()
            ensures
                - #current_solution will be set to one of the following:
                    - current_solution (i.e. it won't be modified at all)
                    - The result of a line search passing through current_solution.  
                - #risk_value == R(#current_solution) 
                - #risk_subgradient == an element of the subgradient of R() at the 
                  point #current_solution
                - Note that risk_value and risk_subgradient are NOT multiplied by get_C()
        !*/

    };

// ----------------------------------------------------------------------------------------

    class oca
    {
        /*!
            INITIAL VALUE
                - get_epsilon() == 0.001
                - get_max_iterations() == 1000000
                - get_subproblem_epsilon() == 1e-5
                - get_subproblem_max_iterations() == 20000
                - get_inactive_plane_threshold() == 15

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for solving the optimization problem defined above
                by the oca_problem abstract class.  

                For reference, OCA solves optimization problems with the following form:
                    Minimize: f(w) == 0.5*dot(w,w) + C*R(w)

                    Where R(w) is a convex function and C > 0


                For a detailed discussion you should consult the following papers
                from the Journal of Machine Learning Research:
                    Optimized Cutting Plane Algorithm for Large-Scale Risk Minimization
                        Vojtech Franc, Soren Sonnenburg; 10(Oct):2157--2192, 2009. 

                    Bundle Methods for Regularized Risk Minimization
                        Choon Hui Teo, S.V.N. Vishwanthan, Alex J. Smola, Quoc V. Le; 11(Jan):311-365, 2010. 
        !*/
    public:

        oca (
        ); 
        /*!
            ensures
                - this object is properly initialized
        !*/

        template <
            typename matrix_type
            >
        void operator() (
            const oca_problem<matrix_type>& problem,
            matrix_type& w
        ) const;
        /*!
            ensures
                - solves the given oca problem and stores the solution in #w
        !*/

        void set_epsilon (
            double eps_
        );
        /*!
            requires
                - eps > 0
            ensures
                - #get_epsilon() == eps 
        !*/

        double get_epsilon (
        ) const; 
        /*!
            ensures
                - returns the error epsilon that determines when training should stop.
                  Smaller values may result in a more accurate solution but may cause
                  the algorithm to take longer to execute.
        !*/

        void set_max_iterations (
            unsigned long max_iter
        ); 
        /*!
            requires
                - max_iter > 0
            ensures
                - #get_max_iterations() == max_iter
        !*/

        unsigned long get_max_iterations (
        ) const;
        /*!
            ensures
                - returns the maximum number of iterations this object will perform
                  while attempting to solve an oca_problem.
        !*/

        void set_subproblem_epsilon (
            double eps
        ); 
        /*!
            requires
                - eps > 0
            ensures
                - #get_subproblem_epsilon() == eps 
        !*/

        double get_subproblem_epsilon (
        ) const; 
        /*!
            ensures
                - returns the accuracy used in solving the quadratic programming
                  subproblem that is part of the overall OCA algorithm.
        !*/

        void set_subproblem_max_iterations (
            unsigned long sub_max_iter
        ); 
        /*!
            requires
                - sub_max_iter > 0
            ensures
                - #get_subproblem_max_iterations() == sub_max_iter
        !*/

        unsigned long get_subproblem_max_iterations (
        ) const; 
        /*!
            ensures
                - returns the maximum number of iterations this object will perform
                  while attempting to solve each quadratic programming subproblem.
        !*/

        void set_inactive_plane_threshold (
            unsigned long inactive_thresh
        ); 
        /*!
            requires
                - inactive_thresh > 0
            ensures
                - #get_inactive_plane_threshold() == inactive_thresh
        !*/

        unsigned long get_inactive_plane_threshold (
        ) const; 
        /*!
            ensures
                - As OCA runs it builds up a set of cutting planes.  Typically
                  cutting planes become inactive after a certain point and can then
                  be removed.  This function returns the number of iterations of
                  inactivity required before a cutting plane is removed.
        !*/

    };
}

// ----------------------------------------------------------------------------------------

#endif // DLIB_OPTIMIZATION_OCA_ABsTRACT_H__


