// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_GLOBAL_FuNCTION_SEARCH_ABSTRACT_Hh_
#ifdef DLIB_GLOBAL_FuNCTION_SEARCH_ABSTRACT_Hh_

#include <vector>
#include "../matrix.h"
#include "upper_bound_function_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct function_spec
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a simple struct that lets you define the valid inputs to a
                multivariate function.  It lets you define bounds constraints for each
                variable as well as say if a variable is integer valued or not.  Therefore,
                an instance of this struct says that a function takes upper.size() input
                variables, where the ith variable must be in the range [lower(i) upper(i)]
                and be an integer if is_integer_variable[i]==true.
        !*/

        function_spec(
            matrix<double,0,1> bound1, 
            matrix<double,0,1> bound2
        );
        /*!
            requires
                - bound1.size() == bound2.size()
                - for all valid i: bound1(i) != bound2(i)
            ensures
                - #is_integer_variable.size() == bound1.size()
                - #lower.size() == bound1.size()
                - #upper.size() == bound1.size()
                - for all valid i:
                    - #is_integer_variable[i] == false
                    - #lower(i) == min(bound1(i), bound2(i))
                    - #upper(i) == max(bound1(i), bound2(i))
        !*/

        function_spec(
            matrix<double,0,1> lower, 
            matrix<double,0,1> upper, 
            std::vector<bool> is_integer
        );
        /*!
            requires
                - bound1.size() == bound2.size() == is_integer.size()
                - for all valid i: bound1(i) != bound2(i)
            ensures
                - #is_integer_variable.size() == bound1.size()
                - #lower.size() == bound1.size()
                - #upper.size() == bound1.size()
                - for all valid i:
                    - #is_integer_variable[i] == is_integer[i] 
                    - #lower(i) == min(bound1(i), bound2(i))
                    - #upper(i) == max(bound1(i), bound2(i))
        !*/

        matrix<double,0,1> lower;
        matrix<double,0,1> upper;
        std::vector<bool> is_integer_variable;
    };

// ----------------------------------------------------------------------------------------

    class function_evaluation_request
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        // You can't make or copy this object, the only way to get one is from the
        // global_function_search class via get_next_x().
        function_evaluation_request() = delete;
        function_evaluation_request(const function_evaluation_request&) = delete;
        function_evaluation_request& operator=(const function_evaluation_request&) = delete;

        // You can however move and swap this object.
        function_evaluation_request(function_evaluation_request&& item);
        function_evaluation_request& operator=(function_evaluation_request&& item);
        /*!
            moving from item causes item.has_been_evaluated() == true,  TODO, clarify 
        !*/

        void swap(
            function_evaluation_request& item
        );
        /*!
            ensures
                - swaps the state of *this and item
        !*/

        ~function_evaluation_request(
        );
        /*!
            ensures
                - frees all resources associated with this object.  
        !*/

        size_t function_idx (
        ) const;

        const matrix<double,0,1>& x (
        ) const;

        bool has_been_evaluated (
        ) const;


        void set (
            double y
        );
        /*!
            requires
                - has_been_evaluated() == false
            ensures
                - #has_been_evaluated() == true
        !*/

    };

// ----------------------------------------------------------------------------------------

    class global_function_search
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:

        global_function_search(
        );
        /*!
            ensures
                - #num_functions() == 0
        !*/

        // This object can't be copied.
        global_function_search(const global_function_search&) = delete;
        global_function_search& operator=(const global_function_search& item) = delete;

        global_function_search(global_function_search&& item) = default;
        global_function_search& operator=(global_function_search&& item) = default;
        /*!
            ensures
                - moves the state of item into *this
                - #item.num_functions() == 0
        !*/

        explicit global_function_search(
            const function_spec& function
        ); 

        explicit global_function_search(
            const std::vector<function_spec>& functions_
        );

        global_function_search(
            const std::vector<function_spec>& functions_,
            const std::vector<std::vector<function_evaluation>>& initial_function_evals,
            const double relative_noise_magnitude = 0.001
        ); 

        size_t num_functions(
        ) const;

        void set_seed (
            time_t seed
        );

        void get_function_evaluations (
            std::vector<function_spec>& specs,
            std::vector<std::vector<function_evaluation>>& function_evals
        ) const;

        void get_best_function_eval (
            matrix<double,0,1>& x,
            double& y,
            size_t& function_idx
        ) const;
        /*!
            requires
                - num_functions() != 0
        !*/

        function_evaluation_request get_next_x (
        ); 
        /*!
            requires
                - num_functions() != 0
        !*/

        double get_pure_random_search_probability (
        ) const; 

        void set_pure_random_search_probability (
            double prob
        );

        double get_solver_epsilon (
        ) const; 

        void set_solver_epsilon (
            double eps
        );

        double get_relative_noise_magnitude (
        ) const; 

        void set_relative_noise_magnitude (
            double value
        );

        size_t get_monte_carlo_upper_bound_sample_num (
        ) const; 

        void set_monte_carlo_upper_bound_sample_num (
            size_t num
        );

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GLOBAL_FuNCTION_SEARCH_ABSTRACT_Hh_


