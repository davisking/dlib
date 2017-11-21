// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GLOBAL_FuNCTION_SEARCH_Hh_
#define DLIB_GLOBAL_FuNCTION_SEARCH_Hh_

#include <vector>
#include "../matrix.h"
#include <mutex>
#include "../rand.h"
#include "upper_bound_function.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct function_spec
    {
        function_spec(
            const matrix<double,0,1>& lower_, 
            const matrix<double,0,1>& upper_
        );

        function_spec(
            const matrix<double,0,1>& lower, 
            const matrix<double,0,1>& upper, 
            std::vector<bool> is_integer
        );

        matrix<double,0,1> lower;
        matrix<double,0,1> upper;
        std::vector<bool> is_integer_variable;
    };

// ----------------------------------------------------------------------------------------

    namespace gopt_impl 
    {
        struct outstanding_function_eval_request
        {
            size_t request_id = 0;   // unique id for this eval request
            matrix<double,0,1> x;   // function x to evaluate 

            // trust region specific stuff
            bool was_trust_region_generated_request = false;
            double predicted_improvement = std::numeric_limits<double>::quiet_NaN();
            double anchor_objective_value = std::numeric_limits<double>::quiet_NaN(); // objective value at center of TR step

            bool operator==(const outstanding_function_eval_request& item) const { return request_id == item.request_id; }
        };

        struct funct_info
        {
            funct_info() = delete;
            funct_info(const funct_info&) = delete;
            funct_info& operator=(const funct_info&) = delete;

            funct_info(
                const function_spec& spec,
                size_t function_idx, 
                const std::shared_ptr<std::mutex>& m
            ) : 
                spec(spec), function_idx(function_idx), m(m)
            {
                best_x = zeros_matrix(spec.lower);
            }

            upper_bound_function build_upper_bound_with_all_function_evals (
            ) const;

            static double find_nn (
                const std::vector<function_evaluation>& evals,
                const matrix<double,0,1>& x
            );


            function_spec spec;
            size_t function_idx = 0;
            std::shared_ptr<std::mutex> m;
            upper_bound_function ub;
            std::vector<outstanding_function_eval_request> incomplete_evals;
            matrix<double,0,1> best_x; 
            double best_objective_value = -std::numeric_limits<double>::infinity(); 
            double radius = 0;
        };

    }

// ----------------------------------------------------------------------------------------

    class function_evaluation_request
    {
    public:

        function_evaluation_request() = delete;
        function_evaluation_request(const function_evaluation_request&) = delete;
        function_evaluation_request& operator=(const function_evaluation_request&) = delete;


        function_evaluation_request(function_evaluation_request&& item);
        function_evaluation_request& operator=(function_evaluation_request&& item);

        void swap(function_evaluation_request& item);

        size_t function_idx (
        ) const;

        const matrix<double,0,1>& x (
        ) const;

        bool has_been_evaluated (
        ) const;

        ~function_evaluation_request();

        void set (
            double y
        );
        /*!
            requires
                - has_been_evaluated() == false
            ensures
                - #has_been_evaluated() == true
        !*/

    private:

        friend class global_function_search;

        explicit function_evaluation_request(
            const gopt_impl::outstanding_function_eval_request& req,
            const std::shared_ptr<gopt_impl::funct_info>& info
        ) : req(req), info(info) {}

        bool m_has_been_evaluated = false;
        gopt_impl::outstanding_function_eval_request req;
        std::shared_ptr<gopt_impl::funct_info> info;
    };

// ----------------------------------------------------------------------------------------

    class global_function_search
    {
    public:

        global_function_search() = delete;

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

        global_function_search(const global_function_search&) = delete;
        global_function_search& operator=(const global_function_search& item) = delete;

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

        function_evaluation_request get_next_x (
        ); 

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

    private:

        std::shared_ptr<gopt_impl::funct_info> best_function(
        ) const;

        std::shared_ptr<gopt_impl::funct_info> best_function(
            size_t& idx
        ) const;

        bool has_incomplete_trust_region_request (
        ) const;


        dlib::rand rnd;
        double pure_random_search_probability = 0.02;
        double min_trust_region_epsilon = 1e-11;
        double relative_noise_magnitude = 0.001;
        size_t num_random_samples = 5000;
        bool do_trust_region_step = true;

        size_t next_request_id = 1;

        std::vector<std::shared_ptr<gopt_impl::funct_info>> functions;
        std::shared_ptr<std::mutex> m;

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GLOBAL_FuNCTION_SEARCH_Hh_

