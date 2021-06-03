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
                multivariate function.  It lets you define bound constraints for each
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
                This object represents a request, by the global_function_search object, to
                evaluate a real-valued function and report back the results.  

            THREAD SAFETY
                You shouldn't let more than one thread touch a function_evaluation_request
                at the same time.  However, it is safe to send instances of this class to
                other threads for processing.  This lets you evaluate multiple
                function_evaluation_requests in parallel.  Any appropriate synchronization
                with regard to the originating global_function_search instance is handled
                automatically.
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
            ensures
                - *this takes the state of item.  
                - #item.has_been_evaluated() == true
        !*/

        ~function_evaluation_request(
        );
        /*!
            ensures
                - frees all resources associated with this object.  
                - It's fine to destruct function_evaluation_requests even if they haven't
                  been evaluated yet. If this happens it will simply be as if the request
                  was never issued.
        !*/

        size_t function_idx (
        ) const;
        /*!
            ensures
                - Returns the function index that identifies which function is to be
                  evaluated.  
        !*/

        const matrix<double,0,1>& x (
        ) const;
        /*!
            ensures
                - returns the input parameters to the function to be evaluated.
        !*/

        bool has_been_evaluated (
        ) const;
        /*!
            ensures
                - If this evaluation request is still outstanding then returns false,
                  otherwise returns true.  That is, if the global_function_search is still
                  waiting for you report back by calling set() then
                  has_been_evaluated()==false.
        !*/

        void set (
            double y
        );
        /*!
            requires
                - has_been_evaluated() == false
            ensures
                - #has_been_evaluated() == true
                - Notifies the global_function_search instance that created this object
                  that when the function_idx()th function is evaluated with x() as input
                  then the output is y.
        !*/

        void swap(
            function_evaluation_request& item
        );
        /*!
            ensures
                - swaps the state of *this and item
        !*/

    };

// ----------------------------------------------------------------------------------------

    class global_function_search
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object performs global optimization of a set of user supplied
                functions.  The goal is to maximize the following objective function:
                    max_{function_i,x_i}: function_i(x_i)
                subject to bound constraints on each element of x_i.  Moreover, each
                element of x_i can be either real valued or integer valued.  Each of the
                functions can also take a different number of variables.  Therefore, the
                final result of the optimization tells you which function produced the
                largest output and what input (i.e. the x value) to that function is
                necessary to obtain that maximal value.

                Importantly, the global_function_search object does not require the user to
                supply derivatives.  Moreover, the functions may contain discontinuities,
                behave stochastically, and have many local maxima.  The global_function_search 
                object will attempt to find the global optima in the face of these challenges.  
                It is also designed to use as few function evaluations as possible, making
                it suitable for optimizing functions that are very expensive to evaluate.  

                It does this by alternating between two modes.  A global exploration mode
                and a local optima refinement mode.  This is accomplished by building and
                maintaining two models of the objective function:
                    1. A global model that upper bounds our objective function.  This is a
                       non-parametric piecewise linear model based on all function
                       evaluations ever seen by the global_function_search object.  
                    2. A local quadratic model fit around the best point seen so far.   
                
                The optimization procedure therefore looks like this:

                    while(not done) 
                    {
                        DO GLOBAL EXPLORE STEP:
                           Find the point that maximizes the upper bounding model since
                           that is the point with the largest possible improvement in the
                           objective function.

                           Evaluate the new point and incorporate it into our models.

                        DO LOCAL REFINEMENT STEP:
                           Find the optimal solution to the local quadratic model.  
                           
                           If this point looks like it will improve on the "best point seen
                           so far" by at least get_solver_epsilon() then we evaluate that
                           point and incorporate it into our models, otherwise we ignore
                           it.
                    }

                You can see that we alternate between global search and local refinement,
                except in the case where the local model seems to have converged to within
                get_solver_epsilon() accuracy.  In that case only global search steps are
                used.  We do this in the hope that the global search will find a new and
                better local optima to explore, which would then reactivate local
                refinement when it has something productive to do. 

                
                Now let's turn our attention to the specific API defined by the
                global_function_search object.  We will begin by showing a short example of
                its use:

                    // Suppose we want to find which of these functions, F() and G(), have
                    // the largest output and what input is necessary to produce the
                    // maximal output.
                    auto F = [](double a, double b) { return  -std::pow(a-2,2.0) - std::pow(b-4,2.0); };
                    auto G = [](double x)           { return 2-std::pow(x-5,2.0); };

                    // We first define function_spec objects that specify bounds on the
                    // inputs to each function.  The search process will only search within
                    // these bounds.
                    function_spec spec_F({-10,-10}, {10,10});
                    function_spec spec_G({-2}, {6});
                    // Then we create a global_function_search object with those function specifications.
                    global_function_search opt({spec_F, spec_G});

                    // Here we run 15 iterations of the search process.  Note that the user
                    // of global_function_search writes the main solver loop, which is
                    // somewhat unusual.  We will discuss why that is in a moment, but for
                    // now let's look at this example.
                    for (int i = 0; i < 15; ++i)
                    {
                        // All we do here is ask the global_function_search object what to
                        // evaluate next, then do what it asked, and then report the
                        // results back by calling function_evaluation_request's set()
                        // method.  
                        function_evaluation_request next = opt.get_next_x();
                        // next.function_idx() tells you which of the functions you should
                        // evaluate.  We have 2 functions here (F and G) so function_idx()
                        // can take only the values 0 and 1.  If, for example, we had 10
                        // functions it would take the values 0 through 9.
                        if (next.function_idx() == 0)
                        {
                            // Call F with the inputs requested by the
                            // global_function_search and report them back.
                            double a = next.x()(0);
                            double b = next.x()(1);
                            next.set(F(a,b));  // Tell the solver what happened.
                        }
                        else
                        {
                            double x = next.x()(0);
                            next.set(G(x));
                        }
                    }

                    // Find out what point gave the largest outputs:
                    matrix<double,0,1> x;
                    double y;
                    size_t function_idx;
                    opt.get_best_function_eval(x,y,function_idx);

                    cout << "function_idx: "<< function_idx << endl; 
                    cout << "y: " << y << endl; 
                    cout << "x: " << x << endl; 

                The above cout statements will print this:

                    function_idx: 1
                    y: 2
                    x: 5 
                
                Which is the correct result since G(5) gives the largest possible output in
                our example.

                So why does the user write the main loop?  Why isn't it embedded inside
                dlib?  Well, there are two answers to this.  The first is that it is.  Most
                users should just call dlib::find_max_global() which does exactly that, it
                runs the loop for you.  However, the API shown above gives you the
                opportunity to run multiple function evaluations in parallel.  For
                instance, it is perfectly valid to call get_next_x() multiple times and
                send the resulting function_evaluation_request objects to separate threads
                for processing.  Those separate threads can run the functions being
                optimized (e.g. F and G or whatever) and report back by calling
                function_evaluation_request::set().  You could even spread the work across
                a compute cluster if you have one.  Note that find_max_global() optionally
                supports this type of parallel execution, however you get more flexibility
                with the global_function_search's API.  As another example, it's possible
                to save the state of the solver to disk so you can restart the optimization
                from that point at a later date when using global_function_search, but not
                with find_max_global().

                So what happens if you have N outstanding function evaluation requests?
                Or in other words, what happens if you called get_next_x() N times and
                haven't yet called their set() methods?  Well, 1 of the N requests will be
                a local refinement step while the N-1 other requests will be global
                exploration steps generated from the current upper bounding model.  This
                should give you an idea of the usefulness of this kind of parallelism.  If
                for example, your functions being optimized were simple convex functions
                this kind of parallelism wouldn't help since essentially all the
                interesting work in the solver is going to be done by the local optimizer.
                However, if your function has a lot of local optima, running many global
                exploration steps in parallel might significantly reduce the time it takes
                to find a good solution.
                
                It should also be noted that our upper bounding model is implemented by the
                dlib::upper_bound_function object, which is a tool that allows us to create
                a tight upper bound on our objective function.  This upper bound is
                non-parametric and gets progressively more accurate as the optimization
                progresses, but also more and more expensive to maintain.  It causes the
                runtime of the entire optimization procedure to be O(N^2) where N is the
                number of objective function evaluations. So problems that require millions
                of function evaluations to find a good solution are not appropriate for the
                global_function_search tool.  However, if your objective function is very
                expensive to evaluate then this relatively expensive upper bounding model
                is well worth its computational cost.
                
                Finally, let's introduce some background literature on this algorithm.  The
                two most relevant papers in the optimization literature are:
                    Global optimization of Lipschitz functions Malherbe, Cédric and Vayatis,
                    Nicolas International Conference on Machine Learning - 2017
                and
                    The NEWUOA software for unconstrained optimization without derivatives By
                    M.J.D. Powell, 40th Workshop on Large Scale Nonlinear Optimization (Erice,
                    Italy, 2004)

                Our upper bounding model is an extension of the AdaLIPO method in the
                Malherbe.  See the documentation of dlib::upper_bound_function for more
                details on that, as we make a number of important extensions.  The other
                part of our method, our local refinement model, is essentially the same
                type of trust region model proposed by Powell in the above paper.  That is,
                each time we do a local refinement step we identify the best point seen so
                far, fit a quadratic function around it using the function evaluations we
                have collected so far, and then use a simple trust region procedure to
                decide the next best point to evaluate based on our quadratic model.  

                The method proposed by Malherbe gives excellent global search performance
                but has terrible convergence properties in the area around a maxima.
                Powell's method on the other hand has excellent convergence in the area
                around a local maxima, as expected by a quadratic trust region method, but
                is aggressively local maxima seeking.  It will simply get stuck in the
                nearest local optima.  Combining the two together as we do here gives us
                excellent performance in both global search and final convergence speed
                near a local optima.  Causing the global_function_search to perform well
                for functions with many local optima while still giving high precision
                solutions.  For instance, on typical tests problems, like the Holder table
                function, the global_function_search object can reliably find the globally
                optimal solution to full floating point precision in under a few hundred
                steps.


            THREAD SAFETY
                You shouldn't let more than one thread touch a global_function_search
                instance at the same time.  
        !*/

    public:

        global_function_search(
        );
        /*!
            ensures
                - #num_functions() == 0
                - #get_relative_noise_magnitude() == 0.001
                - #get_solver_epsilon() == 0
                - #get_monte_carlo_upper_bound_sample_num() == 5000
                - #get_pure_random_search_probability() == 0.02
        !*/

        explicit global_function_search(
            const function_spec& function
        ); 
        /*!
            ensures
                - #num_functions() == 1
                - #get_function_evaluations() will indicate that there are no function evaluations yet.
                - #get_relative_noise_magnitude() == 0.001
                - #get_solver_epsilon() == 0
                - #get_monte_carlo_upper_bound_sample_num() == 5000
                - #get_pure_random_search_probability() == 0.02
        !*/

        explicit global_function_search(
            const std::vector<function_spec>& functions
        );
        /*!
            ensures
                - #num_functions() == functions.size()
                - #get_function_evaluations() will indicate that there are no function evaluations yet.
                - #get_relative_noise_magnitude() == 0.001
                - #get_solver_epsilon() == 0
                - #get_monte_carlo_upper_bound_sample_num() == 5000
                - #get_pure_random_search_probability() == 0.02
        !*/

        global_function_search(
            const std::vector<function_spec>& functions,
            const std::vector<std::vector<function_evaluation>>& initial_function_evals,
            const double relative_noise_magnitude = 0.001
        ); 
        /*!
            requires
                - functions.size() == initial_function_evals.size()
                - relative_noise_magnitude >= 0
            ensures
                - #num_functions() == functions.size()
                - #get_function_evaluations() will return the provided initial_function_evals.
                - #get_relative_noise_magnitude() == relative_noise_magnitude
                - #get_solver_epsilon() == 0
                - #get_monte_carlo_upper_bound_sample_num() == 5000
                - #get_pure_random_search_probability() == 0.02
        !*/

        // This object can't be copied.
        global_function_search(const global_function_search&) = delete;
        global_function_search& operator=(const global_function_search& item) = delete;
        // But it can be moved
        global_function_search(global_function_search&& item) = default;
        global_function_search& operator=(global_function_search&& item) = default;
        /*!
            ensures
                - moves the state of item into *this
                - #item.num_functions() == 0
        !*/

        void set_seed (
            time_t seed
        );
        /*!
            ensures
                - Part of this object's algorithm uses random sampling to decide what
                  points to evaluate next.  Calling set_seed() lets you set the seed used
                  by the random number generator.   Note that if you don't call set_seed()
                  you will always get the same deterministic behavior.
        !*/

        size_t num_functions(
        ) const;
        /*!
            ensures
                - returns the number of functions being optimized.
        !*/

        void get_function_evaluations (
            std::vector<function_spec>& specs,
            std::vector<std::vector<function_evaluation>>& function_evals
        ) const;
        /*!
            ensures
                - #specs.size() == num_functions()
                - #function_evals.size() == num_functions()
                - This function allows you to query the state of the solver.  In
                  particular, you can find the function_specs for each function being
                  optimized and their recorded evaluations.  
                - for all valid i:
                    - function_evals[i] == all the function evaluations that have been
                      recorded for the ith function (i.e. the function with the
                      function_spec #specs[i]).  That is, this is the record of all the x
                      and y values reported back by function_evaluation_request::set()
                      calls.
        !*/

        void get_best_function_eval (
            matrix<double,0,1>& x,
            double& y,
            size_t& function_idx
        ) const;
        /*!
            requires
                - num_functions() != 0
            ensures
                - if (no function evaluations have been recorded yet) then
                    - The outputs of this function are in a valid but undefined state.
                - else
                    - This function tells you which function has produced the largest
                      output seen so far.  It also tells you the inputs to that function
                      that leads to those outputs (x) as well as the output value itself (y).
                    - 0 <= #function_idx < num_functions()
                    - #function_idx == the index of the function that produced the largest output seen so far.
                    - #x == the input parameters to the function that produced the largest outputs seen so far.
                    - #y == the largest output seen so far.
        !*/

        function_evaluation_request get_next_x (
        ); 
        /*!
            requires
                - num_functions() != 0
            ensures
                - Generates and returns a function evaluation request.  See the discussion
                  in the WHAT THIS OBJECT REPRESENTS section above for details.
        !*/

        double get_pure_random_search_probability (
        ) const; 
        /*!
            ensures
                - When we decide to do a global explore step we will, with probability
                  get_pure_random_search_probability(), sample a point completely at random
                  rather than using the upper bounding model.  Therefore, if you set this
                  probability to 0 then we will depend entirely on the upper bounding
                  model.  Alternatively, if you set get_pure_random_search_probability() to
                  1 then we won't use the upper bounding model at all and instead use pure
                  random search to do global exploration.  Pure random search is much
                  faster than using the upper bounding model, so if you know that your
                  objective function is especially simple it can be faster to use pure
                  random search.  However, if you really know your function that well you
                  should probably use a gradient based optimizer :)
        !*/

        void set_pure_random_search_probability (
            double prob
        );
        /*!
            requires
                - prob >= 0
            ensures
                - #get_pure_random_search_probability() == prob
        !*/

        double get_solver_epsilon (
        ) const; 
        /*!
            ensures
                - As discussed in the WHAT THIS OBJECT REPRESENTS section, we only do a
                  local refinement step if we haven't already found the peak of the current
                  local optima.  get_solver_epsilon() sets the tolerance for deciding if
                  the local search method has found the local optima.  Therefore, when the
                  local trust region model runs we check if its predicted improvement in
                  the objective function is greater than get_solver_epsilon().  If it isn't
                  then we assume it has converged and we should focus entirely on global
                  search.

                  This means that, for instance, setting get_solver_epsilon() to 0
                  essentially instructs the solver to find each local optima to full
                  floating point precision and only then to focus on pure global search.
        !*/

        void set_solver_epsilon (
            double eps
        );
        /*!
            requires
                - eps >= 0
            ensures
                - #get_solver_epsilon() == eps 
        !*/

        double get_relative_noise_magnitude (
        ) const; 
        /*!
            ensures
                - Returns the value of the relative noise magnitude parameter to the
                  dlib::upper_bound_function's used by this object.  See the
                  upper_bound_function's documentation for a detailed discussion of this
                  parameter's meaning.  Most users should leave this value as its default
                  setting.
        !*/

        void set_relative_noise_magnitude (
            double value
        );
        /*!
            requires
                - value >= 0
            ensures
                - #get_relative_noise_magnitude() == value 
        !*/

        size_t get_monte_carlo_upper_bound_sample_num (
        ) const; 
        /*!
            ensures
                - To find the point that maximizes the upper bounding model we use
                  get_monte_carlo_upper_bound_sample_num() random evaluations and select
                  the largest upper bound from that set.   So this parameter influences how
                  well we estimate the maximum point on the upper bounding model. 
        !*/

        void set_monte_carlo_upper_bound_sample_num (
            size_t num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_monte_carlo_upper_bound_sample_num() == num 
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GLOBAL_FuNCTION_SEARCH_ABSTRACT_Hh_


