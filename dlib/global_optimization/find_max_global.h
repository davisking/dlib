// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FiND_GLOBAL_MAXIMUM_hH_
#define DLIB_FiND_GLOBAL_MAXIMUM_hH_

#include "find_max_global_abstract.h"
#include "global_function_search.h"
#include "../metaprogramming.h"
#include <utility>
#include <chrono>
#include <memory>
#include <thread>
#include "../threads/thread_pool_extension.h"
#include "../statistics/statistics.h"

namespace dlib
{
    namespace gopt_impl
    {

    // ----------------------------------------------------------------------------------------

        class disable_decay_to_scalar 
        {
            const matrix<double,0,1>& a;
        public:
            disable_decay_to_scalar(const matrix<double,0,1>& a) : a(a){}
            operator const matrix<double,0,1>&() const { return a;}
        };


        template <typename T, size_t... indices> 
        auto _cwv (
            T&& f, 
            const matrix<double,0,1>& a, 
            compile_time_integer_list<indices...>
        ) -> decltype(f(a(indices-1)...)) 
        {
            DLIB_CASSERT(a.size() == sizeof...(indices), 
                "You invoked dlib::call_function_and_expand_args(f,a) but the number of arguments expected by f() doesn't match the size of 'a'. "
                << "Expected " << sizeof...(indices) << " arguments but got " << a.size() << "."
            );  
            return f(a(indices-1)...); 
        }

        // Visual studio, as of November 2017, doesn't support C++11 and can't compile this code.  
        // So we write the terrible garbage in the #else for visual studio.  When Visual Studio supports C++11 I'll update this #ifdef to use the C++11 code.
#ifndef _MSC_VER 
        template <size_t max_unpack>
        struct call_function_and_expand_args
        {
            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(_cwv(std::forward<T>(f),a,typename make_compile_time_integer_range<max_unpack>::type()))
            {
                return _cwv(std::forward<T>(f),a,typename make_compile_time_integer_range<max_unpack>::type());
            }

            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(call_function_and_expand_args<max_unpack-1>::template go(std::forward<T>(f),a))
            {
                return call_function_and_expand_args<max_unpack-1>::go(std::forward<T>(f),a);
            }
        };

        template <>
        struct call_function_and_expand_args<0>
        {
            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(f(disable_decay_to_scalar(a)))
            {
                return f(disable_decay_to_scalar(a));
            }
        };
#else
        template <size_t max_unpack>
        struct call_function_and_expand_args
        {         
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(disable_decay_to_scalar(a)))  {return f(disable_decay_to_scalar(a));   }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0))) { DLIB_CASSERT(a.size() == 1); return f(a(0)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0),a(1))) { DLIB_CASSERT(a.size() == 2); return f(a(0),a(1)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0), a(1), a(2))) { DLIB_CASSERT(a.size() == 3); return f(a(0), a(1),a(2)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0), a(1), a(2), a(3))) { DLIB_CASSERT(a.size() == 4); return f(a(0), a(1), a(2), a(3)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0), a(1), a(2), a(3), a(4))) { DLIB_CASSERT(a.size() == 5); return f(a(0), a(1), a(2), a(3), a(4)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0), a(1), a(2), a(3), a(4), a(5))) { DLIB_CASSERT(a.size() == 6); return f(a(0), a(1), a(2), a(3), a(4), a(5)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(f(a(0), a(1), a(2), a(3), a(4), a(5), a(6))) { DLIB_CASSERT(a.size() == 7); return f(a(0), a(1), a(2), a(3), a(4), a(5), a(6)); }
        };
#endif
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T> 
    auto call_function_and_expand_args(
        T&& f, 
        const matrix<double,0,1>& a
    ) -> decltype(gopt_impl::call_function_and_expand_args<40>::go(f,a))
    {
        // unpack up to 40 parameters when calling f()
        return gopt_impl::call_function_and_expand_args<40>::go(std::forward<T>(f),a);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    struct max_function_calls
    {
        max_function_calls() = default;
        explicit max_function_calls(size_t max_calls) : max_calls(max_calls) {}
        size_t max_calls = std::numeric_limits<size_t>::max();
    };

// ----------------------------------------------------------------------------------------

    const auto FOREVER = std::chrono::hours(24*356*290); // 290 years

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <
            typename funct
            >
        std::pair<size_t,function_evaluation> find_max_global (
            thread_pool& tp,
            std::vector<funct>& functions,
            std::vector<function_spec> specs,
            const max_function_calls num,
            const std::chrono::nanoseconds max_runtime,
            double solver_epsilon,
            double ymult,
            std::vector<std::vector<function_evaluation>> initial_function_evals
        ) 
        {
            // Decide which parameters should be searched on a log scale.  Basically, it's
            // common for machine learning models to have parameters that should be searched on
            // a log scale (e.g. SVM C).  These parameters are usually identifiable because
            // they have bounds like [1e-5 1e10], that is, they span a very large range of
            // magnitudes from really small to really big.  So there we are going to check for
            // that and if we find parameters with that kind of bound constraints we will
            // transform them to a log scale automatically.
            std::vector<std::vector<bool>> log_scale(specs.size());
            for (size_t i = 0; i < specs.size(); ++i)
            {
                for (long j = 0; j < specs[i].lower.size(); ++j)
                {
                    if (!specs[i].is_integer_variable[j] && specs[i].lower(j) > 0 && specs[i].upper(j)/specs[i].lower(j) >= 1000)
                    {
                        log_scale[i].push_back(true);
                        specs[i].lower(j) = std::log(specs[i].lower(j));
                        specs[i].upper(j) = std::log(specs[i].upper(j));
                    }
                    else
                    {
                        log_scale[i].push_back(false);
                    }
                }
            }

            if (initial_function_evals.empty()) 
            {
                initial_function_evals.resize(specs.size());
            }

            global_function_search opt(specs, {initial_function_evals});
            opt.set_solver_epsilon(solver_epsilon);

            running_stats_decayed<double> objective_funct_eval_time(functions.size()*5);
            std::mutex eval_time_mutex;
            using namespace std::chrono;

            const auto time_to_stop = steady_clock::now() + max_runtime;

            double max_solver_overhead_time = 0;

            // Now run the main solver loop.
            for (size_t i = 0; i < num.max_calls && steady_clock::now() < time_to_stop; ++i)
            {
                const auto get_next_x_start_time = steady_clock::now();
                auto next = std::make_shared<function_evaluation_request>(opt.get_next_x());
                const auto get_next_x_runtime = steady_clock::now() - get_next_x_start_time;

                auto execute_call = [&functions,&ymult,&log_scale,&eval_time_mutex,&objective_funct_eval_time,next]() {
                    matrix<double,0,1> x = next->x();
                    // Undo any log-scaling that was applied to the variables before we pass them
                    // to the functions being optimized.
                    for (long j = 0; j < x.size(); ++j)
                    {
                        if (log_scale[next->function_idx()][j])
                            x(j) = std::exp(x(j));
                    }
                    const auto funct_eval_start = steady_clock::now();
                    double y = ymult*call_function_and_expand_args(functions[next->function_idx()], x);
                    const double funct_eval_runtime = duration_cast<nanoseconds>(steady_clock::now() - funct_eval_start).count();
                    next->set(y);
                    
                    std::lock_guard<std::mutex> lock(eval_time_mutex);
                    objective_funct_eval_time.add(funct_eval_runtime);
                };

                tp.add_task_by_value(execute_call);

                std::lock_guard<std::mutex> lock(eval_time_mutex);
                const double obj_funct_time = objective_funct_eval_time.mean()/std::max(1ul,tp.num_threads_in_pool());
                const double solver_overhead_time = duration_cast<nanoseconds>(get_next_x_runtime).count();
                max_solver_overhead_time = std::max(max_solver_overhead_time, solver_overhead_time);
                // Don't start thinking about the logic below until we have at least 5 objective
                // function samples for each objective function.  This way we have a decent idea how
                // fast these things are.  The solver overhead is really small initially so none of
                // the stuff below really matters in the beginning anyway.
                if (objective_funct_eval_time.current_n() > functions.size()*5) 
                {
                    // If calling opt.get_next_x() is taking a long time relative to how long it takes
                    // to evaluate the objective function then we should spend less time grinding on the
                    // internal details of the optimizer and more time running the actual objective
                    // function.  E.g. if we could just run 2x more objective function calls in the same
                    // amount of time then we should just do that.  The main slowness in the solver is
                    // from the Monte Carlo sampling, which we can turn down if the objective function
                    // is really fast to evaluate.  This is because the point of the Monte Carlo part is
                    // to try really hard to avoid calls to really expensive objective functions.  But
                    // if the objective function is not expensive then we should just call it.
                    if (obj_funct_time < solver_overhead_time) 
                    {
                        // Reduce the amount of Monte Carlo sampling we do.  If it goes low enough
                        // we will disable it altogether.
                        const size_t new_val = static_cast<size_t>(std::floor(opt.get_monte_carlo_upper_bound_sample_num()*0.8));
                        opt.set_monte_carlo_upper_bound_sample_num(std::max<size_t>(1, new_val));
                        // At this point just disable the upper bounding Monte Carlo search stuff and
                        // use only pure random search since the objective function is super cheap to
                        // evaluate, making this more fancy search a waste of time.
                        if (opt.get_monte_carlo_upper_bound_sample_num() == 1) 
                        {
                            opt.set_pure_random_search_probability(1);
                        }
                    } else if (obj_funct_time > 1.5*max_solver_overhead_time) // Consider reenabling
                    {
                        // The Monte Carlo overhead grows over time as the solver accumulates more
                        // information about the objective function.  So we only want to reenable it
                        // or make it bigger if the objective function really is more expensive.  So
                        // we compare to the max solver runtime we have seen so far. If the
                        // objective function has suddenly gotten more expensive then we start to
                        // turn the Monte Carlo modeling back on.
                        const size_t new_val = static_cast<size_t>(std::ceil(opt.get_monte_carlo_upper_bound_sample_num()*1.28));
                        opt.set_monte_carlo_upper_bound_sample_num(std::min<size_t>(5000, new_val));
                        // Set this back to its default value.
                        opt.set_pure_random_search_probability(0.02);
                    }
                }
            }
            tp.wait_for_all_tasks();


            matrix<double,0,1> x;
            double y;
            size_t function_idx;
            opt.get_best_function_eval(x,y,function_idx);
            // Undo any log-scaling that was applied to the variables before we output them. 
            for (long j = 0; j < x.size(); ++j)
            {
                if (log_scale[function_idx][j])
                    x(j) = std::exp(x(j));
            }
            return std::make_pair(function_idx, function_evaluation(x,y/ymult));
        }

        template <
            typename funct
            >
        std::pair<size_t,function_evaluation> find_max_global (
            std::vector<funct>& functions,
            std::vector<function_spec> specs,
            const max_function_calls num,
            const std::chrono::nanoseconds max_runtime,
            double solver_epsilon,
            double ymult,
            const std::vector<std::vector<function_evaluation>>& initial_function_evals
        ) 
        {
            // disabled, don't use any threads
            thread_pool tp(0);

            return find_max_global(tp, functions, std::move(specs), num, max_runtime, solver_epsilon, ymult, initial_function_evals);
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_max_global (
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals = {}
    ) 
    {
        return impl::find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon, +1, initial_function_evals);
    }

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_min_global (
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals = {}
    ) 
    {
        return impl::find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon, -1, initial_function_evals);
    }

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_max_global (
        thread_pool& tp,
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals = {}
    ) 
    {
        return impl::find_max_global(tp, functions, std::move(specs), num, max_runtime, solver_epsilon, +1, initial_function_evals);
    }

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_min_global (
        thread_pool& tp,
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals = {}
    ) 
    {
        return impl::find_max_global(tp, functions, std::move(specs), num, max_runtime, solver_epsilon, -1, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon, {initial_function_evals}).second;
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_min_global(functions, std::move(specs), num, max_runtime, solver_epsilon, {initial_function_evals}).second;
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_max_global(tp, functions, std::move(specs), num, max_runtime, solver_epsilon, {initial_function_evals}).second;
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_min_global(tp, functions, std::move(specs), num, max_runtime, solver_epsilon, {initial_function_evals}).second;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    )
    {
        return find_max_global(std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    )
    {
        return find_min_global(std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    )
    {
        return find_max_global(tp, std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    )
    {
        return find_min_global(tp, std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        double solver_epsilon,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_max_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_max_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_max_global(tp, std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        thread_pool& tp,
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0,
        const std::vector<function_evaluation>& initial_function_evals = {}
    ) 
    {
        return find_min_global(tp, std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon, initial_function_evals);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FiND_GLOBAL_MAXIMUM_hH_

