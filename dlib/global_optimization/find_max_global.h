// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FiND_GLOBAL_MAXIMUM_hH_
#define DLIB_FiND_GLOBAL_MAXIMUM_hH_

#include "find_max_global_abstract.h"
#include "global_function_search.h"
#include "../metaprogramming.h"
#include <utility>
#include <chrono>

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
            std::vector<funct>& functions,
            std::vector<function_spec> specs,
            const max_function_calls num,
            const std::chrono::nanoseconds max_runtime,
            double solver_epsilon,
            double ymult
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

            global_function_search opt(specs);
            opt.set_solver_epsilon(solver_epsilon);

            const auto time_to_stop = std::chrono::steady_clock::now() + max_runtime;

            // Now run the main solver loop.
            for (size_t i = 0; i < num.max_calls && std::chrono::steady_clock::now() < time_to_stop; ++i)
            {
                auto next = opt.get_next_x();
                matrix<double,0,1> x = next.x();
                // Undo any log-scaling that was applied to the variables before we pass them
                // to the functions being optimized.
                for (long j = 0; j < x.size(); ++j)
                {
                    if (log_scale[next.function_idx()][j])
                        x(j) = std::exp(x(j));
                }
                double y = ymult*call_function_and_expand_args(functions[next.function_idx()], x);
                next.set(y);
            }


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
        double solver_epsilon = 0
    ) 
    {
        return impl::find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon, +1);
    }

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_min_global (
        std::vector<funct>& functions,
        std::vector<function_spec> specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 0
    ) 
    {
        return impl::find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon, -1);
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
        double solver_epsilon = 0
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_max_global(functions, std::move(specs), num, max_runtime, solver_epsilon).second;
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
        double solver_epsilon = 0
    ) 
    {
        std::vector<funct> functions(1,std::move(f));
        std::vector<function_spec> specs(1, function_spec(bound1, bound2, is_integer_variable));
        return find_min_global(functions, std::move(specs), num, max_runtime, solver_epsilon).second;
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
        double solver_epsilon 
    )
    {
        return find_max_global(std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon);
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
        double solver_epsilon 
    )
    {
        return find_min_global(std::move(f), bound1, bound2, is_integer_variable, num, FOREVER, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon);
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
        double solver_epsilon
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        double solver_epsilon
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, FOREVER, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_max_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_min_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon);
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
        double solver_epsilon 
    ) 
    {
        return find_max_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        double solver_epsilon 
    ) 
    {
        return find_min_global(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, FOREVER, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_min_global (
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 0
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_max_global(std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon);
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
        double solver_epsilon = 0
    ) 
    {
        return find_min_global(std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FiND_GLOBAL_MAXIMUM_hH_

