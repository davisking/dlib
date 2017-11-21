// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FiND_GLOBAL_MAXIMUM_hH_
#define DLIB_FiND_GLOBAL_MAXIMUM_hH_

#include "global_function_search.h"

#include "../metaprogramming.h"
#include <utility>

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
                "You invoked dlib::call_with_vect(f,a) but the number of arguments expected by f() doesn't match the size of 'a'. "
                << "Expected " << sizeof...(indices) << " arguments but got " << a.size() << "."
            );  
            return f(a(indices-1)...); 
        }

        // Visual studio, as of November 2017, doesn't support C++11 and can't compile this code.  
        // So we write the terrible garbage in the #else for visual studio.  When Visual Studio supports C++11 I'll update this #ifdef to use the C++11 code.
#ifndef _MSC_VER 
        template <size_t max_unpack>
        struct call_with_vect
        {
            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(_cwv(std::forward<T>(f),a,typename make_compile_time_integer_range<max_unpack>::type()))
            {
                return _cwv(std::forward<T>(f),a,typename make_compile_time_integer_range<max_unpack>::type());
            }

            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(call_with_vect<max_unpack-1>::template go(std::forward<T>(f),a))
            {
                return call_with_vect<max_unpack-1>::go(std::forward<T>(f),a);
            }
        };

        template <>
        struct call_with_vect<0>
        {
            template <typename T>
            static auto go(T&& f, const matrix<double,0,1>& a) -> decltype(f(disable_decay_to_scalar(a)))
            {
                return f(disable_decay_to_scalar(a));
            }
        };
#else
        template <size_t max_unpack>
        struct call_with_vect
        {         
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), disable_decay_to_scalar(a))  {return f(disable_decay_to_scalar(a));   }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0)) { DLIB_CASSERT(a.size() == 1); return f(a(0)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0),a(1)) { DLIB_CASSERT(a.size() == 2); return f(a(0),a(1)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0), a(1), a(2)) { DLIB_CASSERT(a.size() == 3); return f(a(0), a(1),a(2)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0), a(1), a(2), a(3)) { DLIB_CASSERT(a.size() == 4); return f(a(0), a(1), a(2), a(3)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0), a(1), a(2), a(3), a(4)) { DLIB_CASSERT(a.size() == 5); return f(a(0), a(1), a(2), a(3), a(4)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0), a(1), a(2), a(3), a(4), a(5)) { DLIB_CASSERT(a.size() == 6); return f(a(0), a(1), a(2), a(3), a(4), a(5)); }
template <typename T> static auto go(T&& f, const matrix<double, 0, 1>& a) -> decltype(std::forward<T>(f), a(0), a(1), a(2), a(3), a(4), a(5), a(6)) { DLIB_CASSERT(a.size() == 7); return f(a(0), a(1), a(2), a(3), a(4), a(5), a(6)); }
        };
#endif
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T> 
    auto call_with_vect(
        T&& f, 
        const matrix<double,0,1>& a
    ) -> decltype(gopt_impl::call_with_vect<40>::go(f,a))
    {
        // unpack up to 40 parameters when calling f()
        return gopt_impl::call_with_vect<40>::go(std::forward<T>(f),a);
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

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_global_maximum (
        std::vector<funct>& functions,
        const std::vector<function_spec>& specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        global_function_search opt(specs);
        opt.set_solver_epsilon(solver_epsilon);

        const auto time_to_stop = std::chrono::steady_clock::now() + max_runtime;

        for (size_t i = 0; i < num.max_calls && std::chrono::steady_clock::now() < time_to_stop; ++i)
        {
            auto next = opt.get_next_x();
            double y = call_with_vect(functions[next.function_idx()], next.x());
            next.set(y);



            // TODO, remove this funky test code
            matrix<double,0,1> x;
            size_t function_idx;
            opt.get_best_function_eval(x,y,function_idx);
            using namespace std;
            cout << "\ni: "<< i << endl;
            cout << "best eval x: "<< trans(x);
            cout << "best eval y: "<< y << endl;
            cout << "best eval function index: "<< function_idx << endl;
            if (std::abs(y  - 21.9210397) < 0.0001)
            {
                cout << "DONE!" << endl;
                //cin.get();
                break;
            }
        }


        matrix<double,0,1> x;
        double y;
        size_t function_idx;
        opt.get_best_function_eval(x,y,function_idx);
        return std::make_pair(function_idx, function_evaluation(x,std::move(y)));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& lower,
        const matrix<double,0,1>& upper,
        const max_function_calls num,
        double solver_epsilon = 1e-11
    ) 
    {
        std::vector<funct> functions(1,f);
        std::vector<function_spec> specs(1, function_spec(lower, upper));
        auto forever = std::chrono::hours(24*356*290);
        return find_global_maximum(functions, specs, num, forever, solver_epsilon).second;
    }

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const double lower,
        const double upper,
        const max_function_calls num,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(f, matrix<double,0,1>({lower}), matrix<double,0,1>({upper}), num, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& lower,
        const matrix<double,0,1>& upper,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        double solver_epsilon = 1e-11
    ) 
    {
        std::vector<funct> functions(1, std::move(f));
        std::vector<function_spec> specs(1, function_spec(lower, upper, is_integer_variable));
        auto forever = std::chrono::hours(24*356*290);
        return find_global_maximum(functions, specs, num, forever, solver_epsilon).second;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& lower,
        const matrix<double,0,1>& upper,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        std::vector<funct> functions(1,f);
        std::vector<function_spec> specs(1, function_spec(lower, upper));
        return find_global_maximum(functions, specs, max_function_calls(), max_runtime, solver_epsilon).second;
    }

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const double lower,
        const double upper,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(f, matrix<double,0,1>({lower}), matrix<double,0,1>({upper}), max_runtime, solver_epsilon);
    }

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& lower,
        const matrix<double,0,1>& upper,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        std::vector<funct> functions(1, std::move(f));
        std::vector<function_spec> specs(1, function_spec(lower, upper, is_integer_variable));
        return find_global_maximum(functions, specs, max_function_calls(), max_runtime, solver_epsilon).second;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FiND_GLOBAL_MAXIMUM_hH_

