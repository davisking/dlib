// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <memory>
#include <iostream>

using namespace std;

class testme
{
public:

    testme(testme&&) = default;
    testme(const testme&) = delete;


    template <typename T>
    auto auto_return(T f) -> decltype(f(4)) { return f(4); }

    template <typename T>
    auto auto_return(T f) -> decltype(f()) { return f(); }

    static int returnint() { return 0; }

    void dostuff()
    {
        thread_local int stuff1 = 999;
        auto x = 4;
        ++stuff1;

        decltype(x) asdf = 9;
        ++asdf;

        auto f = []() { cout << "in a lambda!" << endl; };
        f();

        auto_return(returnint);
    }

    template <typename ...T>
    void variadic_template(
        T&& ...args
    )
    {
    }



    std::shared_ptr<int> asdf;
};

// ------------------------------------------------------------------------------------

