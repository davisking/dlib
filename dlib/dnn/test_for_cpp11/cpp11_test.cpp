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

    void dostuff()
    {
        auto x = 4;

        decltype(x) asdf = 9;

        auto f = [](){ cout << "in a lambda!" << endl; };
        f();
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

