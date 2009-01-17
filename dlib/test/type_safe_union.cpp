// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/type_safe_union.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.type_safe_union");

    class test
    {

    private:

        enum kind
        {
            FLOAT, DOUBLE, CHAR, STRING, NONE
        };

        void operator() (float val)
        {
            DLIB_CASSERT(val == f_val,"");
            last_kind = FLOAT;
        }

        void operator() (double val)
        {
            DLIB_CASSERT(val == d_val,"");
            last_kind = DOUBLE;
        }

        void operator() (char val)
        {
            DLIB_CASSERT(val == c_val,"");
            last_kind = CHAR;
        }

        void operator()(std::string& val)
        {
            DLIB_CASSERT(val == s_val,"");
            last_kind = STRING;
        }

    // ------------------------------

        friend class type_safe_union<float, double, char, std::string>;
        typedef      type_safe_union<float, double, char, std::string> tsu;
        tsu a, b, c;

        float f_val;
        double d_val;
        char c_val;
        std::string s_val;

        kind last_kind;

    public:
        void test_stuff()
        {
            DLIB_CASSERT(a.is_empty() == true,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(a.contains<double>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == false,"");
            DLIB_CASSERT(a.contains<long>() == false,"");



            f_val = 4.345f;
            a.get<float>() = f_val;

            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<float>() == true,"");
            DLIB_CASSERT(a.contains<double>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == false,"");
            DLIB_CASSERT(a.contains<long>() == false,"");


            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == FLOAT,"");

        // -----------

            d_val = 4.345;
            a.get<double>() = d_val;
            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == DOUBLE,"");

        // -----------

            c_val = 'a';
            a.get<char>() = c_val;
            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == CHAR,"");

        // -----------

            s_val = "test string";
            a.get<std::string>() = s_val;
            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == STRING,"");

        // -----------
            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(a.contains<double>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == true,"");
            DLIB_CASSERT(a.contains<long>() == false,"");
        // -----------

            a.swap(b);

            DLIB_CASSERT(a.is_empty() == true,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(a.contains<double>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == false,"");
            DLIB_CASSERT(a.contains<long>() == false,"");

            DLIB_CASSERT(b.is_empty() == false,"");
            DLIB_CASSERT(b.contains<char>() == false,"");
            DLIB_CASSERT(b.contains<float>() == false,"");
            DLIB_CASSERT(b.contains<double>() == false,"");
            DLIB_CASSERT(b.contains<std::string>() == true,"");
            DLIB_CASSERT(b.contains<long>() == false,"");


            last_kind = NONE;
            b.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == STRING,"");

        // -----------

            b.swap(a);

            DLIB_CASSERT(b.is_empty() == true,"");
            DLIB_CASSERT(b.contains<char>() == false,"");
            DLIB_CASSERT(b.contains<float>() == false,"");
            DLIB_CASSERT(b.contains<double>() == false,"");
            DLIB_CASSERT(b.contains<std::string>() == false,"");
            DLIB_CASSERT(b.contains<long>() == false,"");

            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(a.contains<double>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == true,"");
            DLIB_CASSERT(a.contains<long>() == false,"");


            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == STRING,"");
            last_kind = NONE;
            b.apply_to_contents(*this);
            DLIB_CASSERT(last_kind == NONE,"");


            a.get<char>() = 'a';
            b.get<char>() = 'b';

            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == true,"");
            DLIB_CASSERT(b.is_empty() == false,"");
            DLIB_CASSERT(b.contains<char>() == true,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(b.contains<float>() == false,"");


            DLIB_CASSERT(a.get<char>() == 'a',"");
            DLIB_CASSERT(b.get<char>() == 'b',"");

            swap(a,b);


            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == true,"");
            DLIB_CASSERT(b.is_empty() == false,"");
            DLIB_CASSERT(b.contains<char>() == true,"");
            DLIB_CASSERT(a.contains<float>() == false,"");
            DLIB_CASSERT(b.contains<float>() == false,"");

            DLIB_CASSERT(a.get<char>() == 'b',"");
            DLIB_CASSERT(b.get<char>() == 'a',"");

        // -----------

            a.get<char>() = 'a';
            b.get<std::string>() = "a string";

            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == true,"");
            DLIB_CASSERT(b.is_empty() == false,"");
            DLIB_CASSERT(b.contains<char>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == false,"");
            DLIB_CASSERT(b.contains<std::string>() == true,"");


            DLIB_CASSERT(a.get<char>() == 'a',"");
            DLIB_CASSERT(b.get<std::string>() == "a string","");

            swap(a,b);

            DLIB_CASSERT(b.is_empty() == false,"");
            DLIB_CASSERT(b.contains<char>() == true,"");
            DLIB_CASSERT(a.is_empty() == false,"");
            DLIB_CASSERT(a.contains<char>() == false,"");
            DLIB_CASSERT(b.contains<std::string>() == false,"");
            DLIB_CASSERT(a.contains<std::string>() == true,"");


            DLIB_CASSERT(b.get<char>() == 'a',"");
            DLIB_CASSERT(a.get<std::string>() == "a string","");



        }

    };



    class type_safe_union_tester : public tester
    {
    public:
        type_safe_union_tester (
        ) :
            tester ("test_type_safe_union",
                    "Runs tests on the type_safe_union object")
        {}

        void perform_test (
        )
        {
            for (int i = 0; i < 10; ++i)
            {
                test a;
                a.test_stuff();
            }
        }
    } a;

}




