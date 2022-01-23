// Copyright (C) 2009  Davis E. King (davis@dlib.net)
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

    struct can_not_copy: noncopyable {};
    void serialize(const can_not_copy&, std::ostream&) {}
    void deserialize(can_not_copy&, std::istream&) {}

    void swap(can_not_copy&, can_not_copy&) {}

    class test
    {

    private:

        enum kind
        {
            FLOAT, DOUBLE, CHAR, STRING, NONE
        };

        void operator() (float val)
        {
            DLIB_TEST(val == f_val);
            last_kind = FLOAT;
        }

        void operator() (double val)
        {
            DLIB_TEST(val == d_val);
            last_kind = DOUBLE;
        }

        void operator() (char val)
        {
            DLIB_TEST(val == c_val);
            last_kind = CHAR;
        }

        void operator()(std::string& val)
        {
            DLIB_TEST(val == s_val);
            last_kind = STRING;
        }

        void operator()(const std::string& val)
        {
            DLIB_TEST(val == s_val);
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
            static_assert(tsu::get_type_id<float>() == 1,       "bad type id");
            static_assert(tsu::get_type_id<double>() == 2,      "bad type id");
            static_assert(tsu::get_type_id<char>() == 3,        "bad type id");
            static_assert(tsu::get_type_id<std::string>() == 4, "bad type id");
            static_assert(tsu::get_type_id<long>() == -1,       "This should be -1");


            DLIB_TEST(a.is_empty() == true);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(a.contains<double>() == false);
            DLIB_TEST(a.contains<std::string>() == false);
            DLIB_TEST(a.contains<long>() == false);

            DLIB_TEST(a.get_type_id<int>() == -1);
            DLIB_TEST(a.get_type_id<float>() == 1);
            DLIB_TEST(a.get_type_id<double>() == 2);
            DLIB_TEST(a.get_type_id<char>() == 3);
            DLIB_TEST(a.get_type_id<std::string>() == 4);
            DLIB_TEST(a.get_type_id<tsu>() == -1);


            f_val = 4.345f;
            a.get<float>() = f_val;
            DLIB_TEST(a.cast_to<float>() == f_val);
            DLIB_TEST(const_cast<const tsu&>(a).cast_to<float>() == f_val);
            bool exception_thrown = false;
            try {a.cast_to<char>(); }
            catch (bad_type_safe_union_cast&) { exception_thrown = true;}
            DLIB_TEST(exception_thrown);


            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(a.contains<float>() == true);
            DLIB_TEST(a.contains<double>() == false);
            DLIB_TEST(a.contains<std::string>() == false);
            DLIB_TEST(a.contains<long>() == false);


            last_kind = NONE;
            const_cast<const tsu&>(a).apply_to_contents(*this);
            DLIB_TEST(last_kind == FLOAT);

        // -----------

            d_val = 4.345;
            a.get<double>() = d_val;
            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_TEST(last_kind == DOUBLE);

        // -----------

            c_val = 'a';
            a.get<char>() = c_val;
            last_kind = NONE;
            const_cast<const tsu&>(a).apply_to_contents(*this);
            DLIB_TEST(last_kind == CHAR);

        // -----------

            s_val = "test string";
            a.get<std::string>() = s_val;
            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_TEST(last_kind == STRING);

            DLIB_TEST(a.cast_to<std::string>() == s_val);
            exception_thrown = false;
            try {a.cast_to<float>(); }
            catch (bad_type_safe_union_cast&) { exception_thrown = true;}
            DLIB_TEST(exception_thrown);

        // -----------
            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(a.contains<double>() == false);
            DLIB_TEST(a.contains<std::string>() == true);
            DLIB_TEST(a.contains<long>() == false);
        // -----------

            a.swap(b);

            DLIB_TEST(a.is_empty() == true);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(a.contains<double>() == false);
            DLIB_TEST(a.contains<std::string>() == false);
            DLIB_TEST(a.contains<long>() == false);

            DLIB_TEST(b.is_empty() == false);
            DLIB_TEST(b.contains<char>() == false);
            DLIB_TEST(b.contains<float>() == false);
            DLIB_TEST(b.contains<double>() == false);
            DLIB_TEST(b.contains<std::string>() == true);
            DLIB_TEST(b.contains<long>() == false);


            last_kind = NONE;
            b.apply_to_contents(*this);
            DLIB_TEST(last_kind == STRING);

        // -----------

            b.swap(a);

            DLIB_TEST(b.is_empty() == true);
            DLIB_TEST(b.contains<char>() == false);
            DLIB_TEST(b.contains<float>() == false);
            DLIB_TEST(b.contains<double>() == false);
            DLIB_TEST(b.contains<std::string>() == false);
            DLIB_TEST(b.contains<long>() == false);

            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(a.contains<double>() == false);
            DLIB_TEST(a.contains<std::string>() == true);
            DLIB_TEST(a.contains<long>() == false);


            last_kind = NONE;
            a.apply_to_contents(*this);
            DLIB_TEST(last_kind == STRING);
            last_kind = NONE;
            b.apply_to_contents(*this);
            DLIB_TEST(last_kind == NONE);


            a.get<char>() = 'a';
            b.get<char>() = 'b';

            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == true);
            DLIB_TEST(b.is_empty() == false);
            DLIB_TEST(b.contains<char>() == true);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(b.contains<float>() == false);


            DLIB_TEST(a.get<char>() == 'a');
            DLIB_TEST(b.get<char>() == 'b');

            swap(a,b);


            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == true);
            DLIB_TEST(b.is_empty() == false);
            DLIB_TEST(b.contains<char>() == true);
            DLIB_TEST(a.contains<float>() == false);
            DLIB_TEST(b.contains<float>() == false);

            DLIB_TEST(a.get<char>() == 'b');
            DLIB_TEST(b.get<char>() == 'a');

        // -----------

            a.get<char>() = 'a';
            b.get<std::string>() = "a string";

            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == true);
            DLIB_TEST(b.is_empty() == false);
            DLIB_TEST(b.contains<char>() == false);
            DLIB_TEST(a.contains<std::string>() == false);
            DLIB_TEST(b.contains<std::string>() == true);


            DLIB_TEST(a.get<char>() == 'a');
            DLIB_TEST(b.get<std::string>() == "a string");

            swap(a,b);

            DLIB_TEST(b.is_empty() == false);
            DLIB_TEST(b.contains<char>() == true);
            DLIB_TEST(a.is_empty() == false);
            DLIB_TEST(a.contains<char>() == false);
            DLIB_TEST(b.contains<std::string>() == false);
            DLIB_TEST(a.contains<std::string>() == true);


            DLIB_TEST(b.get<char>() == 'a');
            DLIB_TEST(a.get<std::string>() == "a string");




            {
                type_safe_union<char, float, std::string> a, b, empty_union;

                ostringstream sout;
                istringstream sin;

                a.get<char>() = 'd';

                serialize(a, sout);

                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<char>() == true);
                DLIB_TEST(b.get<char>() == 'd');

                DLIB_TEST(a.contains<int>() == false);
                DLIB_TEST(a.contains<float>() == false);
                DLIB_TEST(a.contains<char>() == true);
                DLIB_TEST(a.get<char>() == 'd');

                sin.clear();
                sout.clear();
                sout.str("");

                a.get<std::string>() = "davis";

                serialize(a, sout);
                sin.str(sout.str());
                deserialize(b, sin);


                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<std::string>() == true);
                DLIB_TEST(b.get<std::string>() == "davis");

                sin.clear();
                sout.clear();
                sout.str("");

                serialize(empty_union, sout);
                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.is_empty() == true);

            }

            {
                type_safe_union<char, float, std::string> a, b, empty_union;

                ostringstream sout;
                istringstream sin;

                a = 'd';

                serialize(a, sout);

                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<char>() == true);
                DLIB_TEST(b.get<char>() == 'd');

                DLIB_TEST(a.contains<int>() == false);
                DLIB_TEST(a.contains<float>() == false);
                DLIB_TEST(a.contains<char>() == true);
                DLIB_TEST(a.get<char>() == 'd');

                sin.clear();
                sout.clear();
                sout.str("");

                a = std::string("davis");

                serialize(a, sout);
                sin.str(sout.str());
                deserialize(b, sin);


                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<std::string>() == true);
                DLIB_TEST(b.get<std::string>() == "davis");

                sin.clear();
                sout.clear();
                sout.str("");

                serialize(empty_union, sout);
                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.is_empty() == true);

            }

            {
                typedef type_safe_union<char, float, std::string, can_not_copy> tsu_type;
                tsu_type a('d'), aa(std::string("davis")), b, empty_union;

                ostringstream sout;
                istringstream sin;


                serialize(a, sout);

                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<char>() == true);
                DLIB_TEST(b.get<char>() == 'd');

                DLIB_TEST(a.contains<int>() == false);
                DLIB_TEST(a.contains<float>() == false);
                DLIB_TEST(a.contains<char>() == true);
                DLIB_TEST(a.get<char>() == 'd');

                DLIB_TEST(aa.contains<int>() == false);
                DLIB_TEST(aa.contains<float>() == false);
                DLIB_TEST(aa.contains<char>() == false);
                DLIB_TEST(aa.contains<std::string>() == true);

                sin.clear();
                sout.clear();
                sout.str("");


                serialize(aa, sout);
                sin.str(sout.str());
                deserialize(b, sin);


                DLIB_TEST(b.contains<int>() == false);
                DLIB_TEST(b.contains<float>() == false);
                DLIB_TEST(b.contains<std::string>() == true);
                DLIB_TEST(b.get<std::string>() == "davis");

                sin.clear();
                sout.clear();
                sout.str("");

                serialize(empty_union, sout);
                sin.str(sout.str());
                deserialize(b, sin);

                DLIB_TEST(b.is_empty() == true);

                a.get<can_not_copy>();
                DLIB_TEST(a.contains<can_not_copy>() == true);

            }

            {
                type_safe_union<int,std::string> a, b;
                a = std::string("asdf");
                b = 3;
                b = std::move(a);

                DLIB_TEST(b.get<std::string>() == "asdf");
            }

            {
                type_safe_union<int,std::string> a = 3;
                type_safe_union<int,std::string> b = std::string("asdf");

                DLIB_TEST(a.get<int>() == 3);
                DLIB_TEST(b.get<std::string>() == "asdf");
            }

            {
                using ptr_t = std::unique_ptr<std::string>;
                type_safe_union<int, ptr_t> a;
                type_safe_union<int, ptr_t> b = ptr_t(new std::string("asdf"));

                a = std::move(b);

                DLIB_TEST(a.contains<ptr_t>());
                DLIB_TEST(!b.contains<ptr_t>());
                DLIB_TEST(*a.get<ptr_t>() == "asdf");

                swap(a,b);

                DLIB_TEST(b.contains<ptr_t>());
                DLIB_TEST(!a.contains<ptr_t>());
                DLIB_TEST(*b.get<ptr_t>() == "asdf");
            }

            {
                //testing copy semantics and move semantics

                struct mytype
                {
                    mytype(int i_ = 0) : i(i_) {}

                    mytype(const mytype& other) : i(other.i) {}
                    mytype& operator=(const mytype& other) {i = other.i; return *this;}

                    mytype(mytype&& other) : i(other.i) {other.i = 0;}
                    mytype& operator=(mytype&& other) {i = other.i ; other.i = 0; return *this;}

                    int i = 0;
                };

                using tsu = type_safe_union<int,mytype>;

                {
                    mytype a(10);
                    tsu ta(a); //copy constructor
                    DLIB_TEST(a.i == 10);
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                }

                {
                    mytype a(10);
                    tsu ta;
                    ta = a; //copy assign
                    DLIB_TEST(a.i == 10);
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                }

                {
                    mytype a(10);
                    tsu ta(std::move(a)); //move constructor
                    DLIB_TEST(a.i == 0);
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                }

                {
                    mytype a(10);
                    tsu ta;
                    ta = std::move(a); //move assign
                    DLIB_TEST(a.i == 0);
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                }

                {
                    tsu ta(mytype(10));
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    tsu tb(ta); //copy constructor
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    DLIB_TEST(tb.cast_to<mytype>().i == 10);
                }

                {
                    tsu ta(mytype(10));
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    tsu tb;
                    tb = ta; //copy assign
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    DLIB_TEST(tb.cast_to<mytype>().i == 10);
                }

                {
                    tsu ta(mytype(10));
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    tsu tb(std::move(ta)); //move constructor
                    DLIB_TEST(ta.is_empty());
                    DLIB_TEST(tb.cast_to<mytype>().i == 10);
                }

                {
                    tsu ta(mytype(10));
                    DLIB_TEST(ta.cast_to<mytype>().i == 10);
                    tsu tb;
                    tb = std::move(ta); //move assign
                    DLIB_TEST(ta.is_empty());
                    DLIB_TEST(tb.cast_to<mytype>().i == 10);
                }
            }

            {
                //testing emplace(), copy semantics, move semantics, swap, overloaded, and new visitor
                type_safe_union<int, float, std::string> a, b;
                a.emplace<std::string>("hello world");

                DLIB_TEST(a.contains<std::string>());
                b = a; //copy
                DLIB_TEST(a.contains<std::string>());
                DLIB_TEST(b.contains<std::string>());
                DLIB_TEST(a.cast_to<std::string>() == "hello world");
                DLIB_TEST(b.cast_to<std::string>() == "hello world");
                a = 1;
                DLIB_TEST(a.contains<int>());
                DLIB_TEST(a.cast_to<int>() == 1);
                b = std::move(a);
                DLIB_TEST(b.contains<int>());
                DLIB_TEST(b.cast_to<int>() == 1);
                DLIB_TEST(a.is_empty());
                DLIB_TEST(a.get_current_type_id() == 0);
                swap(a, b);
                DLIB_TEST(a.contains<int>());
                DLIB_TEST(a.cast_to<int>() == 1);
                DLIB_TEST(b.is_empty());
                DLIB_TEST(b.get_current_type_id() == 0);
                //visit can return non-void types
                auto ret = visit(overloaded(
                    [](int) {
                        return std::string("int");
                    },
                    [](float) {
                        return std::string("float");
                    },
                    [](const std::string&) {
                        return std::string("std::string");
                    }
                ), a);
                static_assert(std::is_same<std::string, decltype(ret)>::value, "bad return type");
                DLIB_TEST(ret == "int");
                //apply_to_contents can only return void
                a = std::string("hello there!");
                std::string str;
                a.apply_to_contents(overloaded(
                    [&str](int) {
                        str = std::string("int");
                    },
                    [&str](float) {
                        str = std::string("float");
                    },
                    [&str](const std::string& item) {
                        str = item;
                    }
                ));
                DLIB_TEST(str == "hello there!");
            }

            {
                //nested unions
                using tsu_a = type_safe_union<int,float,std::string>;
                using tsu_b = type_safe_union<int,float,std::string,tsu_a>;

                tsu_b object(dlib::in_place_tag<tsu_a>{}, std::string("hello from bottom node"));
                DLIB_TEST(object.contains<tsu_a>());
                DLIB_TEST(object.get<tsu_a>().get<std::string>() == "hello from bottom node");
                auto ret = visit(overloaded(
                    [](int) {
                        return std::string("int");
                    },
                    [](float) {
                        return std::string("float");
                    },
                    [](std::string) {
                        return std::string("std::string");
                    },
                    [](const tsu_a& item) {
                        return visit( overloaded(
                            [](int) {
                                return std::string("nested int");
                            },
                            [](float) {
                                return std::string("nested float");
                            },
                            [](std::string str) {
                                return str;
                            }
                        ), item);
                    }
                ), object);
                static_assert(std::is_same<std::string, decltype(ret)>::value, "bad type");
                DLIB_TEST(ret == "hello from bottom node");
            }

            {
                //struct visitor
                using tsu = type_safe_union<int,float,std::string>;

                struct visitor_private
                {
                    std::string operator()(int)
                    {
                        return std::string("int");
                    }

                    std::string operator()(float)
                    {
                        return std::string("float");
                    }

                    std::string operator()(const std::string& str)
                    {
                        return str;
                    }
                };

                visitor_private visitor;
                tsu a = std::string("hello from private visitor");
                auto ret = visit(visitor, a);
                static_assert(std::is_same<std::string, decltype(ret)>::value, "bad type");
                DLIB_TEST(ret == "hello from private visitor");
            }
        }
    };

    namespace test_for_each_1
    {
        /*! Local classes aren't allowed to have template member functions... !*/

        using tsu = type_safe_union<int,float,std::string>;
        static_assert(type_safe_union_size<tsu>::value == 3, "bad number of types");
        static_assert(std::is_same<type_safe_union_alternative_t<0, tsu>, int>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<1, tsu>, float>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<2, tsu>, std::string>::value, "bad type");

        static_assert(std::is_same<type_safe_union_alternative_t<0, const tsu>, const int>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<1, const tsu>, const float>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<2, const tsu>, const std::string>::value, "bad type");

        static_assert(std::is_same<type_safe_union_alternative_t<0, volatile tsu>, volatile int>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<1, volatile tsu>, volatile float>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<2, volatile tsu>, volatile std::string>::value, "bad type");

        static_assert(std::is_same<type_safe_union_alternative_t<0, const volatile tsu>, const volatile int>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<1, const volatile tsu>, const volatile float>::value, "bad type");
        static_assert(std::is_same<type_safe_union_alternative_t<2, const volatile tsu>, const volatile std::string>::value, "bad type");

        struct for_each_visitor
        {
            std::vector<int> type_indices;

            template<typename T>
            void operator()(dlib::in_place_tag<T>, const tsu& item)
            {
                type_indices.push_back(item.get_type_id<T>());
            }
        };

        void test()
        {
            tsu a;
            for_each_visitor visitor;
            for_each_type(visitor, a);
            
            DLIB_TEST(visitor.type_indices.size() == 3);
            DLIB_TEST(visitor.type_indices[0] == 1);
            DLIB_TEST(visitor.type_indices[1] == 2);
            DLIB_TEST(visitor.type_indices[2] == 3);
        }
    }

    namespace test_for_each_2
    {
        /*! Local classes aren't allowed to have template member functions... !*/
        
        using tsu = type_safe_union<int,float,std::string>;

        //for_each() that demonstrates an actual use-case. 
        //Instead of something simple like a target index, you might want to set the variant
        //based on some specific state that is unique to one of the alternative types.
        //For example, you might want to conditionally set the variant based on hashes.

        struct for_each_visitor
        {
            for_each_visitor(int target_index_) : target_index(target_index_) {}

            template<typename TagType>
            void operator()(TagType tag, tsu& item)
            {
                if (item.get_type_id<TagType>() == target_index)
                    item = tsu{tag};
            }

            const int target_index = 0;
        };

        void test()
        {
            tsu a;
            for_each_type(for_each_visitor{1}, a);
            DLIB_TEST(a.contains<int>());
            a.clear();
            for_each_type(for_each_visitor{2}, a);
            DLIB_TEST(a.contains<float>());
            a.clear();
            for_each_type(for_each_visitor{3}, a);
            DLIB_TEST(a.contains<std::string>());
            a.clear();
            for_each_type(for_each_visitor{-1}, a);
            DLIB_TEST(a.is_empty());
            a.clear();
            for_each_type(for_each_visitor{215465}, a);
            DLIB_TEST(a.is_empty());
        }
    }

    namespace test_for_each_3
    {
        using tsu1 = type_safe_union<int,float,std::string>;
        using tsu2 = type_safe_union<std::string,long,char>;

        struct serializer_typeid
        {
            serializer_typeid(std::ostream& out_) : out(out_) {}

            template<typename T>
            void operator()(const T& x)
            {
                dlib::serialize(typeid(T).hash_code(), out);
                dlib::serialize(x, out);
            }

            std::ostream& out;
        };

        struct deserializer_typeid
        {
            deserializer_typeid(std::istream& in_) : in(in_) 
            {
                dlib::deserialize(hash_code, in);
            }

            template<typename T, typename TSU>
            void operator()(in_place_tag<T>, TSU&& me)
            {
                if (typeid(T).hash_code() == hash_code)
                    dlib::deserialize(me.template get<T>(), in);
            }

            std::size_t hash_code = 0;
            std::istream& in;
        };

        void test()
        {
            tsu1 a;
            a.get<std::string>() = "hello from tsu1";

            std::stringstream out;
            visit(serializer_typeid(out), a);

            tsu2 b;
            for_each_type(deserializer_typeid(out), b);

            DLIB_TEST(b.contains<std::string>());
            DLIB_TEST(b.get<std::string>() == "hello from tsu1");
        }
    }


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
                test_for_each_1::test();
                test_for_each_2::test();
                test_for_each_3::test();
            }
        }
    } a;

}




