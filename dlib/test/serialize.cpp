// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <iostream>
#include <fstream>
#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/serialize.h>
#include <dlib/image_transforms.h>
#include <dlib/rand.h>

#include "tester.h"

namespace  
{

// ----------------------------------------------------------------------------------------

    using namespace test;
    using namespace dlib;
    using namespace std;

    struct test_object
    {
        signed char i1;
        signed short i2;
        signed long i3;
        unsigned char i4;
        unsigned short i5;
        unsigned long i6;
        uint64 i7;
        int64 i8;

        signed char i1_0;
        signed short i2_0;
        signed long i3_0;
        unsigned char i4_0;
        unsigned short i5_0;
        unsigned long i6_0;
        uint64 i7_0;
        int64 i8_0;

        signed char i1_n;
        signed short i2_n;
        signed long i3_n;


        float f1;
        double f2;
        long double f3;
        float f1_inf;
        double f2_inf;
        long double f3_inf;
        float f1_ninf;
        double f2_ninf;
        long double f3_ninf;
        float f1_qnan;
        double f2_qnan;
        long double f3_qnan;
        float f1_snan;
        double f2_snan;
        long double f3_snan;

        std::string s1;
        std::wstring s2;

        int array[10];

        bool b_true;
        bool b_false;


        void set_state_1(
        )
        {
            i1 = 1;
            i2 = 2;
            i3 = 3;
            i4 = 4;
            i5 = 5;
            i6 = 6;
            i7 = 7;
            i8 = 8;

            i1_0 = 0;
            i2_0 = 0;
            i3_0 = 0;
            i4_0 = 0;
            i5_0 = 0;
            i6_0 = 0;
            i7_0 = 0;
            i8_0 = 0;

            i1_n = -1;
            i2_n = -2;
            i3_n = -3;

            f1 = 123.456f;
            f2 = 543.341;
            f3 = 5234234.23;

            f1_inf = numeric_limits<float>::infinity();
            f2_inf = numeric_limits<double>::infinity();
            f3_inf = numeric_limits<long double>::infinity();
            f1_ninf = -numeric_limits<float>::infinity();
            f2_ninf = -numeric_limits<double>::infinity();
            f3_ninf = -numeric_limits<long double>::infinity();
            f1_qnan = numeric_limits<float>::quiet_NaN();
            f2_qnan = numeric_limits<double>::quiet_NaN();
            f3_qnan = numeric_limits<long double>::quiet_NaN();
            f1_snan = numeric_limits<float>::signaling_NaN();
            f2_snan = numeric_limits<double>::signaling_NaN();
            f3_snan = numeric_limits<long double>::signaling_NaN();

            s1 = "davis";
            s2 = L"yo yo yo";

            for (int i = 0; i < 10; ++i)
                array[i] = i; 

            b_true = true;
            b_false = false;
        }

        void set_state_2(
        )
        {
            i1 = 10;
            i2 = 20;
            i3 = 30;
            i4 = 40;
            i5 = 50;
            i6 = 60;
            i7 = 70;
            i8 = 80;

            i1_0 = 5;
            i2_0 = 6;
            i3_0 = 7;
            i4_0 = 8;
            i5_0 = 9;
            i6_0 = 10;
            i7_0 = 11;
            i8_0 = 12;

            i1_n = -13;
            i2_n = -25;
            i3_n = -12;

            f1 = 45.3f;
            f2 = 0.001;
            f3 = 2.332;

            f1_inf = f1;
            f2_inf = f2;
            f3_inf = f3;
            f1_ninf = f1;
            f2_ninf = f2;
            f3_ninf = f3;
            f1_qnan = f1;
            f2_qnan = f2;
            f3_qnan = f3;
            f1_snan = f1;
            f2_snan = f2;
            f3_snan = f3;

            s1 = "";
            s2 = L"";

            for (int i = 0; i < 10; ++i)
                array[i] = 10-i; 

            b_true = false;
            b_false = true;
        }

        void assert_in_state_1 (
        )
        {
            DLIB_TEST (i1 == 1);
            DLIB_TEST (i2 == 2);
            DLIB_TEST (i3 == 3);
            DLIB_TEST (i4 == 4);
            DLIB_TEST (i5 == 5);
            DLIB_TEST (i6 == 6);
            DLIB_TEST (i7 == 7);
            DLIB_TEST (i8 == 8);

            DLIB_TEST (i1_0 == 0);
            DLIB_TEST (i2_0 == 0);
            DLIB_TEST (i3_0 == 0);
            DLIB_TEST (i4_0 == 0);
            DLIB_TEST (i5_0 == 0);
            DLIB_TEST (i6_0 == 0);
            DLIB_TEST (i7_0 == 0);
            DLIB_TEST (i8_0 == 0);

            DLIB_TEST (i1_n == -1);
            DLIB_TEST (i2_n == -2);
            DLIB_TEST (i3_n == -3);

            DLIB_TEST (abs(f1 -123.456) < 1e-5);
            DLIB_TEST (abs(f2 - 543.341) < 1e-10);
            DLIB_TEST (abs(f3 - 5234234.23) < 1e-10);

            DLIB_TEST (f1_inf == numeric_limits<float>::infinity());
            DLIB_TEST (f2_inf == numeric_limits<double>::infinity());
            DLIB_TEST (f3_inf == numeric_limits<long double>::infinity());
            DLIB_TEST (f1_ninf == -numeric_limits<float>::infinity());
            DLIB_TEST (f2_ninf == -numeric_limits<double>::infinity());
            DLIB_TEST (f3_ninf == -numeric_limits<long double>::infinity());
            DLIB_TEST (!(f1_qnan <= numeric_limits<float>::infinity() && f1_qnan >= -numeric_limits<float>::infinity() ));
            DLIB_TEST (!(f2_qnan <= numeric_limits<double>::infinity() && f1_qnan >= -numeric_limits<double>::infinity() ));
            DLIB_TEST (!(f3_qnan <= numeric_limits<long double>::infinity() && f1_qnan >= -numeric_limits<long double>::infinity() ));
            DLIB_TEST (!(f1_snan <= numeric_limits<float>::infinity() && f1_qnan >= -numeric_limits<float>::infinity() ));
            DLIB_TEST (!(f2_snan <= numeric_limits<double>::infinity() && f1_qnan >= -numeric_limits<double>::infinity() ));
            DLIB_TEST (!(f3_snan <= numeric_limits<long double>::infinity() && f1_qnan >= -numeric_limits<long double>::infinity() ));

            DLIB_TEST (s1 == "davis");
            DLIB_TEST (s2 == L"yo yo yo");

            for (int i = 0; i < 10; ++i)
            {
                DLIB_TEST (array[i] == i);
            }

            DLIB_TEST (b_true == true);
            DLIB_TEST (b_false == false);

        }

        void assert_in_state_2 (
        )
        {
            DLIB_TEST (i1 == 10);
            DLIB_TEST (i2 == 20);
            DLIB_TEST (i3 == 30);
            DLIB_TEST (i4 == 40);
            DLIB_TEST (i5 == 50);
            DLIB_TEST (i6 == 60);
            DLIB_TEST (i7 == 70);
            DLIB_TEST (i8 == 80);

            DLIB_TEST (i1_0 == 5);
            DLIB_TEST (i2_0 == 6);
            DLIB_TEST (i3_0 == 7);
            DLIB_TEST (i4_0 == 8);
            DLIB_TEST (i5_0 == 9);
            DLIB_TEST (i6_0 == 10);
            DLIB_TEST (i7_0 == 11);
            DLIB_TEST (i8_0 == 12);

            DLIB_TEST (i1_n == -13);
            DLIB_TEST (i2_n == -25);
            DLIB_TEST (i3_n == -12);

            DLIB_TEST (abs(f1 - 45.3) < 1e-5);
            DLIB_TEST (abs(f2 - 0.001) < 1e-10);
            DLIB_TEST (abs(f3 - 2.332) < 1e-10);
            DLIB_TEST (abs(f1_inf - 45.3) < 1e-5);
            DLIB_TEST (abs(f2_inf - 0.001) < 1e-10);
            DLIB_TEST (abs(f3_inf - 2.332) < 1e-10);
            DLIB_TEST (abs(f1_ninf - 45.3) < 1e-5);
            DLIB_TEST (abs(f2_ninf - 0.001) < 1e-10);
            DLIB_TEST (abs(f3_ninf - 2.332) < 1e-10);
            DLIB_TEST (abs(f1_qnan - 45.3) < 1e-5);
            DLIB_TEST (abs(f2_qnan - 0.001) < 1e-10);
            DLIB_TEST (abs(f3_qnan - 2.332) < 1e-10);
            DLIB_TEST (abs(f1_snan - 45.3) < 1e-5);
            DLIB_TEST (abs(f2_snan - 0.001) < 1e-10);
            DLIB_TEST (abs(f3_snan - 2.332) < 1e-10);

            DLIB_TEST (s1 == "");
            DLIB_TEST (s2 == L"");

            for (int i = 0; i < 10; ++i)
            {
                DLIB_TEST (array[i] == 10-i);
            }

            DLIB_TEST (b_true == false);
            DLIB_TEST (b_false == true);

        }

    };

// ----------------------------------------------------------------------------------------

    void serialize (
        const test_object& item,
        std::ostream& out
    )
    {
        dlib::serialize(item.i1,out);
        dlib::serialize(item.i2,out);
        dlib::serialize(item.i3,out);
        dlib::serialize(item.i4,out);
        dlib::serialize(item.i5,out);
        dlib::serialize(item.i6,out);
        dlib::serialize(item.i7,out);
        dlib::serialize(item.i8,out);

        dlib::serialize(item.i1_0,out);
        dlib::serialize(item.i2_0,out);
        dlib::serialize(item.i3_0,out);
        dlib::serialize(item.i4_0,out);
        dlib::serialize(item.i5_0,out);
        dlib::serialize(item.i6_0,out);
        dlib::serialize(item.i7_0,out);
        dlib::serialize(item.i8_0,out);

        dlib::serialize(item.i1_n,out);
        dlib::serialize(item.i2_n,out);
        dlib::serialize(item.i3_n,out);


        dlib::serialize(item.f1,out);
        dlib::serialize(item.f2,out);
        dlib::serialize(item.f3,out);

        dlib::serialize(item.f1_inf,out);
        dlib::serialize(item.f2_inf,out);
        dlib::serialize(item.f3_inf,out);
        dlib::serialize(item.f1_ninf,out);
        dlib::serialize(item.f2_ninf,out);
        dlib::serialize(item.f3_ninf,out);
        dlib::serialize(item.f1_qnan,out);
        dlib::serialize(item.f2_qnan,out);
        dlib::serialize(item.f3_qnan,out);
        dlib::serialize(item.f1_snan,out);
        dlib::serialize(item.f2_snan,out);
        dlib::serialize(item.f3_snan,out);

        dlib::serialize(item.s1,out);
        dlib::serialize(item.s2,out);

        dlib::serialize(item.array,out);

        dlib::serialize(item.b_true,out);
        dlib::serialize(item.b_false,out);
    }

// ----------------------------------------------------------------------------------------

    void deserialize (
        test_object& item,
        std::istream& in 
    )
    {
        dlib::deserialize(item.i1,in);
        dlib::deserialize(item.i2,in);
        dlib::deserialize(item.i3,in);
        dlib::deserialize(item.i4,in);
        dlib::deserialize(item.i5,in);
        dlib::deserialize(item.i6,in);
        dlib::deserialize(item.i7,in);
        dlib::deserialize(item.i8,in);

        dlib::deserialize(item.i1_0,in);
        dlib::deserialize(item.i2_0,in);
        dlib::deserialize(item.i3_0,in);
        dlib::deserialize(item.i4_0,in);
        dlib::deserialize(item.i5_0,in);
        dlib::deserialize(item.i6_0,in);
        dlib::deserialize(item.i7_0,in);
        dlib::deserialize(item.i8_0,in);

        dlib::deserialize(item.i1_n,in);
        dlib::deserialize(item.i2_n,in);
        dlib::deserialize(item.i3_n,in);


        dlib::deserialize(item.f1,in);
        dlib::deserialize(item.f2,in);
        dlib::deserialize(item.f3,in);

        dlib::deserialize(item.f1_inf,in);
        dlib::deserialize(item.f2_inf,in);
        dlib::deserialize(item.f3_inf,in);
        dlib::deserialize(item.f1_ninf,in);
        dlib::deserialize(item.f2_ninf,in);
        dlib::deserialize(item.f3_ninf,in);
        dlib::deserialize(item.f1_qnan,in);
        dlib::deserialize(item.f2_qnan,in);
        dlib::deserialize(item.f3_qnan,in);
        dlib::deserialize(item.f1_snan,in);
        dlib::deserialize(item.f2_snan,in);
        dlib::deserialize(item.f3_snan,in);

        dlib::deserialize(item.s1,in);
        dlib::deserialize(item.s2,in);

        dlib::deserialize(item.array,in);

        dlib::deserialize(item.b_true,in);
        dlib::deserialize(item.b_false,in);
    }
    
    template<typename T>
    bool pointers_values_equal(const unique_ptr<T>& l, const unique_ptr<T>& r)
    {
        return l && r ? *l == *r : l == r;
    }

    template<typename T>
    bool pointers_values_equal(const shared_ptr<T>& l, const shared_ptr<T>& r)
    {
        return l && r ? *l == *r : l == r;
    }
    
    struct immutable_type
    {
        immutable_type() = default;
        immutable_type(const immutable_type& other)             = delete;
        immutable_type& operator=(const immutable_type& other)  = delete;
        immutable_type(immutable_type&& other)                  = delete;
        immutable_type& operator=(immutable_type&& other)       = delete;
        
        friend void serialize(const immutable_type&, std::ostream&) {}
        friend void deserialize(immutable_type&, std::istream&) {}
        bool operator==(const immutable_type&) const {return true;}
    };
    
    struct my_custom_type
    {
        int a;
        float b;
        std::vector<float> c;
        std::list<string> d;
        std::forward_list<string> e;
        std::pair<int,string> f;
        std::tuple<int,string,float> g;
        std::map<string,int> h;
        std::unordered_map<string, int> i;
        std::multimap<string, int> j;
        std::unordered_multimap<string, int> k;
        std::set<string> l;
        std::unordered_set<string> m;
        std::multiset<string> n;
        std::unordered_multiset<string> o;
        std::shared_ptr<string> ptr_shared1;
        std::shared_ptr<string> ptr_shared2;
        std::vector<std::complex<double>> p; 
#if __cplusplus >= 201703L
        std::variant<int,float,std::string> q;
        std::optional<std::vector<std::string>> r;
#endif

        bool operator==(const my_custom_type& rhs) const
        {         
#if __cplusplus >= 201703L
            const bool cpp17_ok = std::tie(q, r) == std::tie(rhs.q, rhs.r);
#else
            const bool cpp17_ok = true;
#endif
            return std::tie(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) == std::tie(rhs.a,rhs.b,rhs.c,rhs.d,rhs.e,rhs.f,rhs.g,rhs.h,rhs.i,rhs.j,rhs.k,rhs.l,rhs.m,rhs.n,rhs.o,rhs.p)
                    && cpp17_ok
                    && pointers_values_equal(ptr_shared1, rhs.ptr_shared1)
                    && pointers_values_equal(ptr_shared2, rhs.ptr_shared2);
        }

#if __cplusplus >= 201703L
        DLIB_DEFINE_DEFAULT_SERIALIZATION(my_custom_type, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ptr_shared1, ptr_shared2, q, r);
#else
        DLIB_DEFINE_DEFAULT_SERIALIZATION(my_custom_type, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ptr_shared1, ptr_shared2);
#endif
    };

    struct my_custom_type_array
    {
        std::vector<my_custom_type> v;

        bool operator==(const my_custom_type_array& rhs) const
        {
            return v == rhs.v;
        }

        DLIB_DEFINE_DEFAULT_SERIALIZATION(my_custom_type_array, v);
    };
    
// ----------------------------------------------------------------------------------------

    // This function returns the contents of the file 'stuff.bin' but using the old 
    // floating point serialization format.
    const std::string get_decoded_string()
    {
        dlib::base64::kernel_1a base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;


        // The base64 encoded data from the file 'stuff.bin' we want to decode and return.
        sout << "AVaifX9zEbXa9aocsrcRuvnNrR3WLuuU5eLWiy0UeXmnKXGLKZz8V44gzT4CM6wnCmAHFQug8G3C";
        sout << "4cuLdNgp2ApkeLcvwFNJRENE0ShrRaxEBFEA8nah7vm8B2VmgImNblCejuP5IcDt60EaCKlqiit8";
        sout << "+JGrzYxqBm3xFS4P+qlOROdbxc7pXBmUdh0rqNSEvn0FBPdoqY/5SpHgA2yAcH8XFrM1cdu0xS3P";
        sout << "8PBcmLMJ7bFdzplwhrjuxtm4NfEOi6Rl9sU44AXycYgJd0+uH+dyoI9X3co5b3YWJtjvdVeztNAr";
        sout << "BfSPfR6oAVNfiMBG7QA=";


        // Put the data into the istream sin
        sin.str(sout.str());
        sout.str("");

        // Decode the base64 text into its compressed binary form
        base64_coder.decode(sin,sout);
        sin.clear();
        sin.str(sout.str());
        sout.str("");

        // Decompress the data into its original form
        compressor.decompress(sin,sout);

        // Return the decoded and decompressed data
        return sout.str();
    }


    // This function returns the contents of the file 'stuff.bin' but using the new 
    // floating point serialization format.
    const std::string get_decoded_string2()
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        // The base64 encoded data from the file 'stuff.bin' we want to decode and return.
        sout << "AVaifX9zEbXa9aocsrcRuvnNqzZLptZ5mRd46xScCIfX6sq/46hG9JwIInElG50EtJKJY/+jAWit";
        sout << "TpDBWrxBz124JRLsBz62h0D3Tqgnd8zygRx7t33Ybw40o07MrhzNEHgYavUukaPje5by78JIWHgk";
        sout << "l7nb/TK+9ndVLrAThJ4v+GiPT3kh9H1tAAAAAQhbLa06pQjhrnjTXcRox1ZBEAV9/q1zAA==";

        // Put the data into the istream sin
        sin.str(sout.str());
        sout.str("");

        // Decode the base64 text into its compressed binary form
        base64_coder.decode(sin,sout);
        sin.clear();
        sin.str(sout.str());
        sout.str("");

        // Decompress the data into its original form
        compressor.decompress(sin,sout);

        // Return the decoded and decompressed data
        return sout.str();
    }

// ----------------------------------------------------------------------------------------

    // Declare the logger we will use in this test.  The name of the tester 
    // should start with "test."
    logger dlog("test.serialize");

    void serialize_test (
    )
    /*!
        ensures
            - runs tests on the serialization code for compliance with the specs
    !*/
    {        


        print_spinner();

        ostringstream sout;
        test_object obj;

        obj.set_state_1();
        obj.assert_in_state_1();
        serialize(obj, sout);
        obj.assert_in_state_1();

        obj.set_state_2();
        obj.assert_in_state_2();
        serialize(obj, sout);
        obj.assert_in_state_2();


        istringstream sin(sout.str());

        deserialize(obj,sin);
        obj.assert_in_state_1();
        deserialize(obj,sin);
        obj.assert_in_state_2();


        // now do the same thing as above but deserialize from some stored binary
        // data to make sure the serialized values are portable between different
        // machines

        sin.clear();
        sin.str(get_decoded_string());
        deserialize(obj,sin);
        obj.assert_in_state_1();
        deserialize(obj,sin);
        obj.assert_in_state_2();


        sin.clear();
        sin.str(get_decoded_string2());
        deserialize(obj,sin);
        obj.assert_in_state_1();
        deserialize(obj,sin);
        obj.assert_in_state_2();


        /*
        // This is the code that produced the encoded data stored in the get_decoded_string() function
            ofstream fout("stuff.bin",ios::binary);
            obj.set_state_1();
            obj.assert_in_state_1();
            serialize(obj, fout);
            obj.assert_in_state_1();

            obj.set_state_2();
            obj.assert_in_state_2();
            serialize(obj, fout);
            obj.assert_in_state_2();
            */


        test_object obj2;
        obj.set_state_1();
        obj2.set_state_2();
        dlib::serialize("serialization_test.dat") << obj << obj2;
        obj.assert_in_state_1();
        obj2.assert_in_state_2();
        obj.set_state_2();
        obj2.set_state_1();
        obj.assert_in_state_2();
        obj2.assert_in_state_1();
        dlib::deserialize("serialization_test.dat") >> obj >> obj2;
        obj.assert_in_state_1();
        obj2.assert_in_state_2();
    }


    template <typename T>
    void test_vector (
    )
    {
        std::vector<T> a, b;

        for (int i = -10; i < 30; ++i)
        {
            a.push_back(i);
        }

        ostringstream sout;
        dlib::serialize(a, sout);
        istringstream sin(sout.str());

        dlib::deserialize(b, sin);


        DLIB_TEST(a.size() == b.size());
        DLIB_TEST(a.size() == 40);
        for (unsigned long i = 0; i < a.size(); ++i)
        {
            DLIB_TEST(a[i] == b[i]);
        }

        std::vector<T> c;
        sout.str("");
        dlib::serialize(c, sout);
        sin.str(sout.str());
        dlib::deserialize(a, sin);
        DLIB_TEST(a.size() == 0);
        DLIB_TEST(c.size() == 0);
    }

    void test_std_array (
    )
    {
        std::array<int,5> a, b;

        a = {1, 2, 3, 4, 5};

        ostringstream sout;
        dlib::serialize(a, sout);
        istringstream sin(sout.str());

        dlib::deserialize(b, sin);


        DLIB_TEST(a.size() == b.size());
        DLIB_TEST(a.size() == 5);
        for (unsigned long i = 0; i < a.size(); ++i)
        {
            DLIB_TEST(a[i] == b[i]);
        }

        std::array<int,0> aa, bb;
        sout.str("");
        dlib::serialize(aa, sout);
        sin.str(sout.str());
        dlib::deserialize(bb, sin);
        DLIB_TEST(bb.size() == 0);
    }

    void test_vector_bool (
    )
    {
        std::vector<bool> a, b;

        a.push_back(true);
        a.push_back(true);
        a.push_back(false);
        a.push_back(true);
        a.push_back(false);
        a.push_back(true);

        ostringstream sout;
        dlib::serialize(a, sout);
        istringstream sin(sout.str());

        dlib::deserialize(b, sin);


        DLIB_TEST(a.size() == b.size());
        DLIB_TEST(a.size() == 6);
        for (unsigned long i = 0; i < a.size(); ++i)
        {
            DLIB_TEST(a[i] == b[i]);
        }
    }

// ----------------------------------------------------------------------------------------

    // This function returns the contents of the file 'matarray.dat'
    const std::string get_decoded_string_matarray_old()
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        // The base64 encoded data from the file 'matarray.dat' we want to decode and return.
        sout << "AW852sEbTIeV+m/wLUcKJKPW+6IclviUWZcFh1daDZ0blDjPNTgPx0Lv56sIEwlG4I6C5OJzJBkZ";
        sout << "PvczLjS7IEKh6eg7amNOyEexsQSgojL1oMe2gDEfkyInUGPJV90sNS0cvp/hIB134V8JCTYUP6vH";
        sout << "9qpegLSIIQG+/NjLWyK2472vC88BJfKgkL3CPLMjQwB3tB928FNLbESDLIvpnb6q9ve68iuoyZZt";
        sout << "z3TTJxHW3MIdgzuhNomvPxfo/Q+7lC/Orj0FewUX90al6DckwzOtLVRidh/ZKpsQsxzJYQGkjdX5";
        sout << "mDzzXKqQb3Y3DnzEmwtRD9CUON3iRv1r26gHWLYorrYA";


        // Put the data into the istream sin
        sin.str(sout.str());
        sout.str("");

        // Decode the base64 text into its compressed binary form
        base64_coder.decode(sin,sout);
        sin.clear();
        sin.str(sout.str());
        sout.str("");

        // Decompress the data into its original form
        compressor.decompress(sin,sout);

        // Return the decoded and decompressed data
        return sout.str();
    }

    // This function returns the contents of the file 'matarray.dat'
    const std::string get_decoded_string_matarray()
    {
        dlib::base64 base64_coder;
        dlib::compress_stream::kernel_1ea compressor;
        std::ostringstream sout;
        std::istringstream sin;

        // The base64 encoded data from the file 'matarray.dat' we want to decode and return.
        sout << "gO6XH2WGbm8Xaw3a5FJbh3V823W6P2Qk/vHaAAAAARccIppHWdmViaKby7JA5PQvXjYMWUYvXRHv";
        sout << "xPdURZl1un3CT/rjT11Yry0y3+1W7GBmfBJ0gVFKGdiGuqoNAMtmzL/ll3YfEQ7ED7aB33aDTktw";
        sout << "AWVkHT+gqTbKwjP+8YvB3s3ziK640ITOAWazAghKDVl7AHGn+fjq29paBZMczuJofl8FinZUhwa9";
        sout << "Ol5gdAEQa6VZDmJUeo2soTJcEDpkW9LkRmXvjQkyEHfEHQNFDfQq4p2U+dHz4lOKlcj3VzQIeG/s";
        sout << "oxa9KhJND4aQ5xeNUUHUzFBU3XhQHlyDIn/RNdX/ZwA=";


        // Put the data into the istream sin
        sin.str(sout.str());
        sout.str("");

        // Decode the base64 text into its compressed binary form
        base64_coder.decode(sin,sout);
        sin.clear();
        sin.str(sout.str());
        sout.str("");

        // Decompress the data into its original form
        compressor.decompress(sin,sout);

        // Return the decoded and decompressed data
        return sout.str();
    }

    void setup_mats_and_arrays (
        array2d<int>& a,
        matrix<int>& m,
        array2d<unsigned char>&  img1,
        array2d<rgb_pixel>&      img2,
        array2d<bgr_pixel>&      img3,
        array2d<rgb_alpha_pixel>& img4,
        array2d<hsi_pixel>&      img5
    )
    {
        a.set_size(3,5);
        int cnt = 0;
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                a[r][c] = cnt++;
            }
        }
        m = mat(a);

        img1.set_size(3,5);
        img2.set_size(3,5);
        img3.set_size(3,5);
        img4.set_size(3,5);
        img5.set_size(3,5);

        assign_all_pixels(img1, 0);
        assign_all_pixels(img2, 0);
        assign_all_pixels(img3, 0);
        assign_all_pixels(img4, 0);
        assign_all_pixels(img5, 0);

        unsigned char pcnt = 0;
        for (long r = 0; r < img1.nr(); ++r)
        {
            for (long c = 0; c < img1.nc(); ++c)
            {
                rgb_alpha_pixel temp;
                temp.red = pcnt++;
                temp.green = pcnt++;
                temp.blue = pcnt++;
                temp.alpha = 150+pcnt++;
                assign_pixel(img1[r][c], temp);
                assign_pixel(img2[r][c], temp);
                assign_pixel(img3[r][c], temp);
                assign_pixel(img4[r][c], temp);
            }
        }

        for (long r = 0; r < img5.nr(); ++r)
        {
            for (long c = 0; c < img5.nc(); ++c)
            {
                img5[r][c].h = pcnt++;
                img5[r][c].s = pcnt++;
                img5[r][c].i = pcnt++;
            }
        }
    }


    void test_deserialize(
        std::istream& fin
    )
    {
        array2d<int> a;
        matrix<int> m;
        array2d<unsigned char>  img1;
        array2d<rgb_pixel>      img2;
        array2d<bgr_pixel>      img3;
        array2d<rgb_alpha_pixel> img4;
        array2d<hsi_pixel>      img5;
        setup_mats_and_arrays(a,m,img1,img2,img3,img4,img5);


        array2d<unsigned char>  img1_;
        array2d<rgb_pixel>      img2_;
        array2d<bgr_pixel>      img3_;
        array2d<rgb_alpha_pixel> img4_;
        array2d<hsi_pixel>      img5_;

        matrix<int> m_;
        array2d<int> a_;

        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));
        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));

        deserialize(img1_, fin); DLIB_TEST(mat(img1_) == mat(img1));
        deserialize(img2_, fin); DLIB_TEST(mat(img2_) == mat(img2));
        deserialize(img3_, fin); DLIB_TEST(mat(img3_) == mat(img3));
        deserialize(img4_, fin); DLIB_TEST(mat(img4_) == mat(img4));
        deserialize(img5_, fin); DLIB_TEST(mat(img5_) == mat(img5));
    }

    void test_deserialize_all_array2d(
        std::istream& fin
    )
    {
        array2d<int> a;
        matrix<int> m;
        array2d<unsigned char>  img1;
        array2d<rgb_pixel>      img2;
        array2d<bgr_pixel>      img3;
        array2d<rgb_alpha_pixel> img4;
        array2d<hsi_pixel>      img5;
        setup_mats_and_arrays(a,m,img1,img2,img3,img4,img5);


        array2d<unsigned char>  img1_;
        array2d<rgb_pixel>      img2_;
        array2d<bgr_pixel>      img3_;
        array2d<rgb_alpha_pixel> img4_;
        array2d<hsi_pixel>      img5_;

        array2d<int> m_;
        array2d<int> a_;

        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));
        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));

        deserialize(img1_, fin); DLIB_TEST(mat(img1_) == mat(img1));
        deserialize(img2_, fin); DLIB_TEST(mat(img2_) == mat(img2));
        deserialize(img3_, fin); DLIB_TEST(mat(img3_) == mat(img3));
        deserialize(img4_, fin); DLIB_TEST(mat(img4_) == mat(img4));
        deserialize(img5_, fin); DLIB_TEST(mat(img5_) == mat(img5));
    }

    void test_deserialize_all_matrix(
        std::istream& fin
    )
    {
        array2d<int> a;
        matrix<int> m;
        array2d<unsigned char>  img1;
        array2d<rgb_pixel>      img2;
        array2d<bgr_pixel>      img3;
        array2d<rgb_alpha_pixel> img4;
        array2d<hsi_pixel>      img5;
        setup_mats_and_arrays(a,m,img1,img2,img3,img4,img5);


        matrix<unsigned char>  img1_;
        matrix<rgb_pixel>      img2_;
        matrix<bgr_pixel>      img3_;
        matrix<rgb_alpha_pixel> img4_;
        matrix<hsi_pixel>      img5_;

        matrix<int> m_;
        matrix<int> a_;

        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));
        deserialize(a_, fin); DLIB_TEST(mat(a_) == mat(a));
        deserialize(m_, fin); DLIB_TEST(mat(m_) == mat(m));

        deserialize(img1_, fin); DLIB_TEST(mat(img1_) == mat(img1));
        deserialize(img2_, fin); DLIB_TEST(mat(img2_) == mat(img2));
        deserialize(img3_, fin); DLIB_TEST(mat(img3_) == mat(img3));
        deserialize(img4_, fin); DLIB_TEST(mat(img4_) == mat(img4));
        deserialize(img5_, fin); DLIB_TEST(mat(img5_) == mat(img5));
    }

    void test_array2d_and_matrix_serialization()
    {
        ostringstream sout;
        array2d<int> a;
        matrix<int> m;
        array2d<unsigned char>  img1;
        array2d<rgb_pixel>      img2;
        array2d<bgr_pixel>      img3;
        array2d<rgb_alpha_pixel> img4;
        array2d<hsi_pixel>      img5;
        setup_mats_and_arrays(a,m,img1,img2,img3,img4,img5);

        serialize(a, sout);
        serialize(m, sout);
        serialize(a, sout);
        serialize(m, sout);

        serialize(img1, sout);
        serialize(img2, sout);
        serialize(img3, sout);
        serialize(img4, sout);
        serialize(img5, sout);

    // --------------------

        {
            istringstream sin(sout.str());
            test_deserialize(sin);
        }
        {
            istringstream sin(sout.str());
            test_deserialize_all_array2d(sin);
        }
        {
            istringstream sin(sout.str());
            test_deserialize_all_matrix(sin);
        }


        {
            istringstream sin(get_decoded_string_matarray());
            test_deserialize(sin);
        }
        {
            istringstream sin(get_decoded_string_matarray());
            test_deserialize_all_array2d(sin);
        }
        {
            istringstream sin(get_decoded_string_matarray());
            test_deserialize_all_matrix(sin);
        }


        {
            // Make sure we can still deserialize the serialization 
            // format for array2d and matrix objects used by older versions 
            // of dlib.
            istringstream sin(get_decoded_string_matarray_old());
            test_deserialize(sin);
        }
    }

// ----------------------------------------------------------------------------------------

    void test_strings()
    {
        string str1 = "stuff";
        char buf[6];
        buf[0] = 0;
        buf[1] = 1;
        buf[2] = 2;
        buf[3] = 0;
        buf[4] = 3;
        buf[5] = 3;

        dlib::serialize("ser_test_string.dat") << str1 << buf << "morestuff" << "";

        string str2, str3, str4;
        char buf2[6];
        memset(buf2,0,sizeof(buf2));
        dlib::deserialize("ser_test_string.dat") >> str2 >> buf2 >> str3 >> str4;
        DLIB_TEST(str2 == "stuff");
        DLIB_TEST(str3 == "morestuff");
        DLIB_TEST(str4 == "");
        DLIB_TEST(buf2[0] == 0);
        DLIB_TEST(buf2[1] == 1);
        DLIB_TEST(buf2[2] == 2);
        DLIB_TEST(buf2[3] == 0);
        DLIB_TEST(buf2[4] == 3);
        DLIB_TEST(buf2[5] == 3);


        ofstream fout("ser_test_string.dat", ios::binary);
        dlib::serialize(str1, fout);
        dlib::serialize(buf, fout);
        dlib::serialize("morestuff", fout);
        fout.close();
        ifstream fin("ser_test_string.dat", ios::binary);
        memset(buf2,0,sizeof(buf2));
        str2.clear();
        str3.clear();
        dlib::deserialize(str2, fin);
        dlib::deserialize(buf2, fin);
        dlib::deserialize(str3, fin);

        DLIB_TEST(str2 == "stuff");
        DLIB_TEST(str3 == "morestuff");
        DLIB_TEST(buf2[0] == 0);
        DLIB_TEST(buf2[1] == 1);
        DLIB_TEST(buf2[2] == 2);
        DLIB_TEST(buf2[3] == 0);
        DLIB_TEST(buf2[4] == 3);
        DLIB_TEST(buf2[5] == 3);



        // make sure ramdump() overloads compile and work.
        {
            matrix<double,2,2> a = {1,2,3,4};
            const matrix<double,2,2> b = {3,2,3,4};
            dlib::serialize("ramdump_mat.dat") << ramdump(a) << ramdump(b);
            matrix<double,2,2> A, B;
            dlib::deserialize("ramdump_mat.dat") >> ramdump(A) >> ramdump(B);

            DLIB_TEST(A == a);
            DLIB_TEST(B == b);
            A = 0;
            B = 0;
            DLIB_TEST(A != a);
            DLIB_TEST(B != b);

            ostringstream sout;
            dlib::serialize(ramdump(a), sout);
            dlib::serialize(ramdump(b), sout);
            istringstream sin(sout.str());
            dlib::deserialize(ramdump(A), sin);
            dlib::deserialize(ramdump(B), sin);

            DLIB_TEST(A == a);
            DLIB_TEST(B == b);
        }
    }
    
    void test_macros_and_serializers()
    {
        std::unique_ptr<string> uptr1, uptr2, uptr3, uptr4;
        uptr1.reset(new string("hello from uptr1"));
    
        my_custom_type t1, t2, t3, t4;
        t1.a = 1;
        t1.b = 2.5;
        t1.c = {1.f, 2.f, 3.f, 4.f, 5.f};
        t1.d.push_back("hello from back of list");
        t1.d.push_back("world from back of list");
        t1.d.push_front("world from front of list");
        t1.d.push_front("hello from front of list");
        t1.e.push_front("world from forward_list");
        t1.e.push_front("hello from forward_list");
        t1.f = make_pair(2, "hello from pair");
        std::get<0>(t1.g) = 2;
        std::get<1>(t1.g) = "hello from tuple";
        std::get<2>(t1.g) = 1.4142;
        t1.h["key"] = 15;
        t1.i["key"] = 16;
        t1.i.insert({"inserted key", 17});
        t1.j.insert({"key", 21});
        t1.j.insert({"key", 22});
        t1.j.insert({"inserted key", 23});
        t1.j.insert({"inserted key", 24});
        t1.j.insert({"key", 25});
        t1.j.insert({"key", 26});
        t1.k.insert({"inserted key", 27});
        t1.k.insert({"inserted key", 28});
        t1.l.insert("hello from set");
        t1.m.insert("hello from unordered_set");
        t1.n.insert("hello from multiset");
        t1.n.insert("hello from multiset");
        t1.o.insert("hello from unordered_multiset");
        t1.o.insert("hello from unordered_multiset");
        t1.ptr_shared1 = make_shared<string>("hello from shared_ptr");
        dlib::rand rng(std::time(NULL));
        for (int i = 0 ; i < 1024 ; i++)
            t1.p.push_back(rng.get_random_gaussian());
#if __cplusplus >= 201703L
        t1.q = "hello there from std::variant, welcome!";
        t1.r = {"hello from optional vector of string"};
#endif
        t2.a = 2;
        t2.b = 4.0;
        t2.c.resize(10);

        my_custom_type_array v1, v2;
        v1.v.push_back(t1);
        v1.v.push_back(t2);

#if __cplusplus >= 201703L
        std::variant<int,float,std::string,immutable_type> i1, i2;
        i1 = std::string("hello from variant that can hold an immutable type");
#else
        // make it so that we don't need to add #if guards in every block below
        std::string i1, i2;
        i1 = "std::variant not supported";
#endif
        {
            dlib::serialize("serialization_test_macros.dat") << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            dlib::deserialize("serialization_test_macros.dat") >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::stringstream ss;
            dlib::serialize(ss) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            dlib::deserialize(ss) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::ostringstream sout;
            dlib::serialize(sout) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            std::istringstream sin(sout.str());
            dlib::deserialize(sin) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<char> buf;
            dlib::serialize(buf) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            dlib::deserialize(buf) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<int8_t> buf;
            dlib::serialize(buf) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            dlib::deserialize(buf) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<uint8_t> buf;
            dlib::serialize(buf) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            dlib::deserialize(buf) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<char> buf1;
            dlib::serialize(buf1) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            std::vector<int8_t> buf2(buf1.begin(), buf1.end());
            dlib::deserialize(buf2) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<char> buf1;
            dlib::serialize(buf1) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            std::vector<uint8_t> buf2(buf1.begin(), buf1.end());
            dlib::deserialize(buf2) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
        
        {
            std::vector<int8_t> buf1;
            dlib::serialize(buf1) << t1 << t2 << v1 << uptr1 << uptr2 << i1;
            std::vector<uint8_t> buf2(buf1.begin(), buf1.end());
            dlib::deserialize(buf2) >> t3 >> t4 >> v2 >> uptr3 >> uptr4 >> i2;

            DLIB_TEST(t1 == t3);
            DLIB_TEST(t2 == t4);
            DLIB_TEST(v1 == v2);
            DLIB_TEST(pointers_values_equal(uptr1, uptr3));
            DLIB_TEST(pointers_values_equal(uptr2, uptr4));
            DLIB_TEST(i1 == i2);
        }
    }

// ----------------------------------------------------------------------------------------

    class serialize_tester : public tester
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a test for the serialization .  When it is constructed
                it adds itself into the testing framework.  The command line switch is
                specified as test_serialize by passing that string to the tester constructor.
        !*/
    public:
        serialize_tester (
        ) :
            tester ("test_serialize",
                    "Runs tests on the serialization code.")
        {}

        void perform_test (
        )
        {
            serialize_test();
            test_vector<char>();
            test_vector<unsigned char>();
            test_vector<int>();
            test_vector_bool();
            test_array2d_and_matrix_serialization();
            test_strings();
            test_std_array();
            test_macros_and_serializers();
        }
    } a;


}


