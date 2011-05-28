// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/hash.h>
#include <dlib/matrix.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.hash");


    class test_hash : public tester
    {
    public:
        test_hash (
        ) :
            tester ("test_hash",
                    "Runs tests on the hash routines.")
        {}

        void perform_test (
        )
        {
            std::string str1 = "some random string";
            std::wstring str2 = L"another String!";
            matrix<unsigned char> mat(2,2);

            mat = 1,2,3,4;

            std::vector<unsigned char> v(4);
            v[0] = 'c';
            v[1] = 'a';
            v[2] = 't';
            v[3] = '!';

            std::map<unsigned char, unsigned char> m;
            m['c'] = 'C';
            m['a'] = 'A';
            m['t'] = 'T';

            dlog << LINFO << "hash(str1): "<< hash(str1);
            dlog << LINFO << "hash(str2): "<< hash(str2);
            dlog << LINFO << "hash(v): "<< hash(v);
            dlog << LINFO << "hash(m): "<< hash(m);
            dlog << LINFO << "hash(mat): "<< hash(mat);

            DLIB_TEST(hash(str1) == 1073638390);
            DLIB_TEST(hash(str2) == 2413364589);
            DLIB_TEST(hash(v) == 4054789286);
            DLIB_TEST(hash(m) == 2865512303);
            DLIB_TEST(hash(mat) == 1043635621);
            DLIB_TEST(murmur_hash3(&str1[0], str1.size(), 0) == 1073638390);

            dlog << LINFO << "hash(str1,1): "<< hash(str1,1);
            dlog << LINFO << "hash(str2,2): "<< hash(str2,2);
            dlog << LINFO << "hash(v,3): "<< hash(v,3);
            dlog << LINFO << "hash(m,4): "<< hash(m,4);
            dlog << LINFO << "hash(mat,5): "<< hash(mat,5);

            DLIB_TEST(hash(str1,1) == 2977753747);
            DLIB_TEST(hash(str2,2) == 3656927287);
            DLIB_TEST(hash(v,3) == 2127112268);
            DLIB_TEST(hash(m,4) == 4200495810);
            DLIB_TEST(hash(mat,5) == 2380427865);
            DLIB_TEST(murmur_hash3(&str1[0], str1.size(), 1) == 2977753747);

        }
    } a;



}



