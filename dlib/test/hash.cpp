// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/hash.h>
#include <dlib/matrix.h>
#include <dlib/byte_orderer.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.hash");


    template <typename T>
    void to_little (
        std::vector<T>& item
    )
    {
        byte_orderer bo;
        for (unsigned long i = 0; i < item.size(); ++i)
            bo.host_to_little(item[i]);
    }


    template <typename T>
    void to_little (
        matrix<T>& item
    )
    {
        byte_orderer bo;
        for (long r = 0; r < item.nr(); ++r)
        {
            for (long c = 0; c < item.nc(); ++c)
            {
                bo.host_to_little(item(r,c));
            }
        }
    }

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
            print_spinner();
            std::string str1 = "some random string";
            matrix<unsigned char> mat(2,2);

            mat = 1,2,3,4;

            matrix<uint64> mat2(2,3);

            mat2 = 1,2,3,4,5,6;

            to_little(mat2);

            std::vector<unsigned char> v(4);
            v[0] = 'c';
            v[1] = 'a';
            v[2] = 't';
            v[3] = '!';

            std::vector<uint16> v2(4);
            v[0] = 'c';
            v[1] = 'a';
            v[2] = 't';
            v[3] = '!';
            to_little(v2);

            std::map<unsigned char, unsigned char> m;
            m['c'] = 'C';
            m['a'] = 'A';
            m['t'] = 'T';

            dlog << LINFO << "hash(str1): "<< hash(str1);
            dlog << LINFO << "hash(v): "<< hash(v);
            dlog << LINFO << "hash(v2): "<< hash(v2);
            dlog << LINFO << "hash(m): "<< hash(m);
            dlog << LINFO << "hash(mat): "<< hash(mat);
            dlog << LINFO << "hash(mat2): "<< hash(mat2);

            DLIB_TEST(hash(str1) == 1073638390);
            DLIB_TEST(hash(v) == 4054789286);
            DLIB_TEST(hash(v2) == 1669671676);
            DLIB_TEST(hash(m) == 2865512303);
            DLIB_TEST(hash(mat) == 1043635621);
            DLIB_TEST(hash(mat2) == 982899794);
            DLIB_TEST(murmur_hash3(&str1[0], str1.size(), 0) == 1073638390);

            dlog << LINFO << "hash(str1,1): "<< hash(str1,1);
            dlog << LINFO << "hash(v,3): "<< hash(v,3);
            dlog << LINFO << "hash(v2,3): "<< hash(v2,3);
            dlog << LINFO << "hash(m,4): "<< hash(m,4);
            dlog << LINFO << "hash(mat,5): "<< hash(mat,5);
            dlog << LINFO << "hash(mat2,6): "<< hash(mat2,6);

            DLIB_TEST(hash(str1,1) == 2977753747);
            DLIB_TEST(hash(v,3) == 2127112268);
            DLIB_TEST(hash(v2,3) == 2999850111);
            DLIB_TEST(hash(m,4) == 4200495810);
            DLIB_TEST(hash(mat,5) == 2380427865);
            DLIB_TEST(hash(mat2,6) == 3098179348 );
            DLIB_TEST(murmur_hash3(&str1[0], str1.size(), 1) == 2977753747);

        }
    } a;



}



