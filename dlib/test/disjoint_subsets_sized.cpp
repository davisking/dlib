// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/disjoint_subsets.h>

#include "tester.h"

namespace
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.disjoint_subsets_sized");

    void test_disjoint_subsets_sized()
    {
        print_spinner();
        disjoint_subsets_sized s;

        DLIB_TEST(s.size() == 0);
        DLIB_TEST(s.get_number_of_sets() == 0);

        s.set_size(5);
        DLIB_TEST(s.size() == 5);
        DLIB_TEST(s.get_number_of_sets() == 5);

        DLIB_TEST(s.find_set(0) == 0);
        DLIB_TEST(s.find_set(1) == 1);
        DLIB_TEST(s.find_set(2) == 2);
        DLIB_TEST(s.find_set(3) == 3);
        DLIB_TEST(s.find_set(4) == 4);

        DLIB_TEST(s.get_size_of_set(0) == 1);
        DLIB_TEST(s.get_size_of_set(1) == 1);
        DLIB_TEST(s.get_size_of_set(2) == 1);
        DLIB_TEST(s.get_size_of_set(3) == 1);
        DLIB_TEST(s.get_size_of_set(4) == 1);

        unsigned long id = s.merge_sets(1,3);
        DLIB_TEST(s.get_number_of_sets() == 4);
        DLIB_TEST(s.find_set(0) == 0);
        DLIB_TEST(s.find_set(1) == id);
        DLIB_TEST(s.find_set(2) == 2);
        DLIB_TEST(s.find_set(3) == id);
        DLIB_TEST(s.find_set(4) == 4);
        DLIB_TEST(s.get_size_of_set(0) == 1);
        DLIB_TEST(s.get_size_of_set(s.find_set(1)) == 2);
        DLIB_TEST(s.get_size_of_set(2) == 1);
        DLIB_TEST(s.get_size_of_set(s.find_set(3)) == 2);
        DLIB_TEST(s.get_size_of_set(4) == 1);

        id = s.merge_sets(s.find_set(1),4);
        DLIB_TEST(s.get_number_of_sets() == 3);
        DLIB_TEST(s.find_set(0) == 0);
        DLIB_TEST(s.find_set(1) == id);
        DLIB_TEST(s.find_set(2) == 2);
        DLIB_TEST(s.find_set(3) == id);
        DLIB_TEST(s.find_set(4) == id);
        DLIB_TEST(s.get_size_of_set(0) == 1);
        DLIB_TEST(s.get_size_of_set(s.find_set(1)) == 3);
        DLIB_TEST(s.get_size_of_set(2) == 1);
        DLIB_TEST(s.get_size_of_set(s.find_set(3)) == 3);
        DLIB_TEST(s.get_size_of_set(s.find_set(4)) == 3);

        unsigned long id2 = s.merge_sets(0,2);
        DLIB_TEST(s.get_number_of_sets() == 2);
        DLIB_TEST(s.find_set(0) == id2);
        DLIB_TEST(s.find_set(1) == id);
        DLIB_TEST(s.find_set(2) == id2);
        DLIB_TEST(s.find_set(3) == id);
        DLIB_TEST(s.find_set(4) == id);
        DLIB_TEST(s.get_size_of_set(s.find_set(0)) == 2);
        DLIB_TEST(s.get_size_of_set(s.find_set(1)) == 3);
        DLIB_TEST(s.get_size_of_set(s.find_set(2)) == 2);
        DLIB_TEST(s.get_size_of_set(s.find_set(3)) == 3);
        DLIB_TEST(s.get_size_of_set(s.find_set(4)) == 3);

        id = s.merge_sets(s.find_set(1),s.find_set(0));
        DLIB_TEST(s.get_number_of_sets() == 1);
        DLIB_TEST(s.find_set(0) == id);
        DLIB_TEST(s.find_set(1) == id);
        DLIB_TEST(s.find_set(2) == id);
        DLIB_TEST(s.find_set(3) == id);
        DLIB_TEST(s.find_set(4) == id);
        DLIB_TEST(s.get_size_of_set(s.find_set(0)) == 5);
        DLIB_TEST(s.get_size_of_set(s.find_set(1)) == 5);
        DLIB_TEST(s.get_size_of_set(s.find_set(2)) == 5);
        DLIB_TEST(s.get_size_of_set(s.find_set(3)) == 5);
        DLIB_TEST(s.get_size_of_set(s.find_set(4)) == 5);

        DLIB_TEST(s.size() == 5);
        s.set_size(1);
        DLIB_TEST(s.size() == 1);
        DLIB_TEST(s.get_number_of_sets() == 1);
        DLIB_TEST(s.find_set(0) == 0);
        DLIB_TEST(s.get_size_of_set(0) == 1);
        s.set_size(2);
        DLIB_TEST(s.size() == 2);
        DLIB_TEST(s.get_number_of_sets() == 2);
        DLIB_TEST(s.find_set(0) == 0);
        DLIB_TEST(s.find_set(1) == 1);
        DLIB_TEST(s.get_size_of_set(0) == 1);
        DLIB_TEST(s.get_size_of_set(1) == 1);
        id = s.merge_sets(0,1);
        DLIB_TEST(s.size() == 2);
        DLIB_TEST(s.get_number_of_sets() == 1);
        DLIB_TEST(id == s.find_set(0));
        DLIB_TEST(id == s.find_set(1));
        DLIB_TEST(s.get_size_of_set(s.find_set(0)) == 2);
        DLIB_TEST(s.get_size_of_set(s.find_set(1)) == 2);
        DLIB_TEST(s.size() == 2);
        s.clear();
        DLIB_TEST(s.size() == 0);
        DLIB_TEST(s.get_number_of_sets() == 0);

    }


    class tester_disjoint_subsets_sized : public tester
    {
    public:
        tester_disjoint_subsets_sized (
        ) :
            tester ("test_disjoint_subsets_sized",
                    "Runs tests on the disjoint_subsets_sized component.")
        {}

        void perform_test (
        )
        {
            test_disjoint_subsets_sized();
        }
    } a;


}
