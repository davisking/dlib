// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/svm.h>
#include <dlib/rand.h>
#include <dlib/string.h>
#include <vector>
#include <sstream>
#include <ctime>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.ekm_and_lisf");


    class empirical_kernel_map_tester : public tester
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a unit test.  When it is constructed
                it adds itself into the testing framework.
        !*/
    public:
        empirical_kernel_map_tester (
        ) :
            tester (
                "test_ekm_and_lisf",       // the command line argument name for this test
                "Run tests on the empirical_kernel_map and linearly_independent_subset_finder objects.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
            thetime = time(0);
        }

        time_t thetime;
        dlib::rand::float_1a rnd;



        void perform_test (
        )
        {
            ++thetime;
            typedef matrix<double,0,1> sample_type;
            //dlog << LINFO << "time seed: " << thetime;
            //rnd.set_seed(cast_to_string(thetime));


            typedef radial_basis_kernel<sample_type> kernel_type;


            for (int n = 1; n < 10; ++n)
            {
                print_spinner();
                dlog << LINFO << "matrix size " << n;

                std::vector<sample_type> samples;
                // make some samples
                for (int i = 0; i < n; ++i)
                {
                    samples.push_back(randm(4,1,rnd));
                    // double up the samples just to mess with the lisf
                    if (n > 5)
                        samples.push_back(samples.back());
                }

                dlog << LINFO << "samples.size(): "<< samples.size();

                const kernel_type kern(1);

                linearly_independent_subset_finder<kernel_type> lisf(kern, 100, 1e-4);
                unsigned long count = 0;
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    if (lisf.add(samples[i]))
                    {
                        DLIB_TEST(equal(lisf[lisf.dictionary_size()-1], samples[i]));
                        ++count;
                    }
                }
                DLIB_TEST(count == lisf.dictionary_size());

                DLIB_TEST(lisf.dictionary_size() == (unsigned int)n);


                dlog << LINFO << "lisf.dictionary_size(): "<< lisf.dictionary_size();

                // make sure the kernel matrices coming out of the lisf are correct
                DLIB_TEST(dlib::equal(lisf.get_kernel_matrix(), kernel_matrix(kern, lisf.get_dictionary()), 1e-8));
                DLIB_TEST(dlib::equal(lisf.get_inv_kernel_marix(), inv(kernel_matrix(kern, lisf.get_dictionary())), 1e-8));

                empirical_kernel_map<kernel_type> ekm;
                ekm.load(lisf);
                DLIB_TEST(ekm.basis_size() == lisf.dictionary_size());

                std::vector<sample_type> proj_samples;
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    double err;
                    proj_samples.push_back(ekm.project(samples[i], err));
                    DLIB_TEST(err <= 1e-4);
                }

                // make sure the EKM did the projection correctly
                DLIB_TEST(dlib::equal(kernel_matrix(kern, samples), kernel_matrix(linear_kernel<sample_type>(), proj_samples), 1e-5));
            }



        }
    };

    // Create an instance of this object.  Doing this causes this test
    // to be automatically inserted into the testing framework whenever this cpp file
    // is linked into the project.  Note that since we are inside an unnamed-namespace 
    // we won't get any linker errors about the symbol a being defined multiple times. 
    empirical_kernel_map_tester a;

}


