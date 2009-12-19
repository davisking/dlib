// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
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
    dlib::logger dlog("test.empirical_kernel_map");


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
                "test_empirical_kernel_map",       // the command line argument name for this test
                "Run tests on the empirical_kernel_map object.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
            thetime = time(0);
        }

        time_t thetime;
        dlib::rand::float_1a rnd;

        template <typename kernel_type>
        void test_with_kernel(const kernel_type& kern)
        {
            typedef typename kernel_type::sample_type sample_type;

            empirical_kernel_map<kernel_type> ekm, ekm2, ekm3;

            for (int j = 0; j < 10; ++j)
            {
                sample_type samp;
                std::vector<sample_type> samples;
                std::vector<sample_type> proj_samples;
                print_spinner();
                const int num = rnd.get_random_8bit_number()%200 + 1;
                // make some random samples
                for (int i = 0; i < num; ++i)
                {
                    samples.push_back(randm(4,1,rnd));
                }
                // add on a little bit to make sure there is at least one non-zero sample.  If all the 
                // samples are zero then empirical_kernel_map_error will be thrown and we don't want that.
                samples.front()(0) += 0.001;

                ekm2.load(kern, samples);
                // test serialization
                ostringstream sout;
                serialize(ekm2, sout);
                ekm2.clear();
                istringstream sin(sout.str());
                deserialize(ekm3, sin);
                // also test swap
                ekm3.swap(ekm);
                DLIB_TEST(ekm.get_kernel() == kern);
                DLIB_TEST(ekm.out_vector_size() != 0);
                DLIB_TEST(ekm2.out_vector_size() == 0);
                DLIB_TEST(ekm3.out_vector_size() == 0);



                // project all the samples into kernel space
                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    proj_samples.push_back(ekm.project(samples[i]));
                }

                DLIB_TEST(max(abs(kernel_matrix(kern, samples) - kernel_matrix(linear_kernel<sample_type>(), proj_samples))) < 1e-12);
                DLIB_TEST(ekm.out_vector_size() == proj_samples[0].size());

                for (int i = 0; i < 30; ++i)
                {
                    const unsigned long idx1 = rnd.get_random_32bit_number()%samples.size();
                    const unsigned long idx2 = rnd.get_random_32bit_number()%samples.size();
                    decision_function<kernel_type> dec_funct = ekm.convert_to_decision_function(proj_samples[idx1]);
                    distance_function<kernel_type> dist_funct = ekm.convert_to_distance_function(proj_samples[idx1]);

                    // make sure the distances match 
                    const double dist_error = abs(length(proj_samples[idx1] - proj_samples[idx2]) - dist_funct(samples[idx2]));
                    DLIB_TEST_MSG( dist_error < 1e-7, dist_error);
                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],proj_samples[idx2]) - dec_funct(samples[idx2])) < 1e-10);

                    // also try the dec_funct with samples that weren't in the original set
                    samp = 100*randm(4,1,rnd);
                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],ekm.project(samp)) - dec_funct(samp)) < 1e-10);
                    samp = randm(4,1,rnd);
                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],ekm.project(samp)) - dec_funct(samp)) < 1e-10);
                }



                proj_samples.clear();


                // now do the projection but use the projection_function returned by get_projection_function()
                projection_function<kernel_type> proj2 = ekm.get_projection_function();
                projection_function<kernel_type> proj;
                sout.clear();
                sout.str("");
                sin.clear();
                sin.str("");
                // test serialization
                serialize(proj2, sout);
                sin.str(sout.str());
                deserialize(proj, sin);

                for (unsigned long i = 0; i < samples.size(); ++i)
                {
                    proj_samples.push_back(proj(samples[i]));
                }

                DLIB_TEST(max(abs(kernel_matrix(kern, samples) - kernel_matrix(linear_kernel<sample_type>(), proj_samples))) < 1e-12);
                DLIB_TEST(ekm.out_vector_size() == proj_samples[0].size());
                DLIB_TEST(proj.out_vector_size() == proj_samples[0].size());

                ekm.clear();
                DLIB_TEST(ekm.out_vector_size() == 0);
                DLIB_TEST(ekm2.out_vector_size() == 0);
                DLIB_TEST(ekm3.out_vector_size() == 0);


                for (int i = 0; i < 30; ++i)
                {
                    const unsigned long idx1 = rnd.get_random_32bit_number()%samples.size();
                    const unsigned long idx2 = rnd.get_random_32bit_number()%samples.size();
                    decision_function<kernel_type> dec_funct = convert_to_decision_function(proj,proj_samples[idx1]);

                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],proj_samples[idx2]) - dec_funct(samples[idx2])) < 1e-10);

                    // also try the dec_funct with samples that weren't in the original set
                    samp = 100*randm(4,1,rnd);
                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],proj(samp)) - dec_funct(samp)) < 1e-10);
                    samp = randm(4,1,rnd);
                    // make sure the dot products match 
                    DLIB_TEST(abs(dot(proj_samples[idx1],proj(samp)) - dec_funct(samp)) < 1e-10);
                }


            }
        }

        void perform_test (
        )
        {
            ++thetime;
            typedef matrix<double,0,1> sample_type;
            dlog << LINFO << "time seed: " << thetime;
            rnd.set_seed(cast_to_string(thetime));

            print_spinner();
            test_with_kernel(linear_kernel<sample_type>());
            print_spinner();
            test_with_kernel(radial_basis_kernel<sample_type>(0.2));
            print_spinner();
        }
    };

    // Create an instance of this object.  Doing this causes this test
    // to be automatically inserted into the testing framework whenever this cpp file
    // is linked into the project.  Note that since we are inside an unnamed-namespace 
    // we won't get any linker errors about the symbol a being defined multple times. 
    empirical_kernel_map_tester a;

}


