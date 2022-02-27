// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <dlib/rand.h>
#include <dlib/compress_stream.h>
#include <dlib/hash.h>
#include <dlib/statistics.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.rand");

    void check_bpp (
        const std::string str
    )
    {
        istringstream rdata;
        ostringstream sout;
        rdata.str(str);
        double compressed_size;
        compress_stream::kernel_1a cs1;
        compress_stream::kernel_2a cs2;

        compress_stream_kernel_1<
            entropy_encoder_model_kernel_5<257,entropy_encoder::kernel_1a,4000000,4>,
            entropy_decoder_model_kernel_5<257,entropy_decoder::kernel_1a,4000000,4>,
            crc32::kernel_1a
            > cs3;


        print_spinner();

        rdata.clear();
        rdata.seekg(0);
        sout.clear();
        sout.str("");
        cs1.compress(rdata,sout);
        compressed_size = sout.str().size();
        compressed_size *= 8;
        compressed_size /= str.size();
        DLIB_TEST_MSG(compressed_size >= 8, "order 0 bps: " << compressed_size);
        dlog << LINFO << "order 0: " << compressed_size;

        print_spinner();

        rdata.clear();
        rdata.seekg(0);
        sout.clear();
        sout.str("");
        cs2.compress(rdata,sout);
        compressed_size = sout.str().size();
        compressed_size *= 8;
        compressed_size /= str.size();
        DLIB_TEST_MSG(compressed_size >= 8, "order 1 bps: " << compressed_size);
        dlog << LINFO << "order 1: " << compressed_size;

        print_spinner();

        rdata.clear();
        rdata.seekg(0);
        sout.clear();
        sout.str("");
        cs3.compress(rdata,sout);
        compressed_size = sout.str().size();
        compressed_size *= 8;
        compressed_size /= str.size();
        DLIB_TEST_MSG(compressed_size >= 8, "order 4 bps: " << compressed_size);
        dlog << LINFO << "order 4: " << compressed_size;

    }

    template <
        typename rand
        >
    void rand_test (
    )
    /*!
        requires
            - rand is an implementation of rand/rand_kernel_abstract.h 
              is instantiated with int
        ensures
            - runs tests on rand for compliance with the specs
    !*/
    {        

        ostringstream seed;
        seed << (unsigned int)time(0);

        ostringstream sout;


        rand r, r2;
        DLIB_TEST(r.get_seed() == "");
        r.set_seed(seed.str());

        DLIB_TEST(r.get_seed() == seed.str());
        r.clear();
        DLIB_TEST(r.get_seed() == "");
        swap(r,r2);
        DLIB_TEST(r.get_seed() == "");
        r.set_seed(seed.str());
        DLIB_TEST(r.get_seed() == seed.str());
        swap(r,r2);
        DLIB_TEST(r2.get_seed() == seed.str());
        DLIB_TEST(r.get_seed() == "");
        swap(r,r2);
        DLIB_TEST(r.get_seed() == seed.str());
        DLIB_TEST(r2.get_seed() == "");

        print_spinner();
        unsigned long size = 100000;
        for (unsigned long i = 0; i < size; ++i) 
        {
            uint32 ch = r.get_random_32bit_number();
            sout.write((char*)&ch,4);
        }

        check_bpp(sout.str());
        sout.clear();
        sout.str("");

        print_spinner();
        for (unsigned long i = 0; i < size; ++i) 
        {
            uint16 ch = r.get_random_16bit_number();
            sout.write((char*)&ch,2);
        }

        check_bpp(sout.str());
        sout.clear();
        sout.str("");

        print_spinner();
        for (unsigned long i = 0; i < size; ++i) 
        {
            unsigned char ch = r.get_random_8bit_number();
            sout.write((char*)&ch,1);
        }

        check_bpp(sout.str());
        sout.clear();
        sout.str("");


        // make sure the things can serialize right
        {
            r.clear();
            r2.clear();


            for (int i =0; i < 1000; ++i)
            {
                r.get_random_32bit_number();
                r.get_random_gaussian();
            }

            ostringstream sout;
            serialize(r, sout);

            istringstream sin(sout.str());
            deserialize(r2, sin);


            for (int i =0; i < 1000; ++i)
            {
                DLIB_TEST(r.get_random_32bit_number() == r2.get_random_32bit_number());
                DLIB_TEST(std::abs(r.get_random_gaussian() - r2.get_random_gaussian()) < 1e-14);
            }
        }


        // make sure calling clear() and set_seed("") do the same thing
        {
            r.clear();
            r2.set_seed("");
            rand r3;


            DLIB_TEST(r.get_seed() == r2.get_seed());
            DLIB_TEST(r.get_seed() == r3.get_seed());


            for (int i =0; i < 1000; ++i)
            {
                const uint32 num1 = r.get_random_32bit_number();
                const uint32 num2 = r2.get_random_32bit_number();
                const uint32 num3 = r3.get_random_32bit_number();
                DLIB_TEST( num1 == num2);
                DLIB_TEST( num1 == num3);
            }
        }

    }


    template <typename rand_type>
    void test_normal_numbers(
        rand_type& rnd
    )
    {
        print_spinner();
        dlog << LINFO << "test normality";
        double cnt1 = 0; // num <= -1.2
        double cnt2 = 0; // num <= -0.5 
        double cnt3 = 0; // num <= 0
        double cnt4 = 0; // num <= 0.5
        double cnt5 = 0; // num <= 1.2

        const unsigned long total = 1000000;
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = rnd.get_random_gaussian();
            if (r <= -1.2) cnt1 += 1;
            if (r <= -0.5) cnt2 += 1;
            if (r <=  0)   cnt3 += 1;
            if (r <=  0.5) cnt4 += 1;
            if (r <=  1.2) cnt5 += 1;
        }

        cnt1 /= total;
        cnt2 /= total;
        cnt3 /= total;
        cnt4 /= total;
        cnt5 /= total;

        dlog << LINFO << "cnt1: "<< cnt1;
        dlog << LINFO << "cnt2: "<< cnt2;
        dlog << LINFO << "cnt3: "<< cnt3;
        dlog << LINFO << "cnt4: "<< cnt4;
        dlog << LINFO << "cnt5: "<< cnt5;

        DLIB_TEST(std::abs(cnt1 - 0.11507) < 0.001);
        DLIB_TEST(std::abs(cnt2 - 0.30854) < 0.001);
        DLIB_TEST(std::abs(cnt3 - 0.5)     < 0.001);
        DLIB_TEST(std::abs(cnt4 - 0.69146) < 0.001);
        DLIB_TEST(std::abs(cnt5 - 0.88493) < 0.001);

    }

    void test_gaussian_random_hash()
    {
        print_spinner();
        dlog << LINFO << "test_gaussian_random_hash()";
        double cnt1 = 0; // num <= -1.2
        double cnt2 = 0; // num <= -0.5 
        double cnt3 = 0; // num <= 0
        double cnt4 = 0; // num <= 0.5
        double cnt5 = 0; // num <= 1.2

        const unsigned long total = 1000000;
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = gaussian_random_hash(i,0,0);
            if (r <= -1.2) cnt1 += 1;
            if (r <= -0.5) cnt2 += 1;
            if (r <=  0)   cnt3 += 1;
            if (r <=  0.5) cnt4 += 1;
            if (r <=  1.2) cnt5 += 1;
        }
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = gaussian_random_hash(0,i,0);
            if (r <= -1.2) cnt1 += 1;
            if (r <= -0.5) cnt2 += 1;
            if (r <=  0)   cnt3 += 1;
            if (r <=  0.5) cnt4 += 1;
            if (r <=  1.2) cnt5 += 1;
        }
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = gaussian_random_hash(0,0,i);
            if (r <= -1.2) cnt1 += 1;
            if (r <= -0.5) cnt2 += 1;
            if (r <=  0)   cnt3 += 1;
            if (r <=  0.5) cnt4 += 1;
            if (r <=  1.2) cnt5 += 1;
        }

        cnt1 /= total*3;
        cnt2 /= total*3;
        cnt3 /= total*3;
        cnt4 /= total*3;
        cnt5 /= total*3;

        dlog << LINFO << "cnt1: "<< cnt1;
        dlog << LINFO << "cnt2: "<< cnt2;
        dlog << LINFO << "cnt3: "<< cnt3;
        dlog << LINFO << "cnt4: "<< cnt4;
        dlog << LINFO << "cnt5: "<< cnt5;

        DLIB_TEST(std::abs(cnt1 - 0.11507) < 0.001);
        DLIB_TEST(std::abs(cnt2 - 0.30854) < 0.001);
        DLIB_TEST(std::abs(cnt3 - 0.5)     < 0.001);
        DLIB_TEST(std::abs(cnt4 - 0.69146) < 0.001);
        DLIB_TEST(std::abs(cnt5 - 0.88493) < 0.001);
    }

    void test_uniform_random_hash()
    {
        print_spinner();
        dlog << LINFO << "test_uniform_random_hash()";
        double cnt1 = 0; // num <= 0.2
        double cnt2 = 0; // num <= 0.4 
        double cnt3 = 0; // num <= 0.6
        double cnt4 = 0; // num <= 0.8
        double cnt5 = 0; // num <= 1.0

        double min_val = 10;
        double max_val = 0;

        const unsigned long total = 1000000;
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = uniform_random_hash(i,0,0);
            min_val = min(r,min_val);
            max_val = max(r,max_val);

            if (r <=  0.2) cnt1 += 1;
            if (r <=  0.4) cnt2 += 1;
            if (r <=  0.6) cnt3 += 1;
            if (r <=  0.8) cnt4 += 1;
            if (r <=  1.0) cnt5 += 1;
        }
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = uniform_random_hash(0,i,0);
            min_val = min(r,min_val);
            max_val = max(r,max_val);

            if (r <=  0.2) cnt1 += 1;
            if (r <=  0.4) cnt2 += 1;
            if (r <=  0.6) cnt3 += 1;
            if (r <=  0.8) cnt4 += 1;
            if (r <=  1.0) cnt5 += 1;
        }
        for (unsigned long i = 0; i < total; ++i)
        {
            const double r = uniform_random_hash(0,0,i);
            min_val = min(r,min_val);
            max_val = max(r,max_val);

            if (r <=  0.2) cnt1 += 1;
            if (r <=  0.4) cnt2 += 1;
            if (r <=  0.6) cnt3 += 1;
            if (r <=  0.8) cnt4 += 1;
            if (r <=  1.0) cnt5 += 1;
        }

        cnt1 /= total*3;
        cnt2 /= total*3;
        cnt3 /= total*3;
        cnt4 /= total*3;
        cnt5 /= total*3;

        dlog << LINFO << "cnt1: "<< cnt1;
        dlog << LINFO << "cnt2: "<< cnt2;
        dlog << LINFO << "cnt3: "<< cnt3;
        dlog << LINFO << "cnt4: "<< cnt4;
        dlog << LINFO << "cnt5: "<< cnt5;
        dlog << LINFO << "min_val: "<< min_val;
        dlog << LINFO << "max_val: "<< max_val;

        DLIB_TEST(std::abs(cnt1 - 0.2) < 0.001);
        DLIB_TEST(std::abs(cnt2 - 0.4) < 0.001);
        DLIB_TEST(std::abs(cnt3 - 0.6) < 0.001);
        DLIB_TEST(std::abs(cnt4 - 0.8) < 0.001);
        DLIB_TEST(std::abs(cnt5 - 1.0) < 0.001);
        DLIB_TEST(std::abs(min_val - 0.0) < 0.001);
        DLIB_TEST(std::abs(max_val - 1.0) < 0.001);
    }

    void test_get_integer()
    {

        print_spinner();
        dlib::rand rnd;


        int big_val = 0;
        int small_val = 0;

        const long long maxval = (((unsigned long long)1)<<62) + (((unsigned long long)1)<<61);
        for (int i = 0; i < 10000000; ++i)
        {
            if (rnd.get_integer(maxval) > maxval/2)
                ++big_val;
            else
                ++small_val;
        }

        // make sure there isn't any funny bias
        DLIB_TEST(std::abs(big_val/(double)small_val - 1) < 0.001);

        //cout << big_val/(double)small_val << endl;

    }

    void test_weibull_distribution()
    {
        print_spinner();
        dlib::rand rnd(0);

        const size_t N = 1024*1024*4;
        const double tol = 0.01;
        double k=1.0, lambda=2.0, g=6.0;

        dlib::running_stats<double> stats;
        for (size_t i = 0; i < N; i++) 
            stats.add(rnd.get_random_weibull(lambda, k, g));

        double expected_mean = g + lambda*std::tgamma(1 + 1.0 / k);
        double expected_var  = lambda*lambda*(std::tgamma(1 + 2.0 / k) - std::pow(std::tgamma(1 + 1.0 / k),2));
        DLIB_TEST(std::abs(stats.mean() - expected_mean) < tol);
        DLIB_TEST(std::abs(stats.variance() - expected_var) < tol);
    }

    void test_exponential_distribution()
    {
        print_spinner();
        dlib::rand rnd(0);

        const size_t N = 1024*1024*5;

        const double lambda = 1.5;
        print_spinner();
        dlib::running_stats<double> stats;
        for (size_t i = 0; i < N; i++) 
            stats.add(rnd.get_random_exponential(lambda));

        DLIB_TEST(std::abs(stats.mean() - 1.0 / lambda) < 0.001);
        DLIB_TEST(std::abs(stats.variance() - 1.0 / (lambda*lambda)) < 0.001);
        DLIB_TEST(std::abs(stats.skewness() - 2.0) < 0.01);
        DLIB_TEST(std::abs(stats.ex_kurtosis() - 6.0) < 0.1);
    }

    void test_beta_distribution()
    {
        print_spinner();
        dlib::rand rnd(0);

        const size_t N = 1024*1024*5;

        const double a = 0.2;
        const double b = 1.5;

        running_stats<double> stats;
        for (size_t i = 0; i < N; i++)
            stats.add(rnd.get_random_beta(a, b));

        const double expected_mean = a / (a + b);
        const double expected_var = a * b / (std::pow(a + b, 2) * (a + b + 1));
        DLIB_TEST(std::abs(stats.mean() - expected_mean) < 1e-5);
        DLIB_TEST(std::abs(stats.variance() - expected_var) < 1e-5);
    }

    void outputs_are_not_changed()
    {
        // dlib::rand has been around a really long time and it is a near certainty that there is
        // client code that depends on dlib::rand yielding the exact random sequence it happens to
        // yield for any given seed.  So we test that the output values of dlib::rand are not
        // changed in this test.

        {
            dlib::rand rnd;
            std::vector<uint32> out;
            for (int i = 0; i < 30; ++i) {
                out.push_back(rnd.get_random_32bit_number());
            }

            const std::vector<uint32> expected = {
                725333953,251387296,3200466189,2466988778,2049276419,2620437198,2806522923,
                2922190659,4151412029,2894696296,1344442829,1165961100,328304965,1533685458,
                3399102146,3995382051,1569312238,2353373514,2512982725,2903494783,787425157,
                699798098,330364342,2870851082,659976556,1726343583,3551405331,3171822159,
                1292599360,955731010};
            DLIB_TEST(out == expected);
        }

        {
            dlib::rand rnd;
            rnd.set_seed("this seed");
            std::vector<uint32> out;
            for (int i = 0; i < 30; ++i) {
                out.push_back(rnd.get_random_32bit_number());
            }

            const std::vector<uint32> expected = {
                856663397,2356564049,1192662566,3478257893,1069117227,
                1922448468,497418632,2504525324,987414451,769612124,77224022,2998161761,
                1364481427,639342008,1778351952,1931573847,3213816676,3019312695,4179936779,
                3637269252,4279821094,3738954922,3651625265,3159592157,333323775,4075800582,
                4237631248,357468843,483435718,1255945812};
            DLIB_TEST(out == expected);
        }

        {
            dlib::rand rnd;
            rnd.set_seed("some other seed");
            std::vector<int> out;
            for (int i = 0; i < 30; ++i) {
                out.push_back(rnd.get_integer(1000));
            }

            const std::vector<int> expected = {
                243,556,158,256,772,84,837,920,767,769,939,394,121,367,575,877,861,506,
                451,845,870,638,825,516,327,25,646,373,386,227};
            DLIB_TEST(out == expected);
        }

    }

    class rand_tester : public tester
    {
    public:
        rand_tester (
        ) :
            tester ("test_rand",
                    "Runs tests on the rand component.")
        {}

        void perform_test (
        )
        {
            dlog << LINFO << "testing kernel_1a";
            outputs_are_not_changed();

            rand_test<dlib::rand>();
            rand_test<dlib::rand>();

            dlib::rand rnd;
            test_normal_numbers(rnd);
            test_gaussian_random_hash();
            test_uniform_random_hash();
            test_get_integer();
            test_weibull_distribution();
            test_exponential_distribution();
            test_beta_distribution();
        }
    } a;

}


