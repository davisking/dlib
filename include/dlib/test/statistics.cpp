// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/statistics.h>
#include <dlib/statistics/running_gradient.h>
#include <dlib/rand.h>
#include <dlib/svm.h>
#include <algorithm>
#include <dlib/matrix.h>
#include <cmath>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.statistics");



    class statistics_tester : public tester
    {
    public:
        statistics_tester (
        ) :
            tester ("test_statistics",
                    "Runs tests on the statistics component.")
        {}

        void test_random_subset_selector ()
        {
            random_subset_selector<double> rand_set;

            for (int j = 0; j < 30; ++j)
            {
                print_spinner();

                running_stats<double> rs, rs2;

                rand_set.set_max_size(1000);

                for (double i = 0; i < 100000; ++i)
                {
                    rs.add(i);
                    rand_set.add(i);
                }


                for (unsigned long i = 0; i < rand_set.size(); ++i)
                    rs2.add(rand_set[i]);


                dlog << LDEBUG << "true mean:    " << rs.mean();
                dlog << LDEBUG << "true sampled: " << rs2.mean();
                double ratio = rs.mean()/rs2.mean();
                DLIB_TEST_MSG(0.96 < ratio  && ratio < 1.04, " ratio: " << ratio);
            }


            {
                random_subset_selector<int> r1, r2;
                r1.set_max_size(300);
                for (int i = 0; i < 4000; ++i)
                    r1.add(i);

                ostringstream sout;
                serialize(r1, sout);
                istringstream sin(sout.str());
                deserialize(r2, sin);

                DLIB_TEST(r1.size() == r2.size());
                DLIB_TEST(r1.max_size() == r2.max_size());
                DLIB_TEST(r1.next_add_accepts() == r2.next_add_accepts());
                DLIB_TEST(std::equal(r1.begin(), r1.end(), r2.begin()));

                for (int i = 0; i < 4000; ++i)
                {
                    r1.add(i);
                    r2.add(i);
                }

                DLIB_TEST(r1.size() == r2.size());
                DLIB_TEST(r1.max_size() == r2.max_size());
                DLIB_TEST(r1.next_add_accepts() == r2.next_add_accepts());
                DLIB_TEST(std::equal(r1.begin(), r1.end(), r2.begin()));
            }
        }

        void test_random_subset_selector2 ()
        {
            random_subset_selector<double> rand_set;
            DLIB_TEST(rand_set.next_add_accepts() == false);
            DLIB_TEST(rand_set.size() == 0);
            DLIB_TEST(rand_set.max_size() == 0);

            for (int j = 0; j < 30; ++j)
            {
                print_spinner();

                running_stats<double> rs, rs2;

                rand_set.set_max_size(1000);
                DLIB_TEST(rand_set.next_add_accepts() == true);

                for (double i = 0; i < 100000; ++i)
                {
                    rs.add(i);
                    if (rand_set.next_add_accepts())
                        rand_set.add(i);
                    else
                        rand_set.add();
                }

                DLIB_TEST(rand_set.size() == 1000);
                DLIB_TEST(rand_set.max_size() == 1000);

                for (unsigned long i = 0; i < rand_set.size(); ++i)
                    rs2.add(rand_set[i]);


                dlog << LDEBUG << "true mean:    " << rs.mean();
                dlog << LDEBUG << "true sampled: " << rs2.mean();
                double ratio = rs.mean()/rs2.mean();
                DLIB_TEST_MSG(0.96 < ratio  && ratio < 1.04, " ratio: " << ratio);
            }
        }

        void test_running_cross_covariance ()
        {
            running_cross_covariance<matrix<double> > rcc1, rcc2;

            matrix<double,0,1> xm, ym;
            const int num = 40;

            dlib::rand rnd;
            for (int i = 0; i < num; ++i)
            {
                matrix<double,0,1> x = randm(4,1,rnd);
                matrix<double,0,1> y = randm(4,1,rnd);

                xm += x/num;
                ym += y/num;

                if (i < 15)
                    rcc1.add(x,y);
                else
                    rcc2.add(x,y);
            }

            rnd.clear();
            matrix<double> cov;
            for (int i = 0; i < num; ++i)
            {
                matrix<double,0,1> x = randm(4,1,rnd);
                matrix<double,0,1> y = randm(4,1,rnd);
                cov += (x-xm)*trans(y-ym);
            }
            cov /= num-1;

            running_cross_covariance<matrix<double> > rcc = rcc1 + rcc2;
            DLIB_TEST(max(abs(rcc.covariance_xy()-cov)) < 1e-14);
            DLIB_TEST(max(abs(rcc.mean_x()-xm)) < 1e-14);
            DLIB_TEST(max(abs(rcc.mean_y()-ym)) < 1e-14);
        }

        std::map<unsigned long,double> dense_to_sparse ( 
            const matrix<double,0,1>& x
        )
        {
            std::map<unsigned long,double> temp;
            for (long i = 0; i < x.size(); ++i)
                temp[i] = x(i);
            return temp;
        }

        void test_running_cross_covariance_sparse()
        {
            running_cross_covariance<matrix<double> > rcc1, rcc2;

            running_covariance<matrix<double> > rc1, rc2;

            matrix<double,0,1> xm, ym;
            const int num = 40;

            rc1.set_dimension(4);
            rc2.set_dimension(4);

            rcc1.set_dimensions(4,5);
            rcc2.set_dimensions(4,5);

            dlib::rand rnd;
            for (int i = 0; i < num; ++i)
            {
                matrix<double,0,1> x = randm(4,1,rnd);
                matrix<double,0,1> y = randm(5,1,rnd);

                xm += x/num;
                ym += y/num;

                if (i < 15)
                {
                    rcc1.add(x,dense_to_sparse(y));
                    rc1.add(x);
                }
                else if (i < 30)
                {
                    rcc2.add(dense_to_sparse(x),y);
                    rc2.add(dense_to_sparse(x));
                }
                else
                {
                    rcc2.add(dense_to_sparse(x),dense_to_sparse(y));
                    rc2.add(x);
                }
            }

            rnd.clear();
            matrix<double> cov, cov2;
            for (int i = 0; i < num; ++i)
            {
                matrix<double,0,1> x = randm(4,1,rnd);
                matrix<double,0,1> y = randm(5,1,rnd);
                cov += (x-xm)*trans(y-ym);
                cov2 += (x-xm)*trans(x-xm);
            }
            cov /= num-1;
            cov2 /= num-1;

            running_cross_covariance<matrix<double> > rcc = rcc1 + rcc2;
            DLIB_TEST_MSG(max(abs(rcc.covariance_xy()-cov)) < 1e-14, max(abs(rcc.covariance_xy()-cov)));
            DLIB_TEST(max(abs(rcc.mean_x()-xm)) < 1e-14);
            DLIB_TEST(max(abs(rcc.mean_y()-ym)) < 1e-14);

            running_covariance<matrix<double> > rc = rc1 + rc2;
            DLIB_TEST(max(abs(rc.covariance()-cov2)) < 1e-14);
            DLIB_TEST(max(abs(rc.mean()-xm)) < 1e-14);
        }

        void test_running_covariance (
        )
        {
            dlib::rand rnd;
            std::vector<matrix<double,0,1> > vects;

            running_covariance<matrix<double,0,1> > cov, cov2;
            DLIB_TEST(cov.in_vector_size() == 0);

            for (unsigned long dims = 1; dims < 5; ++dims)
            {
                for (unsigned long samps = 2; samps < 10; ++samps)
                {
                    vects.clear();
                    cov.clear();
                    DLIB_TEST(cov.in_vector_size() == 0);
                    for (unsigned long i = 0; i < samps; ++i)
                    {
                        vects.push_back(randm(dims,1,rnd));
                        cov.add(vects.back());

                    }
                    DLIB_TEST(cov.in_vector_size() == (long)dims);

                    DLIB_TEST(equal(mean(mat(vects)), cov.mean()));
                    DLIB_TEST_MSG(equal(covariance(mat(vects)), cov.covariance()),
                              max(abs(covariance(mat(vects)) - cov.covariance()))
                              << "   dims = " << dims << "   samps = " << samps
                              );
                }
            }

            for (unsigned long dims = 1; dims < 5; ++dims)
            {
                for (unsigned long samps = 2; samps < 10; ++samps)
                {
                    vects.clear();
                    cov.clear();
                    cov2.clear();
                    DLIB_TEST(cov.in_vector_size() == 0);
                    for (unsigned long i = 0; i < samps; ++i)
                    {
                        vects.push_back(randm(dims,1,rnd));
                        if ((i%2) == 0)
                            cov.add(vects.back());
                        else
                            cov2.add(vects.back());

                    }
                    DLIB_TEST((cov+cov2).in_vector_size() == (long)dims);

                    DLIB_TEST(equal(mean(mat(vects)), (cov+cov2).mean()));
                    DLIB_TEST_MSG(equal(covariance(mat(vects)), (cov+cov2).covariance()),
                              max(abs(covariance(mat(vects)) - (cov+cov2).covariance()))
                              << "   dims = " << dims << "   samps = " << samps
                              );
                }
            }

        }

        void test_running_stats()
        {
            print_spinner();

            running_stats<double> rs, rs2;

            running_scalar_covariance<double> rsc1, rsc2;
            running_scalar_covariance_decayed<double> rscd1(1000000), rscd2(1000000);

            for (double i = 0; i < 100; ++i)
            {
                rs.add(i);

                rsc1.add(i,i);
                rsc2.add(i,i);
                rsc2.add(i,-i);

                rscd1.add(i,i);
                rscd2.add(i,i);
                rscd2.add(i,-i);
            }

            // make sure the running_stats and running_scalar_covariance agree
            DLIB_TEST_MSG(std::abs(rs.mean() - rsc1.mean_x()) < 1e-10, std::abs(rs.mean() - rsc1.mean_x()));
            DLIB_TEST(std::abs(rs.mean() - rsc1.mean_y()) < 1e-10);
            DLIB_TEST(std::abs(rs.stddev() - rsc1.stddev_x()) < 1e-10);
            DLIB_TEST(std::abs(rs.stddev() - rsc1.stddev_y()) < 1e-10);
            DLIB_TEST(std::abs(rs.variance() - rsc1.variance_x()) < 1e-10);
            DLIB_TEST(std::abs(rs.variance() - rsc1.variance_y()) < 1e-10);
            DLIB_TEST(rs.current_n() == rsc1.current_n());

            DLIB_TEST(std::abs(rsc1.correlation() - 1) < 1e-10);
            DLIB_TEST(std::abs(rsc2.correlation() - 0) < 1e-10);


            DLIB_TEST_MSG(std::abs(rs.mean() - rscd1.mean_x()) < 1e-2, std::abs(rs.mean() - rscd1.mean_x()) << " " << rscd1.mean_x());
            DLIB_TEST(std::abs(rs.mean() - rscd1.mean_y()) < 1e-2);
            DLIB_TEST_MSG(std::abs(rs.stddev() - rscd1.stddev_x()) < 1e-2, std::abs(rs.stddev() - rscd1.stddev_x()));
            DLIB_TEST(std::abs(rs.stddev() - rscd1.stddev_y()) < 1e-2);
            DLIB_TEST_MSG(std::abs(rs.variance() - rscd1.variance_x()) < 1e-2, std::abs(rs.variance() - rscd1.variance_x()));
            DLIB_TEST(std::abs(rs.variance() - rscd1.variance_y()) < 1e-2);
            DLIB_TEST(std::abs(rscd1.correlation() - 1) < 1e-2);
            DLIB_TEST(std::abs(rscd2.correlation() - 0) < 1e-2);



            // test serialization of running_stats
            ostringstream sout;
            serialize(rs, sout);
            istringstream sin(sout.str());
            deserialize(rs2, sin);
            // make sure the running_stats and running_scalar_covariance agree
            DLIB_TEST_MSG(std::abs(rs2.mean() - rsc1.mean_x()) < 1e-10, std::abs(rs2.mean() - rsc1.mean_x()));
            DLIB_TEST(std::abs(rs2.mean() - rsc1.mean_y()) < 1e-10);
            DLIB_TEST(std::abs(rs2.stddev() - rsc1.stddev_x()) < 1e-10);
            DLIB_TEST(std::abs(rs2.stddev() - rsc1.stddev_y()) < 1e-10);
            DLIB_TEST(std::abs(rs2.variance() - rsc1.variance_x()) < 1e-10);
            DLIB_TEST(std::abs(rs2.variance() - rsc1.variance_y()) < 1e-10);
            DLIB_TEST(rs2.current_n() == rsc1.current_n());

            rsc1.clear();
            rsc1.add(1, -1);
            rsc1.add(0, 0);
            rsc1.add(1, -1);
            rsc1.add(0, 0);
            rsc1.add(1, -1);
            rsc1.add(0, 0);

            DLIB_TEST(std::abs(rsc1.covariance() - -0.3) < 1e-10);
        }

        void test_skewness_and_kurtosis_1()
        {

            dlib::rand rnum;
            running_stats<double> rs1;

            double tp = 0;

            rnum.set_seed("DlibRocks");

            for(int i = 0; i< 1000000; i++)
            {
                tp = rnum.get_random_gaussian();
                rs1.add(tp);
            }   

            // check the unbiased skewness and excess kurtosis of one million Gaussian
            // draws are both near_vects zero.
            DLIB_TEST(abs(rs1.skewness()) < 0.1);
            DLIB_TEST(abs(rs1.ex_kurtosis()) < 0.1);
        }

        void test_skewness_and_kurtosis_2()
        {

            string str = "DlibRocks";

            for(int j = 0; j<5 ; j++)
            {
                matrix<double,1,100000> dat;
                dlib::rand rnum;
                running_stats<double> rs1;
     
                double tp = 0;
                double n = 100000;
                double xb = 0;
    
                double sknum = 0;
                double skdenom = 0;
                double unbi_skew = 0;
    
                double exkurnum = 0;
                double exkurdenom = 0;
                double unbi_exkur = 0;

                random_shuffle(str.begin(), str.end());
                rnum.set_seed(str);

                for(int i = 0; i<n; i++)
                {
                    tp = rnum.get_random_gaussian();
                    rs1.add(tp);
                    dat(i)=tp;
                    xb += dat(i);
                }   
    
                xb = xb/n;

                for(int i = 0; i < n; i++ )
                { 
                    sknum += pow(dat(i) - xb,3);
                    skdenom += pow(dat(i) - xb,2);
                    exkurnum += pow(dat(i) - xb,4);
                    exkurdenom += pow(dat(i)-xb,2);
                }

                sknum = sknum/n;
                skdenom = pow(skdenom/n,1.5);
                exkurnum = exkurnum/n;
                exkurdenom = pow(exkurdenom/n,2);
    
                unbi_skew = sqrt(n*(n-1))/(n-2)*sknum/skdenom;
                unbi_exkur = (n-1)*((n+1)*(exkurnum/exkurdenom-3)+6)/((n-2)*(n-3));

                dlog << LINFO << "Skew Diff: " <<  unbi_skew - rs1.skewness();
                dlog << LINFO << "Kur Diff: " << unbi_exkur - rs1.ex_kurtosis();
                
                // Test an alternative implementation of the unbiased skewness and excess
                // kurtosis against the one in running_stats.
                DLIB_TEST(abs(unbi_skew - rs1.skewness()) < 1e-10);
                DLIB_TEST(abs(unbi_exkur - rs1.ex_kurtosis()) < 1e-10);
            }
        }

        void test_randomize_samples()
        {
            std::vector<unsigned int> t(15),u(15),v(15);

            for (unsigned long i = 0; i < t.size(); ++i)
            {
                t[i] = i;
                u[i] = i+1;
                v[i] = i+2;
            }
            randomize_samples(t,u,v);

            DLIB_TEST(t.size() == 15);
            DLIB_TEST(u.size() == 15);
            DLIB_TEST(v.size() == 15);

            for (unsigned long i = 0; i < t.size(); ++i)
            {
                const unsigned long val = t[i];
                DLIB_TEST(u[i] == val+1);
                DLIB_TEST(v[i] == val+2);
            }
        }
        void test_randomize_samples2()
        {
            dlib::matrix<int,15,1> t(15),u(15),v(15);

            for (long i = 0; i < t.size(); ++i)
            {
                t(i) = i;
                u(i) = i+1;
                v(i) = i+2;
            }
            randomize_samples(t,u,v);

            DLIB_TEST(t.size() == 15);
            DLIB_TEST(u.size() == 15);
            DLIB_TEST(v.size() == 15);

            for (long i = 0; i < t.size(); ++i)
            {
                const long val = t(i);
                DLIB_TEST(u(i) == val+1);
                DLIB_TEST(v(i) == val+2);
            }
        }

        void another_test()
        {
            std::vector<double> a;

            running_stats<double> rs1, rs2;

            for (int i = 0; i < 10; ++i)
            {
                rs1.add(i);
                a.push_back(i);
            }

            DLIB_TEST(std::abs(variance(mat(a)) - rs1.variance()) < 1e-13);
            DLIB_TEST(std::abs(stddev(mat(a)) - rs1.stddev()) < 1e-13);
            DLIB_TEST(std::abs(mean(mat(a)) - rs1.mean()) < 1e-13);

            for (int i = 10; i < 20; ++i)
            {
                rs2.add(i);
                a.push_back(i);
            }

            DLIB_TEST(std::abs(variance(mat(a)) - (rs1+rs2).variance()) < 1e-13);
            DLIB_TEST(std::abs(mean(mat(a)) - (rs1+rs2).mean()) < 1e-13);
            DLIB_TEST((rs1+rs2).current_n() == 20);

            running_scalar_covariance<double> rc1, rc2, rc3;
            dlib::rand rnd;
            for (double i = 0; i < 10; ++i)
            {
                const double a = i + rnd.get_random_gaussian();
                const double b = i + rnd.get_random_gaussian();
                rc1.add(a,b);
                rc3.add(a,b);
            }
            for (double i = 11; i < 20; ++i)
            {
                const double a = i + rnd.get_random_gaussian();
                const double b = i + rnd.get_random_gaussian();
                rc2.add(a,b);
                rc3.add(a,b);
            }

            DLIB_TEST(std::abs((rc1+rc2).mean_x() - rc3.mean_x()) < 1e-13);
            DLIB_TEST(std::abs((rc1+rc2).mean_y() - rc3.mean_y()) < 1e-13);
            DLIB_TEST_MSG(std::abs((rc1+rc2).variance_x() - rc3.variance_x()) < 1e-13, std::abs((rc1+rc2).variance_x() - rc3.variance_x()));
            DLIB_TEST(std::abs((rc1+rc2).variance_y() - rc3.variance_y()) < 1e-13);
            DLIB_TEST(std::abs((rc1+rc2).covariance() - rc3.covariance()) < 1e-13);
            DLIB_TEST((rc1+rc2).current_n() == rc3.current_n());

        }

        void test_average_precision()
        {
            std::vector<bool> items;
            DLIB_TEST(average_precision(items) == 1);
            DLIB_TEST(average_precision(items,1) == 0);

            items.push_back(true);
            DLIB_TEST(average_precision(items) == 1);
            DLIB_TEST(std::abs(average_precision(items,1) - 0.5) < 1e-14);

            items.push_back(true);
            DLIB_TEST(average_precision(items) == 1);
            DLIB_TEST(std::abs(average_precision(items,1) - 2.0/3.0) < 1e-14);

            items.push_back(false);

            DLIB_TEST(average_precision(items) == 1);
            DLIB_TEST(std::abs(average_precision(items,1) - 2.0/3.0) < 1e-14);

            items.push_back(true);

            DLIB_TEST(std::abs(average_precision(items) - (2.0+3.0/4.0)/3.0) < 1e-14);

            items.push_back(true);

            DLIB_TEST(std::abs(average_precision(items)   - (2.0 + 4.0/5.0 + 4.0/5.0)/4.0) < 1e-14);
            DLIB_TEST(std::abs(average_precision(items,1) - (2.0 + 4.0/5.0 + 4.0/5.0)/5.0) < 1e-14);
        }


        template <typename sample_type>
        void check_distance_metrics (
            const std::vector<frobmetric_training_sample<sample_type> >& samples
        )
        {
            running_stats<double> rs;
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                for (unsigned long j = 0; j < samples[i].near_vects.size(); ++j)
                {
                    const double d1 = length_squared(samples[i].anchor_vect - samples[i].near_vects[j]);
                    for (unsigned long k = 0; k < samples[i].far_vects.size(); ++k)
                    {
                        const double d2 = length_squared(samples[i].anchor_vect - samples[i].far_vects[k]);
                        rs.add(d2-d1);
                    }
                }
            }

            dlog << LINFO << "dist gap max:    "<< rs.max();
            dlog << LINFO << "dist gap min:    "<< rs.min();
            dlog << LINFO << "dist gap mean:   "<< rs.mean();
            dlog << LINFO << "dist gap stddev: "<< rs.stddev();
            DLIB_TEST(rs.min() >= 0.99);
            DLIB_TEST(rs.mean() >= 0.9999);
        }

        void test_vector_normalizer_frobmetric(dlib::rand& rnd)
        { 
            print_spinner();
            typedef matrix<double,0,1> sample_type;
            vector_normalizer_frobmetric<sample_type> normalizer;

            std::vector<frobmetric_training_sample<sample_type> > samples;
            frobmetric_training_sample<sample_type> samp;

            const long key = 1;
            const long dims = 5;
            // Lets make some two class training data.  Each sample will have dims dimensions but
            // only the one with index equal to key will be meaningful.  In particular, if the key
            // dimension is > 0 then the sample is class +1 and -1 otherwise.  

            long k = 0;
            for (int i = 0; i < 50; ++i)
            {
                samp.clear();
                samp.anchor_vect = gaussian_randm(dims,1,k++);
                if (samp.anchor_vect(key) > 0)
                    samp.anchor_vect(key) = rnd.get_random_double() + 5;
                else
                    samp.anchor_vect(key) = -(rnd.get_random_double() + 5);

                matrix<double,0,1> temp;

                for (int j = 0; j < 5; ++j)
                {
                    // Don't always put an equal number of near_vects and far_vects vectors into the
                    // training samples.
                    const int numa = rnd.get_random_32bit_number()%2 + 1;
                    const int numb = rnd.get_random_32bit_number()%2 + 1;

                    for (int num = 0; num < numa; ++num)
                    {
                        temp = gaussian_randm(dims,1,k++); temp(key) = 0.1;
                        //temp = gaussian_randm(dims,1,k++); temp(key) = std::abs(temp(key));
                        if (samp.anchor_vect(key) > 0) samp.near_vects.push_back(temp);
                        else                    samp.far_vects.push_back(temp);
                    }

                    for (int num = 0; num < numb; ++num)
                    {
                        temp = gaussian_randm(dims,1,k++); temp(key) = -0.1;
                        //temp = gaussian_randm(dims,1,k++); temp(key) = -std::abs(temp(key));
                        if (samp.anchor_vect(key) < 0) samp.near_vects.push_back(temp);
                        else                    samp.far_vects.push_back(temp);
                    }
                }
                samples.push_back(samp);
            }

            normalizer.set_epsilon(0.0001);
            normalizer.set_c(100);
            normalizer.set_max_iterations(6000);
            normalizer.train(samples);

            dlog << LINFO << "learned transform: \n" << normalizer.transform();

            matrix<double,0,1> total;

            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                samples[i].anchor_vect = normalizer(samples[i].anchor_vect);
                total += samples[i].anchor_vect;
                for (unsigned long j = 0; j < samples[i].near_vects.size(); ++j)
                    samples[i].near_vects[j] = normalizer(samples[i].near_vects[j]);
                for (unsigned long j = 0; j < samples[i].far_vects.size(); ++j)
                    samples[i].far_vects[j] = normalizer(samples[i].far_vects[j]);
            }
            total /= samples.size();
            dlog << LINFO << "sample transformed means: "<< trans(total);
            DLIB_TEST(length(total) < 1e-9);
            check_distance_metrics(samples);

            // make sure serialization works
            stringstream os;
            serialize(normalizer, os);
            vector_normalizer_frobmetric<sample_type> normalizer2;
            deserialize(normalizer2, os);
            DLIB_TEST(equal(normalizer.transform(), normalizer2.transform()));
            DLIB_TEST(equal(normalizer.transformed_means(), normalizer2.transformed_means()));
            DLIB_TEST(normalizer.in_vector_size() == normalizer2.in_vector_size());
            DLIB_TEST(normalizer.out_vector_size() == normalizer2.out_vector_size());
            DLIB_TEST(normalizer.get_max_iterations() == normalizer2.get_max_iterations());
            DLIB_TEST(std::abs(normalizer.get_c() - normalizer2.get_c()) < 1e-14);
            DLIB_TEST(std::abs(normalizer.get_epsilon() - normalizer2.get_epsilon()) < 1e-14);

        }

        void prior_frobnorm_test()
        {
            frobmetric_training_sample<matrix<double,0,1> > sample;
            std::vector<frobmetric_training_sample<matrix<double,0,1> > > samples;

            matrix<double,3,1> x, near_, far_;
            x    = 0,0,0;
            near_ = 1,0,0;
            far_  = 0,1,0;

            sample.anchor_vect = x;
            sample.near_vects.push_back(near_);
            sample.far_vects.push_back(far_);

            samples.push_back(sample);

            vector_normalizer_frobmetric<matrix<double,0,1> > trainer;
            trainer.set_c(100);
            print_spinner();
            trainer.train(samples);

            matrix<double,3,3> correct;
            correct = 0, 0, 0,
                      0, 1, 0, 
                      0, 0, 0;

            dlog << LDEBUG << trainer.transform();
            DLIB_TEST(max(abs(trainer.transform()-correct)) < 1e-8);

            trainer.set_uses_identity_matrix_prior(true);
            print_spinner();
            trainer.train(samples);
            correct = 1, 0, 0,
                      0, 2, 0, 
                      0, 0, 1;

            dlog << LDEBUG << trainer.transform();
            DLIB_TEST(max(abs(trainer.transform()-correct)) < 1e-8);

        }

        void test_lda ()
        {
            // This test makes sure we pick the right direction in a simple 2D -> 1D LDA
            typedef matrix<double,2,1> sample_type;

            std::vector<unsigned long> labels;
            std::vector<sample_type> samples;
            for (int i=0; i<4; i++)
            {
                sample_type s;
                s(0) = i;
                s(1) = i+1;
                samples.push_back(s);
                labels.push_back(1);       

                sample_type s1;
                s1(0) = i+1;
                s1(1) = i;
                samples.push_back(s1);      
                labels.push_back(2);       
            }

            matrix<double> X;  
            X.set_size(8,2);
            for (int i=0; i<8; i++){
                X(i,0) = samples[i](0);
                X(i,1) = samples[i](1);
            }  

            matrix<double,0,1> mean;   

            dlib::compute_lda_transform(X,mean,labels,1);

            std::vector<double> vals1, vals2;
            for (unsigned long i = 0; i < samples.size(); ++i)
            {
                double val = X*samples[i]-mean;
                if (i%2 == 0)
                    vals1.push_back(val);
                else
                    vals2.push_back(val);
                dlog << LINFO << "1D LDA output: " << val;
            }

            if (vals1[0] > vals2[0])
                swap(vals1, vals2);

            const double err = equal_error_rate(vals1, vals2).first;
            dlog << LINFO << "LDA ERR: " << err;
            DLIB_TEST(err == 0);
            DLIB_TEST(equal_error_rate(vals2, vals1).first == 1);
        }

        void test_equal_error_rate() 
        {
            auto result = equal_error_rate({}, {});
            DLIB_TEST(result.first == 0);
            DLIB_TEST(result.second == 0);

            // no error case
            result = equal_error_rate({1,1,1}, {2,2,2});
            DLIB_TEST_MSG(result.first == 0, result.first);
            DLIB_TEST_MSG(result.second == 2, result.second);

            // max error case
            result = equal_error_rate({2,2,2}, {1,1,1});
            DLIB_TEST_MSG(result.first == 1, result.first);
            DLIB_TEST_MSG(result.second == 2, result.second);
            // Another way to have max error
            result = equal_error_rate({1,1,1}, {1,1,1});
            DLIB_TEST_MSG(result.second == 1, result.second);
            DLIB_TEST_MSG(result.first == 1, result.first);

            // wildly unbalanced  
            result = equal_error_rate({}, {1,1,1});
            DLIB_TEST_MSG(result.first == 0, result.first);

            // wildly unbalanced  
            result = equal_error_rate({1,1,1}, {});
            DLIB_TEST_MSG(result.first == 0, result.first);

            // 25% error case   
            result = equal_error_rate({1,1,1,3}, {2, 2, 0, 2});
            DLIB_TEST_MSG(result.first == 0.25, result.first);
            DLIB_TEST_MSG(result.second == 2, result.second);
        }

        void test_running_stats_decayed()
        {
            print_spinner();
            std::vector<double> tmp(300);
            std::vector<double> tmp_var(tmp.size());
            dlib::rand rnd;
            const int num_rounds = 100000;
            for (int rounds = 0; rounds < num_rounds; ++rounds)
            {
                running_stats_decayed<double> rs(100);

                for (size_t i = 0; i < tmp.size(); ++i)
                {
                    rs.add(rnd.get_random_gaussian() + 1);
                    tmp[i] += rs.mean();
                    if (i > 0)
                        tmp_var[i] += rs.variance();
                }
            }

            // should print all 1s basically since the mean and variance should always be 1.
            for (size_t i = 0; i < tmp.size(); ++i)
            {
                DLIB_TEST(std::abs(1-tmp[i]/num_rounds) < 0.001);
                if (i > 1)
                    DLIB_TEST(std::abs(1-tmp_var[i]/num_rounds) < 0.01);
            }
        }

        void test_running_scalar_covariance_decayed()
        {
            print_spinner();
            std::vector<double> tmp(300);
            std::vector<double> tmp_var(tmp.size());
            std::vector<double> tmp_covar(tmp.size());
            dlib::rand rnd;
            const int num_rounds = 500000;
            for (int rounds = 0; rounds < num_rounds; ++rounds)
            {
                running_scalar_covariance_decayed<double> rs(100);

                for (size_t i = 0; i < tmp.size(); ++i)
                {
                    rs.add(rnd.get_random_gaussian() + 1, rnd.get_random_gaussian() + 1);
                    tmp[i] += (rs.mean_y()+rs.mean_x())/2;
                    if (i > 0)
                    {
                        tmp_var[i] += (rs.variance_y()+rs.variance_x())/2;
                        tmp_covar[i] += rs.covariance();
                    }
                }
            }

            // should print all 1s basically since the mean and variance should always be 1.
            for (size_t i = 0; i < tmp.size(); ++i)
            {
                DLIB_TEST(std::abs(1-tmp[i]/num_rounds) < 0.001);
                if (i > 1)
                {
                    DLIB_TEST(std::abs(1-tmp_var[i]/num_rounds) < 0.01);
                    DLIB_TEST(std::abs(tmp_covar[i]/num_rounds) < 0.001);
                }
            }
        }

        void test_probability_values_are_increasing() {
            DLIB_TEST(probability_values_are_increasing(std::vector<double>{1,2,3,4,5,6,7,8}) > 0.99);
            DLIB_TEST(probability_values_are_increasing(std::vector<double>{8,7,6,5,4,4,3,2}) < 0.01);
            DLIB_TEST(probability_values_are_increasing_robust(std::vector<double>{1,2,3,4,5,6,7,8}) > 0.99);
            DLIB_TEST(probability_values_are_increasing_robust(std::vector<double>{8,7,6,5,4,4,3,2}) < 0.01);
            DLIB_TEST(probability_values_are_increasing(std::vector<double>{1,2,1e10,3,4,5,6,7,8}) < 0.3);
            DLIB_TEST(probability_values_are_increasing_robust(std::vector<double>{1,2,1e100,3,4,5,6,7,8}) > 0.99);
        }

        void test_event_corr()
        {
            print_spinner();
            DLIB_TEST(event_correlation(1000,1000,500,2000) == 0);
            DLIB_TEST(std::abs(event_correlation(1000,1000,300,2000) + 164.565757010104) < 1e-11);
            DLIB_TEST(std::abs(event_correlation(1000,1000,700,2000) - 164.565757010104) < 1e-11);

            DLIB_TEST(event_correlation(10,1000,5,2000) == 0);
            DLIB_TEST(event_correlation(1000,10,5,2000) == 0);
            DLIB_TEST(std::abs(event_correlation(10,1000,1,2000) - event_correlation(1000,10,1,2000)) < 1e-11);
            DLIB_TEST(std::abs(event_correlation(10,1000,9,2000) - event_correlation(1000,10,9,2000)) < 1e-11);

            DLIB_TEST(std::abs(event_correlation(10,1000,1,2000) + 3.69672251700842) < 1e-11);
            DLIB_TEST(std::abs(event_correlation(10,1000,9,2000) - 3.69672251700842) < 1e-11);
        }

        void perform_test (
        )
        {
            prior_frobnorm_test();
            dlib::rand rnd;
            for (int i = 0; i < 5; ++i)
                test_vector_normalizer_frobmetric(rnd);

            test_random_subset_selector();
            test_random_subset_selector2();
            test_running_covariance();
            test_running_cross_covariance();
            test_running_cross_covariance_sparse();
            test_running_stats();
            test_skewness_and_kurtosis_1();
            test_skewness_and_kurtosis_2();
            test_randomize_samples();
            test_randomize_samples2();
            another_test();
            test_average_precision();
            test_lda();
            test_event_corr();
            test_running_stats_decayed();
            test_running_scalar_covariance_decayed();
            test_equal_error_rate();
            test_probability_values_are_increasing();
        }
    } a;

}


