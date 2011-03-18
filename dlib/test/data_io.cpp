// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/svm.h>
#include <dlib/data_io.h>
#include "create_iris_datafile.h"
#include <vector>
#include <sstream>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.data_io");


    class test_data_io : public tester
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a unit test.  When it is constructed
                it adds itself into the testing framework.
        !*/
    public:
        test_data_io (
        ) :
            tester (
                "test_data_io",       // the command line argument name for this test
                "Run tests on the data_io stuff.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }


        template <typename sample_type>
        void run_test()
        {
            print_spinner();

            typedef typename sample_type::value_type::second_type scalar_type;

            std::vector<sample_type> samples;
            std::vector<scalar_type> labels;

            load_libsvm_formatted_data("iris.scale",samples, labels);
            save_libsvm_formatted_data("iris.scale2", samples, labels);

            DLIB_TEST(samples.size() == 150);
            DLIB_TEST(labels.size() == 150);
            DLIB_TEST(sparse_vector::max_index_plus_one(samples) == 5);
            fix_nonzero_indexing(samples);
            DLIB_TEST(sparse_vector::max_index_plus_one(samples) == 4);

            load_libsvm_formatted_data("iris.scale2",samples, labels);

            DLIB_TEST(samples.size() == 150);
            DLIB_TEST(labels.size() == 150);

            DLIB_TEST(sparse_vector::max_index_plus_one(samples) == 5);
            fix_nonzero_indexing(samples);
            DLIB_TEST(sparse_vector::max_index_plus_one(samples) == 4);

            one_vs_one_trainer<any_trainer<sample_type,scalar_type>,scalar_type> trainer;

            typedef sparse_linear_kernel<sample_type> kernel_type;
            trainer.set_trainer(krr_trainer<kernel_type>());

            randomize_samples(samples, labels);
            matrix<double> cv = cross_validate_multiclass_trainer(trainer, samples, labels, 4);

            dlog << LINFO << "confusion matrix: \n" << cv;
            const scalar_type cv_accuracy = sum(diag(cv))/sum(cv);
            dlog << LINFO << "cv accuracy: " << cv_accuracy;
            DLIB_TEST(cv_accuracy > 0.97);




            {
                print_spinner();
                typedef matrix<scalar_type,0,1> dsample_type;
                std::vector<dsample_type> dsamples = sparse_to_dense(samples);
                DLIB_TEST(dsamples.size() == 150);
                DLIB_TEST(dsamples[0].size() == 4);
                DLIB_TEST(sparse_vector::max_index_plus_one(dsamples) == 4);

                one_vs_one_trainer<any_trainer<dsample_type,scalar_type>,scalar_type> trainer;

                typedef linear_kernel<dsample_type> kernel_type;
                trainer.set_trainer(rr_trainer<kernel_type>());

                cv = cross_validate_multiclass_trainer(trainer, dsamples, labels, 4);

                dlog << LINFO << "dense confusion matrix: \n" << cv;
                const scalar_type cv_accuracy = sum(diag(cv))/sum(cv);
                dlog << LINFO << "dense cv accuracy: " << cv_accuracy;
                DLIB_TEST(cv_accuracy > 0.97);
            }

        }




        void perform_test (
        )
        {
            print_spinner();
            create_iris_datafile();

            run_test<std::map<unsigned int, double> >();
            run_test<std::map<unsigned int, float> >();
            run_test<std::vector<std::pair<unsigned int, float> > >();
            run_test<std::vector<std::pair<unsigned long, double> > >();
        }
    };

    test_data_io a;

}


