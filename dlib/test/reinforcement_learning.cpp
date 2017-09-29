// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/control.h>
#include <vector>
#include <sstream>
#include <ctime>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.rl");

    template <bool have_prior>
    struct chain_model
    {
        typedef int state_type;
        typedef int action_type; // 0 is move left, 1 is move right
        const static bool force_last_weight_to_1 = have_prior;


        const static int num_states = 4; // not required in the model interface

        matrix<double,8,1> offset;
        chain_model()
        {
            offset = 
                2.048 ,
                2.56 ,
                2.048 ,
                3.2 ,
                2.56 ,
                4 ,
                3.2, 
                5 ;
            if (!have_prior)
                offset = 0;

        }

        unsigned long num_features(
        ) const 
        {
            if (have_prior)
                return num_states*2 + 1; 
            else
                return num_states*2; 
        }

        action_type find_best_action (
            const state_type& state,
            const matrix<double,0,1>& w
        ) const
        {
            if (w(state*2)+offset(state*2) >= w(state*2+1)+offset(state*2+1))
                //if (w(state*2) >= w(state*2+1))
                return 0;
            else
                return 1;
        }

        void get_features (
            const state_type& state,
            const action_type& action,
            matrix<double,0,1>& feats
        ) const
        {
            feats.set_size(num_features());
            feats = 0;
            feats(state*2 + action) = 1;
            if (have_prior)
                feats(num_features()-1) = offset(state*2+action);
        }

    };

    typedef extended_process_sample<chain_model<true> > extended_sample_type;

    void test_lspi_prior1()
    {
        typedef process_sample<chain_model<true> > sample_type;

        print_spinner();
        std::vector<sample_type> samples;

        samples.push_back(sample_type(0,0,0,0));
        samples.push_back(sample_type(0,1,1,0));

        samples.push_back(sample_type(1,0,0,0));
        samples.push_back(sample_type(1,1,2,0));

        samples.push_back(sample_type(2,0,1,0));
        samples.push_back(sample_type(2,1,3,0));

        samples.push_back(sample_type(3,0,2,0));
        samples.push_back(sample_type(3,1,3,1));

        lspi<chain_model<true> > trainer;
        //trainer.be_verbose();
        trainer.set_lambda(0);
        policy<chain_model<true> > pol = trainer.train(samples);

        dlog << LINFO << pol.get_weights();

        matrix<double,0,1> w = pol.get_weights();
        DLIB_TEST(pol.get_weights().size() == 9);
        DLIB_TEST(w(w.size()-1) == 1);
        w(w.size()-1) = 0;
        DLIB_TEST_MSG(length(w) < 1e-12, length(w));

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 1, 1};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    void test_lspi_prior2()
    {
        typedef process_sample<chain_model<true> > sample_type;

        print_spinner();
        std::vector<sample_type> samples;

        samples.push_back(sample_type(0,0,0,0));
        samples.push_back(sample_type(0,1,1,0));

        samples.push_back(sample_type(1,0,0,0));
        samples.push_back(sample_type(1,1,2,0));

        samples.push_back(sample_type(2,0,1,0));
        samples.push_back(sample_type(2,1,3,1));

        samples.push_back(sample_type(3,0,2,0));
        samples.push_back(sample_type(3,1,3,0));

        lspi<chain_model<true> > trainer;
        //trainer.be_verbose();
        trainer.set_lambda(0);
        policy<chain_model<true> > pol = trainer.train(samples);

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 1, 0};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    void test_lspi_noprior1(std::vector<extended_sample_type> samples)
    {
        print_spinner();

        lspi<chain_model<false> > trainer;
        //trainer.be_verbose();
        policy<chain_model<false> > pol = trainer.train(samples);

        dlog << LINFO << pol.get_weights();
        DLIB_TEST(pol.get_weights().size() == 8);

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 1, 1};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    void test_lspi_noprior2(std::vector<extended_sample_type> samples)
    {
        print_spinner();

        lspi<chain_model<false> > trainer;
        //trainer.be_verbose();
        policy<chain_model<false> > pol = trainer.train(samples);

        dlog << LINFO << pol.get_weights();
        DLIB_TEST(pol.get_weights().size() == 8);

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 0, 0};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    template <
            template<typename> typename control_agent
            >
    void test_ql_sarsa_noprior1(std::vector<std::vector<extended_sample_type> > trials, bool normal)
    {
        print_spinner();

        control_agent<chain_model<false> > trainer;
        //trainer.be_verbose();

        policy<chain_model<false> > pol;
        if(normal)
            pol = trainer.train(trials);
        else
            pol = trainer.train(trials.begin(), trials.end());

        dlog << LINFO << pol.get_weights();
        DLIB_TEST(pol.get_weights().size() == 8);

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 1, 1};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    template <
            template<typename> typename control_agent
            >
    void test_ql_sarsa_noprior2(std::vector<std::vector<extended_sample_type> > trials, bool normal)
    {
        print_spinner();

        control_agent<chain_model<false> > trainer;
        //trainer.be_verbose();
        policy<chain_model<false> > pol;
        if(normal)
            pol = trainer.train(trials);
        else
            pol = trainer.train(trials.begin(), trials.end());

        dlog << LINFO << pol.get_weights();
        DLIB_TEST(pol.get_weights().size() == 8);

        for(unsigned int i = 0; i < 4; i++)
            dlog << LINFO << "action: " << pol(i);

        int correct[] = {1, 1, 0, 0};
        for(unsigned int i = 0; i < 4; i++)
            DLIB_TEST(pol(i) == correct[i]);
    }

    class rl_tester : public tester
    {
    public:
        rl_tester (
        ) :
            tester (
                "test_rl",       // the command line argument name for this test
                "Run tests on the reinforcement learning objects (lspi, qlearning & sarsa).", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }

        void perform_test (
        )
        {
            std::vector<extended_sample_type> samples_noprior1;
            samples_noprior1.push_back(extended_sample_type(0,0,0,1,0));
            samples_noprior1.push_back(extended_sample_type(0,1,1,1,0));

            samples_noprior1.push_back(extended_sample_type(1,0,0,1,0));
            samples_noprior1.push_back(extended_sample_type(1,1,2,1,0));

            samples_noprior1.push_back(extended_sample_type(2,0,1,1,0));
            samples_noprior1.push_back(extended_sample_type(2,1,3,1,0));

            samples_noprior1.push_back(extended_sample_type(3,0,3,1,0));
            samples_noprior1.push_back(extended_sample_type(3,1,3,1,1));

            //

            std::vector<extended_sample_type> samples_noprior2;
            samples_noprior2.push_back(extended_sample_type(0,0,0,1,0));
            samples_noprior2.push_back(extended_sample_type(0,1,1,1,0));

            samples_noprior2.push_back(extended_sample_type(1,0,0,1,0));
            samples_noprior2.push_back(extended_sample_type(1,1,2,0,1));

            samples_noprior2.push_back(extended_sample_type(2,0,1,1,0));
            samples_noprior2.push_back(extended_sample_type(2,1,3,0,0));

            samples_noprior2.push_back(extended_sample_type(3,0,2,0,0));
            samples_noprior2.push_back(extended_sample_type(3,1,3,0,0));

            //

            dlog << LINFO << "lspi: \n";
            test_lspi_prior1();
            test_lspi_prior2();
            test_lspi_noprior1(samples_noprior1);
            test_lspi_noprior2(samples_noprior2);

            std::vector<std::vector<extended_sample_type> > trials1(100, samples_noprior1);
            std::vector<std::vector<extended_sample_type> > trials2(100, samples_noprior2);

            dlog << LINFO << "qlearning: \n";
            test_ql_sarsa_noprior1<qlearning>(trials1, true);
            test_ql_sarsa_noprior2<qlearning>(trials2, true);
            test_ql_sarsa_noprior1<qlearning>(trials1, false);
            test_ql_sarsa_noprior2<qlearning>(trials2, false);


            // Changing manually the next_action values for this toy case
            int change_to_0_i_1[] = {0,0,0,0,0,0,0,1,1,1,1,1,2,2,2,3};
            int change_to_0_j_1[] = {0,1,2,3,4,5,6,0,1,2,3,4,0,1,2,0};
            for(unsigned int i = 0; i < sizeof(change_to_0_i_1)/sizeof(int); i++)
                trials1[change_to_0_i_1[i]][change_to_0_j_1[i]].next_action = 0;


            int change_to_0_i_2[] = {0,0,0,1};
            int change_to_0_j_2[] = {0,1,2,0};
            for(unsigned int i = 0; i < sizeof(change_to_0_i_2)/sizeof(int); i++)
                trials2[change_to_0_i_2[i]][change_to_0_j_2[i]].next_action = 0;

            dlog << LINFO << "sarsa: \n";
            test_ql_sarsa_noprior1<sarsa>(trials1, true);
            test_ql_sarsa_noprior2<sarsa>(trials2, true);
            test_ql_sarsa_noprior1<sarsa>(trials1, false);
            test_ql_sarsa_noprior2<sarsa>(trials2, false);

        }
    };

    rl_tester a;
}

