// Copyright (C) 2017  Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/control.h>
#include <dlib/serialize.h>
#include <vector>
#include <sstream>
#include <ctime>

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    logger dlog("test.rl");

    template <
            int height,
            int width
            >
    class cliff_model
    {
    public:
        // actions allowed in the model
        enum class actions {up = 0, right, down, left};
        constexpr static int num_actions = 4;

        // some constants we need
        constexpr static double EPS = 1e-16;
        constexpr static int HEIGHT = height;
        constexpr static int WIDTH = width;

        // we define the model's types
        typedef int state_type;
        typedef actions action_type;

        // Constructors
        explicit cliff_model(
            int seed = 0
        ) : gen(seed){}

        cliff_model(const cliff_model<height, width>&) = default;
        cliff_model<height, width>& operator=(const cliff_model<height, width>&) = default;

        // Functions that will use the agent

        unsigned int num_features(
        ) const { return num_actions * height * width; }

        void get_features(
            const state_type &state,
            const action_type &action,
            matrix<double,0,1>& feats
        ) const
        {
            feats = 0;
            feats(num_actions*state + static_cast<int>(action)) = 1; //only this one is 1
        }

        // It's possible that the allowed actions differ among states.
        // In this case all movements are always allowed so we don't need to use state.
        action_type random_action(
            const state_type& state
        ) const
        {
            uniform_int_distribution<int> dist(0,num_actions-1);
            return static_cast<action_type>(dist(gen));
        }

        action_type find_best_action(
            const state_type& state,
            const matrix<double,0,1>& w
        ) const
        {
            auto best = numeric_limits<double>::lowest();
            std::vector<int> best_indexes;

            for(auto i = 0; i < num_actions; i++)
            {
                matrix<double,0,1> feats(num_features());
                get_features(state, static_cast<action_type>(i), feats);
                auto product = dot(w, feats);

                if(product > best){
                    best = product;
                    best_indexes.clear();
                }
                if(abs(product - best) < EPS)
                    best_indexes.push_back(i);
            }

            // returns a random action between the best ones.
            uniform_int_distribution<unsigned long> dist(0, best_indexes.size()-1);
            return static_cast<action_type>(best_indexes[dist(gen)]);
        }

        // This functions gives the rewards, that is, tells the agent how good are its movements
        double reward(
            const state_type &state,
            const action_type &action,
            const state_type &new_state
        ) const
        {
            return !is_final(new_state) ? -1 : is_success(new_state) ? 100 : -100;
        }

        state_type initial_state(
        ) const { return static_cast<state_type>((height-1) * width); }

        // This is an important function, basically it allows the agent to move around the environment
        state_type step(
            const state_type& state,
            const action_type& action
        ) const
        {
            if(out_of_bounds(state, action))
                return state;

            return action == actions::up    ?   state - width   :
                   action == actions::down  ?   state + width   :
                   action == actions::right ?   state + 1       :
                                                state - 1       ;
        }

        // this functions allow the agent to know in which state of the simulation it's in
        bool is_success(
            const state_type &state
        ) const { return state == height*width - 1; }

        bool is_failure(
            const state_type &state
        ) const { return state/width == height-1 && state%width > 0 && state%width < width-1;}

        bool is_final(
            const state_type& state
        ) const { return is_success(state) || is_failure(state); }

        const std::default_random_engine& get_generator(
        ) const { return gen; }

    private:

        bool out_of_bounds(
            const state_type& state,
            const action_type& action
        ) const
        {
            bool result;

            switch(action)
            {
            case actions::up:
                result = state / width == 0;
                break;
            case actions::down:
                result = state / width == height-1;
                break;
            case actions::left:
                result = state % width == 0;
                break;
            case actions::right:
                result = state % width == width-1;
                break;
            }

            return result;
        }

        template < int H, int W>
        friend void serialize(const cliff_model<H, W>& item, std::ostream& out);

        template < int H, int W>
        friend void deserialize(cliff_model<H, W>& item, std::istream& in);

        mutable default_random_engine gen; //mutable because it doesn't changes the model state
    };

    template < int height, int width >
    inline void serialize(const cliff_model<height, width>& item, std::ostream& out)
    {
        int version = 1;
        dlib::serialize(version, out);
        dlib::serialize(item.gen, out);
    }

    template < int height, int width >
    inline void deserialize(cliff_model<height, width>& item, std::istream& in)
    {
        int version = 0;
        dlib::deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing reinforcement learning test model object.");

        item = cliff_model<height, width>();
        dlib::deserialize(item.gen, in);
    }

    template <
        int height,
        int width,
        template<typename> class algorithm_t
        >
    void test()
    {
        constexpr static int seed = 7;

        typedef cliff_model<height, width> model_t;
        const int max_steps = 100;

        print_spinner();
        algorithm_t<model_t> algorithm;
        model_t model(seed);
        auto my_policy = algorithm.train(policy<model_t>(model), std::default_random_engine(seed));

        auto s = model.initial_state();
        double r = 0.;
        int i;

        for(i = 0; i < max_steps && !model.is_final(s); i++){
            auto a = my_policy(s);
            auto new_s = model.step(s, a);
            r += model.reward(s,a,new_s);
            s = new_s;
        }

        dlog << LINFO << "height, width:   " << height << "," << width;
        dlog << LINFO << "steps:   " << i;
        dlog << LINFO << "state:   (" << s/width << "," << s%width << ")";
        dlog << LINFO << "success: " << (model.is_success(s) ? "true" : "false");
        dlog << LINFO << "failure: " << (model.is_failure(s) ? "true" : "false");
        dlog << LINFO << "reward:  " << r;
        DLIB_TEST(i != max_steps);
        DLIB_TEST(model.is_success(s));
        DLIB_TEST(r > 0);
    }

    void policy_serialization_test(){
        cliff_model<3, 5> model(8);
        policy<decltype(model)> gp(model), gres;

        for(uint i = 0u; i < gp.get_weights().size(); i++)
            gp.get_weights()(i) = i;

        ostringstream sout;
        serialize(gp, sout);
        istringstream sin(sout.str());

        deserialize(gres, sin);
        dlog << LINFO << "policy serializing:  " <<
                (gp.get_weights() == gres.get_weights() && gp.get_model().get_generator() == gres.get_model().get_generator());
        DLIB_TEST(gp.get_weights() == gres.get_weights() && gp.get_model().get_generator() == gres.get_model().get_generator());
    }

    void epsilon_policy_serialization_test(){
        cliff_model<3, 5> model(11);
        policy<decltype(model)> gp(model);

        for(uint i = 0u; i < gp.get_weights().size(); i++)
            gp.get_weights()(i) = i;

        epsilon_policy<decltype(gp)> ep(0.3, gp);
        auto eres = ep; // epsilon_policy is not default constructible

        auto state = ep.get_model().initial_state();
        for(uint i = 0u; i < 3; i++)
            state = ep.get_model().step(state, ep(state));

        ostringstream sout;
        serialize(ep, sout);
        istringstream sin(sout.str());

        auto cstate = state;
        for(uint i = 0; i < 5; i++)
            state = ep.get_model().step(state, ep(state));

        deserialize(eres, sin);
        for(uint i = 0; i < 5; i++)
            cstate = eres.get_model().step(cstate, eres(cstate));

        dlog << LINFO << "epsilon policy serializing:  " <<
                (ep.get_weights() == eres.get_weights() && ep.get_generator() == eres.get_generator() &&
                 ep.get_model().get_generator() == eres.get_model().get_generator() ? "True" : "False");
        dlog << LINFO << "same state stepping after serializing:  " << (state == cstate ? "True" : "False");
        DLIB_TEST(state == cstate);
        DLIB_TEST(ep.get_weights() == eres.get_weights() && ep.get_generator() == eres.get_generator() &&
                  ep.get_model().get_generator() == eres.get_model().get_generator());
    }

    class rl_tester : public tester
    {
    public:
        rl_tester (
        ) :
            tester (
                "test_rl",       // the command line argument name for this test
                "Run tests on the qlearning and sarsa objects.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }

        void perform_test (
        )
        {
            test<4,5,qlearning>();
            test<5,5,qlearning>();
            test<4,7,qlearning>();
            test<5,10,qlearning>();

            test<4,5,sarsa>();
            test<5,5,sarsa>();
            test<4,7,sarsa>();
            test<5,10,sarsa>();

            policy_serialization_test();
            epsilon_policy_serialization_test();
        }
    };

    rl_tester a;
}

