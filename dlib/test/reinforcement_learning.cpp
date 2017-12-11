// Copyright (C) 2017  Adrián Javaloy (adrian.javaloy@gmail.com)
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

    template <
            int height,
            int width,
            template<typename,typename> class feature_extractor_type
            >
    class cliff_model
    {
    public:
        // constants and actions allowed
        enum class actions {up = 0, right, down, left};
        constexpr static double EPS = 1e-16;
        constexpr static int HEIGHT = height;
        constexpr static int WIDTH = width;

        // model types
        typedef int state_type;
        typedef actions action_type;
        typedef int reward_type;

        typedef feature_extractor_type<state_type, action_type> feature_extractor;

        explicit cliff_model(
            int seed = 0
        ) : fe(height, width, 4), gen(seed) {}

        action_type random_action(
            const state_type& state // since all movements are always allowed we don't use state
        ) const
        {
            std::uniform_int_distribution<int> dist(0,3);
            return static_cast<action_type>(dist(gen));
        }

        action_type find_best_action(
            const state_type& state,
            const matrix<double,0,1>& w
        ) const
        {
            // it looks for the best actions in state according to w
            auto best = std::numeric_limits<double>::lowest();
            auto best_indexes = std::vector<int>();

            for(auto i = 0; i < 4; i++){
                auto feats = get_features(state, static_cast<action_type>(i));
                auto product = dot(w, feats);

                if(product > best){
                    best = product;
                    best_indexes.clear();
                }
                if(std::abs(product - best) < EPS)
                    best_indexes.push_back(i);
            }

            // returns a random action between the best ones.
            std::uniform_int_distribution<unsigned long> dist(0, best_indexes.size()-1);
            return static_cast<action_type>(best_indexes[dist(gen)]);
        }

        const feature_extractor& get_feature_extractor(
        ) const { return fe; }

        auto states_size(
        ) const -> decltype(get_feature_extractor().num_features())
        {
            return get_feature_extractor().num_features();
        }

        auto get_features(
            const state_type &state,
            const action_type &action
        ) const -> decltype(get_feature_extractor().get_features(state, action))
        { return get_feature_extractor().get_features(state, action); }

        reward_type reward(
            const state_type &state,
            const action_type &action,
            const state_type &new_state
        ) const
        {
            return !is_final(new_state) ? -1 : is_success(new_state) ? 100 : -100;
        }

        state_type initial_state(
        ) const { return static_cast<state_type>((height-1) * width); }

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

        bool is_success(
            const state_type &state
        ) const { return state == height*width - 1; }

        bool is_failure(
            const state_type &state
        ) const { return state/width == height-1 && state%width > 0 && state%width < width-1;}

        bool is_final(
            const state_type& state
        ) const { return is_success(state) || is_failure(state); }

    private:
        bool out_of_bounds(
            const state_type& state,
            const action_type& action
        ) const
        {
            bool result;

            switch(action){
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

        feature_extractor fe;
        mutable std::default_random_engine gen; //mutable because it doesn't changes the model state
    };

    template <
            typename state_type,
            typename action_type
            >
    class feature_extractor
    {
    public:
        feature_extractor(
            int h,
            int w,
            int na
        ) : height(h), width(w), num_actions(na) {}

        inline long num_features(
        ) const { return num_actions * height * width; }

        matrix<double,0,1> get_features(
            const state_type &state,
            const action_type &action
        ) const
        {
            matrix<double,0,1> feats(num_features());
            feats = 0;
            //for(auto i = 0u; i < num_actions; i++)
            //    feats(num_actions * state + i) = 1;
            feats(num_actions*state + static_cast<int>(action)) = 1;

            return feats;
        }

    private:
        int height, width, num_actions;
    };

    template <
        int height,
        int width,
        typename algorithm_t
        >
    void test()
    {
        constexpr static int seed = 7;

        typedef cliff_model<height, width, feature_extractor> model_t;
        const int max_steps = 100;

        print_spinner();
        algorithm_t algorithm;
        model_t model(seed);
        auto policy = algorithm.train(model, std::default_random_engine(seed));

        auto s = model.initial_state();
        auto r = static_cast<typename model_t::reward_type>(0);
        int i;

        for(i = 0; i < max_steps && !model.is_final(s); i++){
            auto a = policy(s);
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
        }
    };

    rl_tester a;
}

