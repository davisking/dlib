// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SARSA_Hh_
#define DLIB_SARSA_Hh_

#include "policy.h"
#include <type_traits>
#include <iostream>

namespace dlib
{
    class sarsa
    {
    public:
        explicit sarsa(
            double lr = 0.2,
            double disc = 0.8,
            unsigned int miters = 100u,
            double eps = 0.1,
            bool v = false
        ) : iterations(miters), verbose(v) {
            set_learning_rate(lr);
            set_discount(disc);
            set_epsilon(eps);
        }

        double get_learning_rate(
        ) const { return learning_rate; }

        void set_learning_rate(
            double value
        )
        {
            DLIB_ASSERT(value >= 0. && value <= 1.,
                "\t sarsa::set_learning_rate(value)"
                "\n\t invalid inputs were given to this function"
                "\n\t value: " << value
            );
            learning_rate = value;
        }

        double get_discount(
        ) const { return discount; }

        void set_discount(
            double value
        )
        {
            DLIB_ASSERT(value >= 0. && value <= 1.,
                "\t sarsa::set_discount(value)"
                "\n\t invalid inputs were given to this function"
                "\n\t value: " << value
            );
            discount = value;
        }

        unsigned int get_iterations(
        ) const { return iterations; }

        void set_iterations(
            unsigned int value
        ) { iterations = value; }

        double get_epsilon(
        ) const { return epsilon; }

        void set_epsilon(
            double value
        )
        {
            DLIB_ASSERT(value >= 0. && value <= 1.,
                "\t sarsa::set_epsilon(value)"
                "\n\t invalid inputs were given to this function"
                "\n\t value: " << value
            );
            epsilon = value;
        }

        bool is_verbose(
        ) const { return verbose; }

        void be_verbose(
        ) { verbose = true; }

        void be_quiet(
        ) { verbose = false; }

        template <
                typename policy_type
                >
        policy_type train_policy(
            const policy_type &policy
        ) const
        {
            typedef typename std::decay<decltype(policy.get_model())>::type::reward_type reward_type;

            if(verbose)
                std::cout << "Starting training..." << std::endl;

            const auto &model = policy.get_model();
            epsilon_policy<policy_type> eps_pol(epsilon, policy);
            auto& w = eps_pol.get_weights();

            DLIB_ASSERT(weights.size() == model.states_size(),
                "\t sarsa::train(weights)"
                "\n\t invalid inputs were given to this function"
                "\n\t weights.size: " << weights.size() <<
                "\n\t features size: " << model.states_size()
            );

            reward_type total_reward = static_cast<reward_type>(0);
            for(auto iter = 0u; iter < iterations; ++iter){
                auto state = model.initial_state();
                auto action = eps_pol(state);

                reward_type reward = static_cast<reward_type>(0);
                while(!model.is_final(state)){
                    auto next_state = model.step(state, action);
                    auto next_action = eps_pol(next_state);
                    auto next_reward = model.reward(state, action, next_state);

                    const auto feats = model.get_features(state, action);
                    const auto feats_next = model.get_features(next_state, next_action);

                    double correction = reward + discount * dot(w, feats_next) - dot(w, feats);
                    w += learning_rate * correction * feats;

                    state = next_state;
                    action = next_action;
                    reward += next_reward;
                }

                total_reward += reward;
                if(verbose)
                    std::cout << "iteration: " << iter << "\t reward: " << reward
                              << "\t mean: " << total_reward/static_cast<int>(iter+1) << std::endl;
            }

            if(verbose)
                std::cout << "Training finished." << std::endl;

            return eps_pol.get_policy();
        }

        template <
                typename model_type
                >
        greedy_policy<model_type> train(
            const model_type &model
        ) const { return train_policy(greedy_policy<model_type>(model)); }

    private:
        double learning_rate;
        double discount;
        unsigned int iterations;
        double epsilon;
        bool verbose;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SARSA_Hh_
