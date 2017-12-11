// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_QLEARNING_Hh_
#define DLIB_QLEARNING_Hh_

#include "policy.h"
#include <iostream>
#include <type_traits>
#include <random>

namespace dlib
{
    class qlearning
    {
    public:
        explicit qlearning(
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
                "\t qlearning::set_learning_rate(value)"
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
                "\t qlearning::set_discount(value)"
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
                "\t qlearning::set_epsilon(value)"
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
            typename policy_type,
            typename prng_engine = std::default_random_engine
            >
        policy_type train_policy(
            const policy_type &policy,
            const prng_engine &gen = prng_engine()
        ) const
        {
            typedef typename std::decay<decltype(policy.get_model())>::type::reward_type reward_type;

            if(verbose)
                std::cout << "Starting training..." << std::endl;

            const auto &model = policy.get_model();
            epsilon_policy<policy_type, prng_engine> eps_pol(epsilon, policy, gen);
            auto& w = eps_pol.get_weights();

            DLIB_ASSERT(weights.size() == model.states_size(),
                "\t qlearning::train(weights)"
                "\n\t invalid inputs were given to this function"
                "\n\t weights.size: " << weights.size() <<
                "\n\t features size: " << model.states_size()
            );

            reward_type total_reward = static_cast<reward_type>(0);
            for(auto iter = 0u; iter < iterations; ++iter){
                auto state = model.initial_state();

                auto steps = 0u;
                reward_type iteration_reward = static_cast<reward_type>(0);
                while(!model.is_final(state)){
                    auto action = eps_pol(state);
                    auto next_state = model.step(state, action);
                    auto reward = model.reward(state, action, next_state);

                    const auto feats = model.get_features(state, action);
                    const auto feats_next_best = model.get_features(next_state, model.find_best_action(next_state, w));

                    double correction = reward + discount * dot(w, feats_next_best) - dot(w, feats);
                    w += learning_rate * correction * feats;

                    state = next_state;
                    iteration_reward += reward;
                    steps++;
                }

                total_reward += iteration_reward;
                if(verbose)
                    std::cout << "iteration: " << iter << "\t reward: " << iteration_reward
                              << "\t mean: " << total_reward/static_cast<int>(iter+1)
                              << "\t steps: " << steps
                              << std::endl;
            }

            if(verbose)
                std::cout << "Training finished." << std::endl;

            return eps_pol.get_policy();
        }

        template <
                typename model_type,
                typename prng_engine = std::default_random_engine
                >
        greedy_policy<model_type> train(
            const model_type &model,
            const prng_engine &gen = prng_engine()
        ) const { return train_policy(greedy_policy<model_type>(model), gen); }

    private:
        double learning_rate;
        double discount;
        unsigned int iterations;
        double epsilon;
        bool verbose;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_QLEARNING_Hh_
