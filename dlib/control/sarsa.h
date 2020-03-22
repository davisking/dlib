// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SARSA_Hh_
#define DLIB_SARSA_Hh_

#include "approximate_linear_models.h"
#include <type_traits>
#include <iostream>
#include <random>

namespace dlib
{
    template<
        typename model_type
        >
    class sarsa
    {
    public:
        explicit sarsa(
            double lr = 0.2,
            double disc = 0.8,
            unsigned int miters = 100u,
            double eps = 0.1,
            bool v = false
        ) : iters(miters), verbose(v) {
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
        ) const { return iters; }

        void set_iterations(
            unsigned int value
        ) { iters = value; }

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
            typename prng_engine = std::default_random_engine
            >
        policy<model_type> train(
            const policy<model_type>& training_policy = policy<model_type>(),
            const prng_engine& gen = prng_engine()
        ) const
        {
            if(verbose)
                std::cout << "Starting training..." << std::endl;

            const auto& model = training_policy.get_model();
            epsilon_policy<policy<model_type>, prng_engine> eps_pol(epsilon, training_policy, gen);
            auto& w = eps_pol.get_weights();

            DLIB_ASSERT(weights.size() == model.states_size(),
                "\t sarsa::train(policy, gen)"
                "\n\t invalid inputs were given to this function"
                "\n\t policy's weights.size: " << weights.size() <<
                "\n\t num of features: " << model.num_features()
            );

            matrix<double,0,1> feats(model.num_features()), feats_next(model.num_features());

            double total_reward = 0.;
            for(auto iter = 0u; iter < iters; ++iter)
            {
                auto state = model.initial_state();
                auto action = eps_pol(state);
                auto steps = 0u;
                double iteration_reward = 0.;

                while(!model.is_final(state))
                {
                    auto next_state = model.step(state, action);
                    auto next_action = eps_pol(next_state);
                    auto reward = model.reward(state, action, next_state);

                    model.get_features(state, action, feats);
                    model.get_features(next_state, next_action, feats_next);

                    double correction = reward + discount * dot(w, feats_next) - dot(w, feats);
                    w += learning_rate * correction * feats;

                    state = next_state;
                    action = next_action;
                    iteration_reward += reward;
                }

                total_reward += iteration_reward;
                if(verbose)
                    std::cout << "iteration: " << iter
                              << "\t reward: " << iteration_reward
                              << "\t mean: " << total_reward/static_cast<int>(iter+1)
                              << "\t steps: " << steps
                              << std::endl;
            }

            if(verbose)
                std::cout << "Training finished." << std::endl;

            return eps_pol.get_policy();
        }

    private:
        double learning_rate;
        double discount;
        unsigned int iters;
        double epsilon;
        bool verbose;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SARSA_Hh_
