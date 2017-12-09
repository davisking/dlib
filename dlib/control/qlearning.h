// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_QLEARNING_Hh_
#define DLIB_QLEARNING_Hh_

#include "policy.h"

namespace dlib
{
    template <
        typename model_type
        >
    class qlearning
    {
    public:
        explicit qlearning(
            double lr = 0.2,
            double disc = 0.8,
            unsigned int miters = 100u,
            double eps = 0.1,
            bool v = false
        ) : max_iterations(miters), verbose(v) {
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

        unsigned int get_max_iterations(
        ) const { return max_iterations; }

        void set_max_iterations(
            unsigned int iterations
        ) { max_iterations = iterations; }

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

        greedy_policy<model_type> train(
            const matrix<double,0,1> &weights
        ) const
        {
            typedef typename model_type::reward_type reward_type;

            epsilon_policy<model_type> eps_pol(epsilon, weights);
            auto& w = eps_pol.get_weights();

            DLIB_ASSERT(weights.size() == model.states_size(),
                "\t qlearning::train(weights)"
                "\n\t invalid inputs were given to this function"
                "\n\t weights.size: " << weights.size() <<
                "\n\t features size: " << model.states_size()
            );

            reward_type total_reward = static_cast<reward_type>(0);
            for(auto iter = 0u; iter < max_iterations; ++iter){
                auto state = model.initial_state();

                reward_type reward = static_cast<reward_type>(0);
                while(!model.is_final(state)){
                    auto action = eps_pol(state);
                    auto next_state = model.step(state, action);
                    auto next_reward = model.reward(state, action, next_state);

                    const auto feats = model.get_features(state, action);
                    const auto feats_next_best = model.get_features(next_state, model.find_best_action(next_state, w));

                    auto prev = w;

                    double correction = reward + discount * dot(w, feats_next_best) - dot(w, feats);
                    //std::cout << "correction " << correction << "\n";
                    w += learning_rate * correction * feats;

                    /*for(auto i = 0; i < model.states_size(); i++)
                        std::cout << w(i) << " ";
                    std::cout << std::endl;

                    for(auto i = 0; i < model.states_size(); i++)
                        std::cout << feats(i) << " ";
                    std::cout << std::endl;


                    if(verbose && sum(abs(w-prev)) != 0){
                        std::cout << "updated:\n";
                        for(auto i = 0; i < model.states_size(); i++){
                            if(prev(i) != w(i))
                                std::cout << "(" << i/5 << "," << i%5 << ") from " << prev(i) << " to " << w(i) << "\n";
                        }
                    }
                    */

                    state = next_state;
                    reward += next_reward;
                }

                total_reward += reward;
                if(verbose)
                    std::cout << "iteration: " << iter << "\t reward: " << reward
                              << "\t mean: " << total_reward/static_cast<int>(iter+1) << std::endl;
            }

            return greedy_policy<model_type>(w);
        }

        greedy_policy<model_type> train(
        ) const
        {
            matrix<double, 0, 1> weights;
            weights = 0;
            return train(weights);
        }

    private:
        double learning_rate;
        double discount;
        unsigned int max_iterations;
        double epsilon;
        bool verbose;

        model_type model;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_QLEARNING_Hh_
