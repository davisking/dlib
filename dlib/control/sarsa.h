// Copyright (C) 2017  Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SARSA_Hh_
#define DLIB_SARSA_Hh_

#include "approximate_linear_models.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
    >
    class sarsa
    {
    public:
        typedef feature_extractor feature_extractor_type;
        typedef typename feature_extractor::state_type state_type;
        typedef typename feature_extractor::action_type action_type;

        explicit sarsa(
            const feature_extractor& fe_
        ) : fe(fe_)
        {
            init();
        }

        sarsa(
        )
        {
            init();
        }

        void be_verbose (
        ) { verbose = true; }

        void be_quiet (
        ) { verbose = false; }

        const feature_extractor& get_feature_extractor(
        ) const { return fe; }

        unsigned long get_max_iterations(
        ) const { return max_iterations; }

        void set_max_iterations(
            unsigned long value
        ) { max_iterations = value; }

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

        template <
                typename vector_type
                >
        policy<feature_extractor> train(
            const vector_type& trials
        ) const
        {
            DLIB_ASSERT(trials.size() > 1,
                "\t policy sarsa::train(trials)"
                "\n\t  invalid inputs were given to this function"
                );
            DLIB_ASSERT(feature_extractor::force_last_weight_to_1 == false,
                "\t policy sarsa::train(trials)"
                "\n\t invalid template parameter were given to this function:"
                "\n\t feature_extractor::force_last_weight_to_1 must be false."
                );

            unsigned long trials_size = trials.size() < max_iterations ? trials.size() : max_iterations;
            matrix<double,0,1> w(fe.num_features()), f1, f2;
            w = 0;

            for(unsigned long i = 0uL; i < trials_size; i++){
                for(unsigned long j = 0uL; j < trials[i].size(); j++){
                    fe.get_features(trials[i][j].state, trials[i][j].action, f1);
                    fe.get_features(trials[i][j].next_state, trials[i][j].next_action, f2);

                    double correction = trials[i][j].reward + discount * dot(w, f2) - dot(w, f1);
                    w = w + learning_rate * correction * f1;
                }

                if(verbose)
                    std::cout << "trial: " << i << "\tweights: " << trans(w) << std::endl;
            }

            return policy<feature_extractor>(w,fe);
        }

        template <
                typename InputIterator,
                typename EndIterator
                >
        policy<feature_extractor> train(
            InputIterator iterator,
            const EndIterator& end_iterator
        ) const
        {
            typedef typename InputIterator::value_type value_type;
            typedef typename value_type::iterator iterator_type;

            DLIB_ASSERT(feature_extractor::force_last_weight_to_1 == false,
                "\t policy sarsa::train(iterator, end_iterator)"
                "\n\t invalid template parameter were given to this function:"
                "\n\t feature_extractor::force_last_weight_to_1 must be false."
                );

            matrix<double,0,1> w(fe.num_features()), f1, f2;
            w = 0;

            unsigned long count = 0uL;
            for( ; !(end_iterator == iterator); ++iterator){
                value_type trial = *iterator;

                for(iterator_type it = trial.begin(); trial.end() != it; ++it){
                    fe.get_features(it->state, it->action, f1);
                    fe.get_features(it->next_state, it->next_action, f2);

                    double correction = it->reward + discount * dot(w, f2) - dot(w, f1);
                    w = w + learning_rate * correction * f1;
                }

                if(verbose)
                    std::cout << "trial: " << count << "\tweights: " << trans(w) << std::endl;
                count++;
            }

            return policy<feature_extractor>(w,fe);
        }

    private:
        void init()
        {
            max_iterations = 100uL;
            learning_rate = 0.3;
            discount = 0.8;

            verbose = false;
        }

        feature_extractor fe;
        unsigned long max_iterations;
        double learning_rate;
        double discount;

        bool verbose;
    };
}

#endif // DLIB_SARSA_Hh_
