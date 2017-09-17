// Copyright (C) 2017  Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_QLEARNING_Hh_
#define DLIB_QLEARNING_Hh_

#include "approximate_linear_models.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
    >
    class qlearning
    {
    public:
        typedef feature_extractor feature_extractor_type;
        typedef typename feature_extractor::state_type state_type;
        typedef typename feature_extractor::action_type action_type;

        explicit qlearning(
            const feature_extractor& fe_
        ) : fe(fe_) {
            init();
        }

        qlearning(
        ) {
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
                        "\t qlearning::qlearning(feature_extractor, max_iterations, learning_rate, discount)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t learning_rate: " << value
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
                        "\t qlearning::qlearning(feature_extractor, max_iterations, learning_rate, discount)"
                        << "\n\t invalid inputs were given to this function"
                        << "\n\t discount: " << value
                        );
            discount = value;
        }

        template <
                typename vector_type
                >
        policy<feature_extractor> train(
            const vector_type& samples
        ) const
        {
            DLIB_ASSERT(samples.size() > 0,
                "\t policy qlearning::train(samples)"
                << "\n\t  invalid inputs were given to this function"
                );
            DLIB_ASSERT(feature_extractor::force_last_weight_to_1 == false,
                "\t policy qlearning::train(samples)"
                << "\n\t invalid template parameter were given to this function:"
                << "\n\t feature_extractor::force_last_weight_to_1 must be false."
                );

            matrix<double,0,1> w(fe.num_features());
            w = 0;
            matrix<double,0,1> f1, f2;

            for(unsigned long iter = 0uL; iter < max_iterations; iter++){
                for(unsigned long i = 0uL; i < samples.size(); i++){
                    fe.get_features(samples[i].state, samples[i].action, f1);
                    fe.get_features(samples[i].next_state,
                                    fe.find_best_action(samples[i].next_state, w),
                                    f2);

                    double correction = samples[i].reward + discount * dot(w, f2) - dot(w, f1);
                    w = w + learning_rate * correction * f1;
                }

                if(verbose)
                    std::cout << "iteration: " << iter << "\tweights: " << trans(w) << std::endl;
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

#endif // DLIB_QLEARNING_Hh_
