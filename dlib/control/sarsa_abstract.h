// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SARSA_ABSTRACT_Hh_
#ifdef DLIB_SARSA_ABSTRACT_Hh_

#include "policy_abstract.h"
#include "model_abstract.h"
#include <random>

namespace dlib
{
    template <
        typename model_type
        >
    class sarsa
    {
        /*!
            REQUIREMENTS ON model_type
                model_type should implement the example_online_model interface defined in
                the approximate_linear_models_abstract.h file.

            WHAT THIS OBJECT REPRESENTS
                This object is an implementation of the well-known reinforcement learning
                algorithm SARSA. It takes an online model and tries to learn the best
                possible policy for the model's environment by interacting with it.

                Supposing we are in state s and action a and we are going to a new state s'
                and taking the action a' in s', then the learning function has the form:
                    Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * Q(s', a'))
                where lr is the learning_rate and disc is the discount factor.
                That formula means that it takes a convex combination of the current qvalue,
                that is, the current expected reward from there, and the new expected qvalue.
        !*/

    public:
        sarsa(
        );
        /*!
            ensures
                - #get_learning_rate() == 0.2
                - #get_discount() == 0.8
                - #get_iterations() == 100
                - #get_epsilon() == 0.1
                - #is_verbose() == false
        !*/

        explicit sarsa(
            double learning_rate,
            double discount,
            unsigned int iterations,
            double epsilon,
            bool verbose
        );
        /*!
          requires
            - learning_rate >= 0 and learning_rate <= 1
            - discount >= 0 and discount <= 1
            - epsilon >= 0 and epsilon <= 1
          ensures
            - #get_learning_rate() == learning_rate
            - #get_discount() == discount
            - #get_iterations() == iterations
            - #get_epsilon() == epsilon
            - #is_verbose() == verbose
        !*/

        double get_learning_rate(
        ) const;
        /*!
            ensures
                - returns the learning rate applied to the learning function.
        !*/

        void set_learning_rate(
            double learning_rate
        );
        /*!
            requires
                - learning_rate >= 0 and learning_rate <= 1.
            ensures
                - #get_learning_rate() == learning_rate
        !*/

        double get_discount(
        ) const;
        /*!
            ensures
                - returns the discount applied to the learning function.
        !*/

        void set_discount(
            double discount
        );
        /*!
            requires
                - discount >= 0 and discount <= 1.
            ensures
                - #get_discount() == discount
        !*/

        unsigned int get_iterations(
        ) const;
        /*!
            ensures
                - returns the maximum number of iterations that sarsa will
                  perform during the training.
        !*/

        void set_iterations(
            unsigned int iterations
        );
        /*!
            ensures
                - #get_iterations() == iterations
        !*/

        double get_epsilon(
        ) const;
        /*!
            ensures
                - returns the probability of taking a random step while training.
        !*/

        void set_epsilon(
            double epsilon
        );
        /*!
            requires
                - epsilon >= 0 and epsilon <= 1.
            ensures
                - #get_epsilon() == epsilon
        !*/

        bool is_verbose(
        ) const;
        /*!
            ensures
                - returns if the class is verbose or not.
        !*/

        void be_verbose(
        );
        /*!
            ensures
                - #is_verbose() == true
        !*/

        void be_quiet(
        );
        /*!
            ensures
                - #is_verbose() == false
        !*/

        template <
            typename prng_engine = std::default_random_engine
            >
        policy_type train(
            const policy<model_type>& policy = policy<model_type>(),
            const prng_engine& gen = prng_engine()
        ) const;
        /*!
            requires
                - prng_engine is a pseudo-random number generator class like the ones
                  defined in std::random. By default it is the standard one.
            ensures
                - returns the policy obtained by applying to the given policy the learning
                  function several times according to the parameters previously fed
                  into this object.
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SARSA_ABSTRACT_Hh_
