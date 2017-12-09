// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_QLEARNING_ABSTRACT_Hh_
#ifdef DLIB_QLEARNING_ABSTRACT_Hh_

#include "policy_abstract.h"
#include "model_abstract.h"

namespace dlib
{
    template <
        typename model_type
        >
    class qlearning
    {
        /*!
            REQUIREMENTS ON model_type
                model_type is an implementation of the model interface declared in
                  model_abstract.h.

            WHAT THIS OBJECT REPRESENTS
                This objects is an implementation of the well-known reinforcement learning
                algorithm Q-learning. This algorithms takes a bunch of process_samples
                as input and outputs a policy that have learnt from that in order to take
                the better results.

                Supposing we are in state s and action a the learning function has the form:
                    Q(s, a) = (1 - lr) * Q(s,a) + lr * (reward + disc * max_a' Q(s, a'))
                where lr is the learning_rate and disc the discount.
                That formula means that it takes a convex combination of the current qvalue
                and the expected qvalue.
        !*/

    public:
        qlearning(
        );
        /*!
            ensures
                - #get_learning_rate() == 0.2
                - #get_discount() == 0.8
                - #get_max_iterations() == 100
                - #get_epsilon() == 0.1
                - #is not verbose
        !*/

        explicit qlearning(
            double learning_rate,
            double discount,
            unsigned int max_iterations,
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
            - #get_max_iterations() == max_iterations
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

        unsigned int get_max_iterations(
        ) const;
        /*!
            ensures
                - returns the maximum number of iterations that qlearning will
                  perform during the training.
        !*/

        void set_max_iterations(
            unsigned int iterations
        );
        /*!
            ensures
                - #get_max_iterations() == iterations
        !*/

        double get_epsilon(
        ) const;
        /*!
            ensures
                - returns the probability of doing a non-optimal step while training.
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

        greedy_policy<model_type> train(
            const matrix<double,0,1> &weights
        ) const;
        /*!
            requires
                - weights.size() == model_type.states_size()
            ensures
                - returns a greedy_policy resulting of doing max_iterations iterations
                  over the model while applying the learning function to the weights
                  matrix of the policy.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_QLEARNING_ABSTRACT_Hh_
