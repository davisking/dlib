// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SARSA_ABSTRACT_Hh_
#ifdef DLIB_SARSA_ABSTRACT_Hh_

#include "approximate_linear_models_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
    >
    class sarsa
    {
        /*!
            REQUIREMENTS ON feature_extractor
                feature_extractor should implement the example_feature_extractor interface
                defined at the top of dlib/control/approximate_linear_models_abstract.h

            WHAT THIS OBJECT REPRESENTS
                This object is an implementation of the SARSA algorithm. It is an on-policy
                algorithm and this implementation is best-suited for control problems.

                It takes a bunch of training data in the form of process_samples and outputs
                a policy that hopefully performs well when run on the process that generated
                those samples.
        !*/

    public:
        using state_type = typename feature_extractor::state_type;
        using action_type = typename feature_extractor::action_type;

        explicit sarsa(
            const feature_extractor& fe_
        );
        /*!
            ensures
                - #get_feature_extractor() == fe_
                - get_learning_rate() >= 0 && get_learning_rate() <= 1
                - get_discount() >= 0 && get_discount() <= 1
        !*/

        sarsa(
        );
        /*!
            ensures
                - #get_feature_extractor() == feature_extractor()
                - get_learning_rate() >= 0 && get_learning_rate() <= 1
                - get_discount() >= 0 && get_discount() <= 1
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that a
                  user can observe the progress of the algorithm.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - this object will not print anything to standard out
        !*/

        const feature_extractor& get_feature_extractor(
        ) const;
        /*!
            ensures
                - returns the feature extractor used by this object.
        !*/

        unsigned long get_max_iterations(
        ) const;
        /*!
            ensures
                - returns the maximum number of iterations of this object's training.
        !*/

        void set_max_iterations(
            unsigned long value
        );
        /*!
            requires
                - value >= 0
            ensures
                - #get_max_iterations() == value
        !*/

        double get_learning_rate(
        ) const;
        /*!
            ensures
                - returns the learning rate used by this object.
        !*/

        void set_learning_rate(
            double value
        );
        /*!
            requires
                - value >= 0 && value <= 1
            ensures
                - get_learning_rate() == value
        !*/

        double get_discount(
        ) const;
        /*!
            ensures
                - returns the discount value used by this object.
        !*/

        void set_discount(
            double value
        );
        /*!
            requires
                - value >= 0 && value <= 1
            ensures
                - get_discount() == value
        !*/

        template <
                typename vector_type
                >
        policy<feature_extractor> train(
            const vector_type& samples
        ) const;
        /*!
            requires
                - samples.size() > 1
                - samples is something with an interface that looks like
                  std::vector<process_sample<feature_extractor>>.  That is, it should
                  be some kind of array of process_sample objects.
                - feature_extractor::force_last_weight_to_1 == false // TODO
            ensures
                - Trains a policy based on the given data and returns the results. The
                  idea is to find a policy that will obtain the largest possible reward
                  when run on the process that generated the samples. In particular,
                  if the returned policy is P then:
                    - P(S) == the best action to take when in state S.
            notice
                - Note that SARSA needs the actual next action taken so it doesn't process
                the last sample. If you are trying to reach a goal the last sample should
                be, at least, the moment when your state is the goal state and the reward
                for doing so the previous step.
        !*/
    };
}

#endif // DLIB_SARSA_ABSTRACT_Hh_
