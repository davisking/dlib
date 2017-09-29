// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_QLEARNING_ABSTRACT_Hh_
#ifdef DLIB_QLEARNING_ABSTRACT_Hh_

#include "approximate_linear_models_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
    >
    class qlearning
    {
        /*!
            REQUIREMENTS ON feature_extractor
                feature_extractor should implement the example_feature_extractor interface
                defined at the top of dlib/control/approximate_linear_models_abstract.h

            WHAT THIS OBJECT REPRESENTS
                This object is an implementation of the well-known reinforcement learning
                algorithm Q-learning. It is an off-policy algorithm and this implementation
                is best-suited for control problems.

                It takes a bunch of training data in the form of process_samples and outputs
                a policy that hopefully performs well when run on the process that generated
                those samples.
        !*/

    public:
        typedef feature_extractor feature_extractor_type;
        typedef typename feature_extractor::state_type state_type;
        typedef typename feature_extractor::action_type action_type;

        explicit qlearning(
            const feature_extractor& fe_
        );
        /*!
            ensures
                - #get_feature_extractor() == fe_
                - get_learning_rate() >= 0 && get_learning_rate() <= 1
                - get_discount() >= 0 && get_discount() <= 1
        !*/

        qlearning(
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
                - samples.size() > 0
                - samples is something with an interface that looks like
                  std::vector<std::vector<process_sample<feature_extractor>>>.  That is, it should
                  be some kind of array of arrays of process_sample objects. The outer vectors are
                  the trials and are ordered time-wise whereas the samples in each inner vector
                  are the sequential steps from the goal until the end (if reached).
                - feature_extractor::force_last_weight_to_1 == false
            ensures
                - Trains a policy based on the given data and returns the results. The
                  idea is to find a policy that will obtain the largest possible reward
                  when run on the process that generated the samples. In particular,
                  if the returned policy is P then:
                    - P(S) == the best action to take when in state S.
        !*/

        template <
                typename InputIterator,
                typename EndIterator
                >
        policy<feature_extractor> train(
            InputIterator iterator, const EndIterator& end_iterator
        ) const;
        /*!
            requires
                - iterator is an input iterator (according to the definition from the stl)
                and it makes reference to something like std::vector<process_sample<feature_extractor>>>,
                i.e., something from which we can take another input iterator and read the process samples.
                - end_iterator is a class that can be right-hand compared with iterator and that comparison
                returns true whenever we've reached the end of the trials.
                - feature_extractor::force_last_weight_to_1 == false
            ensures
                - Trains a policy based on the given data and returns the results. The
                  idea is to find a policy that will obtain the largest possible reward
                  when run on the process that generated the samples. In particular,
                  if the returned policy is P then:
                    - P(S) == the best action to take when in state S.
        !*/
    };
}

#endif // DLIB_QLEARNING_ABSTRACT_Hh_
