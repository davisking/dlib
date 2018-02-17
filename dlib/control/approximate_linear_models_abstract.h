// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_
#ifdef DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_

#include <../matrix_abstract.h>
#include <random>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct example_offline_model {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object defines the inferface that any model has to implement if it
                is to be used in an offline fashion along with some method like the lspi
                method defined in the file lspi_abstract.h. Being offline only means that
                it already holds the data and will not interact with the environment to get
                them.

                In particular, this object models a Q(state, action) function where
                    Q(state, action) == dot(w, PSI(state, action))
                where PSI(state, action) is a feature vector and w is a parameter vector.

                Therefore, an offline model defines how the PSI(x,y) feature vector is
                calculated. It also defines the types used to represent the state and
                action objects.

            THREAD SAFETY
                Instances of this object are required to be threadsafe, that is, it should
                be safe for multiple threads to make concurrent calls to the member
                functions of this object.
        !*/

        // The states and actions can be any type as long as you provide typedefs for them.
        typedef U state_type;
        typedef V action_type;
        // We can also say that the last element in the weights vector w must be 1. This
        // can be useful for including a prior into your model.
        const static bool force_last_weight_to_1 = false;

        example_offline_model(
        );
        /*!
            ensures
                - this object is properly initialized.
        !*/

        unsigned long num_features(
        ) const;
        /*!
            ensures
                - returns the dimensionality of the PSI() feature vector.
        !*/

        action_type find_best_action (
            const state_type& state,
            const matrix<double,0,1>& w
        ) const;
        /*!
            ensures
                - returns the action A that maximizes Q(state, A) = dot(w,PSI(state,A)).
                  That is, this function finds the best action to take in the given state
                  when our model is parameterized by the given weight vector w.
        !*/

        void get_features(
            const state_type& state,
            const action_type& action,
            matrix<double,0,1>& feats
        ) const;
        /*!
            ensures
                - #feats.size() == num_features()
                - #feats == PSI(state, action)
        */!
    };

// ----------------------------------------------------------------------------------------

    struct example_online_model
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object defines the inferface that any model has to implement if it
                is to be used in an online fashion along with some method like the qlearning
                method defined in the file qlearning_abstract.h.

                Being online means that the model doesn't hold prior data but it interacts
                with the environment and performing actions from some given state turning
                that state into a new one as well as getting some reward for doing so.

                In particular, this object models a Q(state, action) function where
                    Q(state, action) == dot(w, PSI(state, action))
                where PSI(state, action) is a feature vector and w is a parameter vector.

                Therefore, an online model defines how the PSI(x,y) feature vector is
                calculated, the types used to represent the state, action and reward
                objects as well as how to interact with the environment.

            THREAD SAFETY
                Instances of this object are required to be threadsafe, that is, it should
                be safe for multiple threads to make concurrent calls to the member
                functions of this object.
        !*/

        // The states and actions can be any type as long as you provide typedefs for them.
        typedef U state_type;
        typedef V action_type;

        example_online_model(
        );
        /*!
            ensures
                - this object is properly initialized.
        !*/

        unsigned long num_features(
        ) const;
        /*!
            ensures
                - returns the dimensionality of the PSI vector.
        !*/

        action_type find_best_action(
            const state_type& state,
            const matrix<double,0,1>& w
        ) const;
        /*!
            ensures
                - returns the action A that maximizes Q(state, A) = dot(w,PSI(state,A)).
                  That is, this function finds the best action to take in the given state
                  when our model is parameterized by the given weight vector.
        !*/

        void get_features(
            const state_type& state,
            const action_type& action,
            matrix<double,0,1>& feats
        ) const;
        /*!
            ensures
                - #feats.size() == num_features()
                - #feats == PSI(state, action)
        !*/

        action_type random_action(
            const state_type& state
        ) const;
        /*!
            ensures
                - returns a random plausible action assuming we are in the given state.
        !*/

        double reward(
            const state_type& state,
            const action_type& action,
            const state_type& new_state
        ) const;
        /*!
            requires
                - action is a pausible action from state.
                - new_state is a possible outcome when performing action on state.
            ensures
                - returns the reward obtained by reaching new_state from state
                  doing action.
        !*/

        state_type initial_state(
        ) const;
        /*!
            ensures
                - returns the initial state of the model.
        !*/

        state_type step(
            const state_type& state,
            const action_type& action
        ) const;
        /*!
            requires
                - action is a plausible action when we are in state.
            ensures
                - returns a new state result of being on the given state and doing the given
                  action.
        !*/

        bool is_success(
            const state_type& state
        ) const;
        /*!
            ensures
                - returns whether state is a goal state (the agent has finished properly).
        !*/

        bool is_failure(
            const state_type& state
        ) const;
        /*!
            ensures
                - returns whether state is a failure state, i.e., a state where the agent has
                  failed its task.
        !*/

        bool is_final(
            const state_type& state
        ) const;
        /*!
            ensures
                - #is_final(state) == is_success(state) || is_failure(state)
        !*/


    };

// ----------------------------------------------------------------------------------------

    template <
        typename model_type
        >
    struct process_sample
    {
        /*!
            REQUIREMENTS ON model_type
                model_type should implement one of the interfaces defined above this file.

            WHAT THIS OBJECT REPRESENTS
                This object holds a training sample for a reinforcement learning algorithm.
                In particular, it should be a sample from some process where the process
                was in state this->state, then took this->action action which resulted in
                receiving this->reward and ending up in the state this->next_state.
        !*/

        typedef typename model_type::state_type state_type;
        typedef typename model_type::action_type action_type;

        process_sample(){}

        process_sample(
            const state_type& s,
            const action_type& a,
            const state_type& n,
            const double& r
        ) : state(s), action(a), next_state(n), reward(r) {}

        state_type  state;
        action_type action;
        state_type  next_state;
        double reward;
    };

    template < typename model_type >
    void serialize (const process_sample<model_type>& item, std::ostream& out);
    template < typename model_type >
    void deserialize (process_sample<model_type>& item, std::istream& in);
    /*!
        provides serialization support.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename model_type
        >
    class policy
    {
        /*!
            REQUIREMENTS ON model_type
                model_type should implement one of the interfaces defined above this file.

            WHAT THIS OBJECT REPRESENTS
                This class represents a greedy policy, that is, it is a policy that given a
                state returns the best possible action based on its weight matrix.
        !*/

    public:

        typedef typename model_type::state_type state_type;
        typedef typename model_type::action_type action_type;

        policy (
            const model_type& model = model_type()
        );
        /*!
            ensures
                - #get_model() == model
                - #get_weights().size() == #get_model().num_features()
                - #get_weights() == 0
        !*/

        policy (
            const matrix<double,0,1>& weights,
            const model_type& model
        );
        /*!
            requires
                - model.num_features() == weights.size()
            ensures
                - #get_model() == model
                - #get_weights() == weights
        !*/

        action_type operator() (
            const state_type& state
        ) const;
        /*!
            ensures
                - returns get_model().find_best_action(state, this->weights);
        !*/

        const model_type& get_model (
        ) const;
        /*!
            ensures
                - returns the model used by this object
        !*/

        const matrix<double,0,1>& get_weights (
        ) const;
        /*!
            ensures
                - returns the weights that the policy is using.
        !*/

        matrix<double,0,1>& get_weights (
        );
        /*!
            ensures
                - returns the weights that the policy is using.
        !*/
    };

    template < typename model_type >
    void serialize(const policy<model_type>& item, std::ostream& out);
    template < typename model_type >
    void deserialize(policy<model_type>& item, std::istream& in);
    /*!
        provides serialization support.
    !*/

    // ----------------------------------------------------------------------------------------

    template <
        typename policy_type,
        typename prng_engine = std::default_random_engine()
        >
    class epsilon_policy
    {
        /*!
            REQUIREMENTS ON policy_type
                policy_type is an object with the same interface as the policy class defined
                above.

            REQUIREMENTS ON prng_engine
                prng_engine should be a PRNG interface like the ones defined in std::random.

            WHAT THIS OBJECT REPRESENTS
                This is a special policy that returns the best action (according to the
                underlying policy) for the given state with probability 1-epsilon
                while it returns a valid random action with probability epsilon.

                It is mainly used to add some exploration in the training process of the
                online reinforcement learning methods such as qlearning and sarsa.
        !*/

    public:

        typedef typename policy_type::state_type state_type;
        typedef typename policy_type::action_type action_type;

        epsilon_policy (
            double epsilon,
            const policy_type& policy,
            const prng_engine& gen = prng_engine()
        );
        /*!
            requires
                - epsilon >= 0 and epsilon <= 1
            ensures
                - #get_epsilon() == epsilon
                - #get_policy() == policy
                - #get_generator() == gen
        !*/

        action_type operator() (
            const state_type& state
        ) const;
        /*!
            ensures
                - returns get_policy()(state, w) with probability 1-epsilon
                  and get_model().random_action(state) with probability epsilon.
        !*/

        const policy_type& get_policy(
        ) const;
        /*!
            ensures
                - returns the underlying policy used by the object.
        !*/

        model_type get_model (
        ) const;
        /*!
            ensures
                - returns the model used by the underlying policy.
        !*/

        const matrix<double,0,1>& get_weights (
        ) const;
        /*!
            ensures
                - returns the weights that the policy is using.
        !*/

        matrix<double,0,1>& get_weights (
        );
        /*!
            ensures
                - returns the weights that the policy is using.
        !*/

        double get_epsilon(
        ) const;
        /*!
            ensures
                - returns the epsilon value used by the policy.
        !*/

        const prng_engine& get_generator(
        ) const;
        /*!
            ensures
                - returns the generator used by the policy.
        !*/

    };

    template < typename policy_type, typename generator >
    inline void serialize(const epsilon_policy<policy_type, generator>& item, std::ostream& out);
    template < typename policy_type, typename generator >
    inline void deserialize(epsilon_policy<policy_type, generator>& item, std::istream& in);
    /*!
        provides serialization support.
    !*/

// ----------------------------------------------------------------------------------------

#endif // DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_
 
