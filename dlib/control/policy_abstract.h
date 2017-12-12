// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_POLICY_ABSTRACT_Hh_
#ifdef DLIB_POLICY_ABSTRACT_Hh_

#include "../matrix.h"
#include "model_abstract.h"
#include <random>

namespace dlib
{

template <
    typename model_type
    >
class example_policy
{
    /*!
        REQUIREMENTS ON model_type
            model_type should implement the interface defined at model_abstract.h.

        WHAT THIS OBJECT REPRESENTS
            This is a policy based on the supplied model_type model.  In
            particular, it maps from model_type::state_type to a model_type::action
            to take in that state.
    !*/

public:

    typedef typename model_type::state_type state_type;
    typedef typename model_type::action_type action_type;

    example_policy (
        const model_type &model
    );
    /*!
        ensures
            - #get_model() == model
            - #get_weights().size() == #get_model().states_size()
            - #get_weights() == 0
    !*/

    example_policy (
        const model_type& model,
        const matrix<double,0,1>& weights
    );
    /*!
        requires
            - model.states_size() == weights.size()
        ensures
            - #get_model() == model
            - #get_weights() == weights
    !*/

    action_type operator() (
        const state_type& state
    ) const;

    const model_type& get_model (
    ) const;
    /*!
        ensures
            - returns the model used by this object
    !*/

    matrix<double,0,1>& get_weights (
    );
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
    !*/

    const matrix<double,0,1>& get_weights (
    ) const;
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
    !*/

};

template < typename model_type >
void serialize(const example_policy<model_type>& item, std::ostream& out);
template < typename model_type >
void deserialize(example_policy<model_type>& item, std::istream& in);
/*!
    provides serialization support.
!*/

// ----------------------------------------------------------------------------------------

template <
    typename model_type
    >
class greedy_policy
{
    /*!
        REQUIREMENTS ON model_type
            model_type should implement the interface defined at model_abstract.h.

        WHAT THIS OBJECT REPRESENTS
            This is an implementation of the policy interface that returns the best action
            based on the weights (i.e. it acts in a greedy fashion).
    !*/

public:

    typedef typename model_type::state_type state_type;
    typedef typename model_type::action_type action_type;

    greedy_policy (
        const model_type &model
    );
    /*!
        ensures
            - #get_model() == model
            - #get_weights().size() == #get_model().states_size()
            - #get_weights() == 0
    !*/

    greedy_policy (
        const model_type& model,
        const matrix<double,0,1>& weights
    );
    /*!
        requires
            - model.states_size() == weights.size()
        ensures
            - #get_model() == model
            - #get_weights() == weights
    !*/

    action_type operator() (
        const state_type& state
    ) const;
    /*!
        ensures
            - returns get_model().find_best_action(state, w);
    !*/

    const model_type& get_model (
    ) const;
    /*!
        ensures
            - returns the model used by this object
    !*/

    matrix<double,0,1>& get_weights (
    );
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
    !*/

    const matrix<double,0,1>& get_weights (
    ) const;
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
    !*/

};

template < typename model_type >
void serialize(const greedy_policy<model_type>& item, std::ostream& out);
template < typename model_type >
void deserialize(greedy_policy<model_type>& item, std::istream& in);
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
            policy_type should implement the example_policy interface defined at the
            top of this file.

        REQUIREMENTS ON prng_engine
            prng_engine should be a PRNG class like the ones defined in std::random.

        WHAT THIS OBJECT REPRESENTS
            This is a special policy that returns the best action (according to the
            underlying policy) for the given state with probability 1-epsilon
            while it returns a valid random action with probability epsilon.
    !*/

public:

    typedef typename policy_type::state_type state_type;
    typedef typename policy_type::action_type action_type;

    epsilon_policy (
        double epsilon,
        const policy_type &policy,
        const prng_engine &gen = prng_engine()
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

    policy_type get_policy(
    ) const;
    /*!
        ensures
            - returns the underlying policy used by the object.
    !*/

    auto get_model (
    ) const -> decltype(get_policy().get_model());
    /*!
        ensures
            - returns the model used by the underlying policy.
    !*/

    matrix<double,0,1>& get_weights (
    );
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
    !*/

    const matrix<double,0,1>& get_weights (
    ) const;
    /*!
        ensures
            - returns the parameter vector (w) associated with this object.  The length
              of the vector is get_model().states_size().
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

}

#endif // DLIB_POLICY_ABSTRACT_Hh_
