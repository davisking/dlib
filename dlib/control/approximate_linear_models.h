// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_APPROXIMATE_LINEAR_MODELS_Hh_
#define DLIB_APPROXIMATE_LINEAR_MODELS_Hh_

#include "approximate_linear_models_abstract.h"
#include <dlib/matrix.h>
#include <random>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename model_type
        >
    struct process_sample
    {
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

    template < typename feature_extractor >
    void serialize (const process_sample<feature_extractor>& item, std::ostream& out)
    {
        serialize(item.state, out);
        serialize(item.action, out);
        serialize(item.next_state, out);
        serialize(item.reward, out);
    }

    template < typename feature_extractor >
    void deserialize (process_sample<feature_extractor>& item, std::istream& in)
    {
        deserialize(item.state, in);
        deserialize(item.action, in);
        deserialize(item.next_state, in);
        deserialize(item.reward, in);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename model_type
        >
    class policy
    {
    public:

        typedef typename model_type::state_type state_type;
        typedef typename model_type::action_type action_type;

        policy (
            const model_type& model_ = model_type()
        ) : model(model_)
        {
            weights.set_size(model.num_features());
            weights = 0;
        }

        policy (
            const matrix<double,0,1>& weights_,
            const model_type &model_
        ) : weights(weights_), model(model_) {}

        action_type operator() (
            const state_type& state
        ) const
        {
            return model.find_best_action(state,weights);
        }

        const model_type& get_model (
        ) const { return model; }

        const matrix<double,0,1>& get_weights (
        ) const { return weights; }

        matrix<double,0,1>& get_weights (
        ) { return weights; }

    private:
        matrix<double,0,1> weights;
        const model_type model;
    };

    template < typename model_type >
    inline void serialize(const policy<model_type>& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.get_model(), out);
        serialize(item.get_weights(), out);
    }
    template < typename model_type >
    inline void deserialize(policy<model_type>& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::policy object.");
        model_type model;
        matrix<double,0,1> w;
        deserialize(model, in);
        deserialize(w, in);
        item = policy<model_type>(w,model);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename policy_type,
        typename prng_engine = std::default_random_engine
        >
    class epsilon_policy
    {
    public:
        typedef typename policy_type::state_type state_type;
        typedef typename policy_type::action_type action_type;

        epsilon_policy (
            double epsilon_,
            policy_type &policy_,
            const prng_engine &gen_ = prng_engine()
        ) : underlying_policy(policy_), epsilon(epsilon_), gen(gen_) {}

        action_type operator() (
            const state_type& state
        ) const
        {
            std::bernoulli_distribution d(epsilon);
            return d(gen) ? get_model().random_action(state) : underlying_policy(state);
        }

        const policy_type& get_policy(
        ) const { return underlying_policy; }

        auto get_model (
        ) const -> decltype(this->get_policy().get_model()) { return underlying_policy.get_model(); }

        matrix<double,0,1>& get_weights (
        ) { return underlying_policy.get_weights(); }

        const matrix<double,0,1>& get_weights (
        ) const { return underlying_policy.get_weights(); }

        double get_epsilon(
        ) const { return epsilon; }

        const prng_engine& get_generator(
        ) const { return gen; }

    private:
        policy_type& underlying_policy;
        double epsilon;

        mutable prng_engine gen;
    };

    template < typename policy_type, typename generator >
    inline void serialize(const epsilon_policy<policy_type, generator>& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.get_policy(), out);
        serialize(item.get_epsilon(), out);
        serialize(item.get_generator(), out);
    }

    template < typename policy_type, typename generator >
    inline void deserialize(epsilon_policy<policy_type, generator>& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::policy object.");

        policy_type policy;
        double epsilon;
        generator gen;
        deserialize(policy, in);
        deserialize(epsilon, in);
        deserialize(gen, in);
        item = epsilon_policy<policy_type, generator>(epsilon, policy, gen);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_APPROXIMATE_LINEAR_MODELS_Hh_

