// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POLICY_Hh_
#define DLIB_POLICY_Hh_

#include "../matrix.h"
#include "policy_abstract.h"
#include <iostream>
#include <random>

namespace dlib
{

    template <
        typename model_type
        >
    class greedy_policy
    {
    public:

        typedef model_type feature_extractor_type;
        typedef typename model_type::state_type state_type;
        typedef typename model_type::action_type action_type;

        greedy_policy (
            const model_type &model_
        ) : model(model_)
        {
            w.set_size(model.states_size());
            w = 0;
        }

        greedy_policy (
            const model_type &model_,
            const matrix<double,0,1>& weights_
        ) : w(weights_), model(model_) {}

        action_type operator() (
            const state_type& state
        ) const
        {
            return model.find_best_action(state,w);
        }

        const model_type& get_model (
        ) const { return model; }

        matrix<double,0,1>& get_weights (
        ) { return w; }

        const matrix<double,0,1>& get_weights (
        ) const { return w; }

    private:
        matrix<double,0,1> w;
        const model_type &model;
    };

    template < typename model_type >
    inline void serialize(const greedy_policy<model_type>& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.get_model(), out);
        serialize(item.get_weights(), out);
    }
    template < typename model_type >
    inline void deserialize(greedy_policy<model_type>& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::greedy_policy object.");
        model_type model;
        matrix<double,0,1> w;
        deserialize(model, in);
        deserialize(w, in);
        item = greedy_policy<model_type>(w,model);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename policy_type,
        typename generator = std::default_random_engine
        >
    class epsilon_policy
    {
    public:
        typedef typename policy_type::state_type state_type;
        typedef typename policy_type::action_type action_type;

        epsilon_policy (
            double epsilon_,
            const policy_type &policy_,
            const generator &gen_ = generator()
        ) : policy(policy_), epsilon(epsilon_), gen(gen_) {}

        action_type operator() (
            const state_type& state
        ) const
        {
            std::bernoulli_distribution d(epsilon);
            return d(gen) ? get_model().random_action(state) : policy(state);
        }

        policy_type get_policy(
        ) const { return policy; }

        auto get_model (
        ) const -> decltype(get_policy().get_model()) { return policy.get_model(); }

        matrix<double,0,1>& get_weights (
        ) { return policy.get_weights(); }

        const matrix<double,0,1>& get_weights (
        ) const { return policy.get_weights(); }

        double get_epsilon(
        ) const { return epsilon; }

        const generator& get_generator(
        ) const { return gen; }

    private:
        policy_type policy;
        double epsilon;

        mutable generator gen;
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
            throw serialization_error("Unexpected version found while deserializing dlib::greedy_policy object.");

        policy_type policy;
        double epsilon;
        generator gen;
        deserialize(policy, in);
        deserialize(epsilon, in);
        deserialize(gen, in);
        item = epsilon_policy<policy_type, generator>(epsilon, policy, gen);
    }

// ----------------------------------------------------------------------------------------

    // For backward compability with lspi
    template < typename model_type >
    using policy = greedy_policy<model_type>; //template aliasing is possible post C++11
}

#endif // DLIB_POLICY_Hh_
