// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POLICY_Hh_
#define DLIB_POLICY_Hh_

#include <random>
#include "../matrix.h"
#include "policy_abstract.h"

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
        )
        {
            w.set_size(model.states_size());
            w = 0;
        }

        greedy_policy (
            const matrix<double,0,1>& weights_,
            const model_type& model_ = model_type()
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
        model_type model;
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
        typename model_type,
        typename generator = std::default_random_engine
        >
    class epsilon_policy
    {
    public:

        typedef model_type feature_extractor_type;
        typedef typename model_type::state_type state_type;
        typedef typename model_type::action_type action_type;

        epsilon_policy (
            double epsilon_,
            const generator &gen_ = std::default_random_engine()
        ) : epsilon(epsilon_), gen(gen_)
        {
            w.set_size(model.states_size());
            w = 0;
        }

        epsilon_policy (
            double epsilon_,
            const matrix<double,0,1>& weights_,
            const model_type& model_ = model_type(),
            const generator gen_ = std::default_random_engine()
        ) : w(weights_), model(model_), epsilon(epsilon_), gen(gen_) {}

        action_type operator() (
            const state_type& state
        ) const
        {
            std::bernoulli_distribution d(epsilon);
            if(d(gen)){
  //              std::cout << "random\n";
                return model.random_action(state);
            }
            else{
//                std::cout << "best\n";
                return model.find_best_action(state,w);
            }
            //return d(gen) ? model.random_action(state) : model.find_best_action(state,w);
        }

        const model_type& get_model (
        ) const { return model; }

        matrix<double,0,1>& get_weights (
        ) { return w; }

        const matrix<double,0,1>& get_weights (
        ) const { return w; }

        double get_epsilon(
        ) const { return epsilon; }

        const generator& get_generator(
        ) const { return gen; }

    private:
        matrix<double,0,1> w;
        model_type model;
        double epsilon;

        mutable generator gen;
    };

    template < typename model_type, typename generator >
    inline void serialize(const epsilon_policy<model_type, generator>& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.get_model(), out);
        serialize(item.get_weights(), out);
        serialize(item.get_epsilon(), out);
        serialize(item.get_generator(), out);
    }
    template < typename model_type, typename generator >
    inline void deserialize(epsilon_policy<model_type, generator>& item, std::istream& in)
    {
        int version = 0;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::greedy_policy object.");
        model_type model;
        matrix<double,0,1> w;
        double epsilon;
        generator gen;
        deserialize(model, in);
        deserialize(w, in);
        deserialize(epsilon, in);
        deserialize(gen, in);
        item = epsilon_policy<model_type, generator>(w,model, epsilon, gen);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_POLICY_Hh_
