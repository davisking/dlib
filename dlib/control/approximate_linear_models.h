// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_APPROXIMATE_LINEAR_MODELS_Hh_
#define DLIB_APPROXIMATE_LINEAR_MODELS_Hh_

#include "approximate_linear_models_abstract.h"

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
        typedef typename model_type::reward_type reward_type;

        process_sample(){}

        process_sample(
            const state_type& s,
            const action_type& a,
            const state_type& n,
            const reward_type& r
        ) : state(s), action(a), next_state(n), reward(r) {}

        state_type  state;
        action_type action;
        state_type  next_state;
        reward_type reward;
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

}

#endif // DLIB_APPROXIMATE_LINEAR_MODELS_Hh_

