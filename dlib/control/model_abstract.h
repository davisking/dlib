// Copyright (C) 2017 Adrián Javaloy (adrian.javaloy@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MODEL_ABSTRACT_Hh_
#ifdef DLIB_MODEL_ABSTRACT_Hh_

#include "approximate_linear_models_abstract.h"
#include "../matrix.h"

namespace dlib
{

    template <
            template<typename, typename> typename feature_extractor_type
            >
    class example_model
    {
        /*!
            REQUIREMENTS ON feature_extractor
                feature_extractor should implement the example_feature_extractor interface defined
                at approximate_linear_models_abstract.h.

            WHAT THIS OBJECT REPRESENTS
                This is an example interface of a model class. This class represents an environment
                where an agent will be deployed at. In particular, this class includes information
                about the state space, action space and how to represent those states feature-wise.
        !*/
    public:

        // You have to define state, action and reward types.
        typedef U state_type;
        typedef V action_type;
        typedef W reward_type;

        // The feature extractor uses the same types as the model.
        typedef feature_extractor_type<state_type, action_type> feature_extractor;

        example_model(
        );
        /*!
            ensures
                - #get_feature_extractor() == feature_extractor()
        !*/

        action_type random_action(
            const state_type &state
        ) const;
        /*!
            ensures
                - returns a random reachable action from state.
        !*/

        action_type find_best_action(
            const state_type &state,
            const matrix<double,0,1> &w
        ) const;
        /*!
            requires
                - w.size() == states_size()
            ensures
                - returns the action that maximizes the product
                  dot(w, get_feature_extractor().get_features(state)).
        !*/

        const feature_extractor& get_feature_extractor(
        ) const;
        /*!
            ensures
                - returns the feature_extractor used by the model.
        !*/

        auto states_size(
        ) const -> decltype(get_feature_extractor().num_features());
        /*!
            ensures
                - returns get_feature_extractor().num_features().
        !*/

        auto get_features(
            const state_type &state
        ) const -> decltype(get_feature_extractor().get_features(state));
        /*!
            ensures
                - returns get_feature_extractor().get_features(state);
        !*/

        state_type initial_state(
        ) const;
        /*!
            ensures
                - returns the initial state of the model.
        !*/

        state_type step(
            const state_type &state,
            const action_type &action
        ) const;
        /*!
            requires
                - action is a valid action from state.
            ensures
                - returns a state that is possible to be in after doing action
                  from state.
        !*/

        // The new_state parameter is need because the model doesn't have to be deterministic.
        // Nonetheless for now I'll suppose that the reward is deterministic.
        reward_type reward(
            const state_type &state,
            const action_type &action,
            const state_type &new_state
        ) const;
        /*!
            requires
                - is possible to be in new_state after doing action from state.
            ensures
                - returns the reward obtained for reaching new_state from state after
                  doing action.
        !*/


    };

    template < template<typename, typename> typename feature_extractor >
    void serialize (const example_model<feature_extractor>& item, std::ostream& out);
    template < template<typename, typename> typename feature_extractor >
    void deserialize (example_model<feature_extractor>& item, std::istream& in);
    /*!
        provides serialization support.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif
