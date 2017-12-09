// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_
#ifdef DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_

#include "model_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename U
    >
    struct example_feature_extractor 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object defines the interface a feature extractor must implement if it
                is to be used with the process_sample and policy objects defined at
                policy_abstract.h.  Moreover, it is meant to represent the core part
                of a model used in a reinforcement learning algorithm.
                
                In particular, this object models a Q(state,action) function where
                    Q(state,action) == dot(w, PSI(state,action))
                    where PSI(state,action) is a feature vector and w is a parameter
                    vector.

                Therefore,  a feature extractor defines how the PSI(x,y) feature vector is
                calculated.  It also defines the types used to represent the state and
                action objects. 


            THREAD SAFETY
                Instances of this object are required to be threadsafe, that is, it should
                be safe for multiple threads to make concurrent calls to the member
                functions of this object.
        !*/

        typedef T state_type;
        typedef U action_type;
        // We can also say that the last element in the weight vector w must be 1.  This
        // can be useful for including a prior into your model.
        const static bool force_last_weight_to_1 = false;

        example_feature_extractor(
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

        void get_features (
            const state_type& state,
            matrix<double,0,1>& feats
        ) const;
        /*!
            ensures
                - #feats.size() == num_features()
                - #feats == PSI(state,action)
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
                model_type should implement the interface defined at model_abstract.h.

            WHAT THIS OBJECT REPRESENTS
                This object holds a training sample for a reinforcement learning algorithm.
                In particular, it should be a sample from some process where the process
                was in state this->state, then took this->action action which resulted in
                receiving this->reward and ending up in the state this->next_state.
        !*/

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
    void serialize (const process_sample<feature_extractor>& item, std::ostream& out);
    template < typename feature_extractor >
    void deserialize (process_sample<feature_extractor>& item, std::istream& in);
    /*!
        provides serialization support.
    !*/

// ----------------------------------------------------------------------------------------

#endif // DLIB_APPROXIMATE_LINEAR_MODELS_ABSTRACT_Hh_
 
