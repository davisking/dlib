// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_SOLVERS_ABSTRACT_H_
#ifdef DLIB_DNn_SOLVERS_ABSTRACT_H_

#include "tensor_abstract.h"
#include <iostream>

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class EXAMPLE_SOLVER 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A solver defines the parameter update rule for a single layer in a deep
                neural network.  It takes a parameter gradient vector and a layer and
                updates the layer's parameters.  Importantly, each solver instance is used
                with only one layer in a network.  This allows us to define solvers that
                have per layer state, for example, a solver may keep a momentum term and
                apply it to its update rule.

                Note that there is no dlib::EXAMPLE_SOLVER type.  It is shown here purely
                to document the interface a solver object must implement.
        !*/

    public:

        EXAMPLE_SOLVER(
        );

        template <typename LAYER_DETAILS>
        void operator() (
            LAYER_DETAILS& l, 
            const tensor& params_grad
        );
        /*!
            requires
                - LAYER_DETAILS implements the EXAMPLE_LAYER_ interface defined in
                  layers_abstract.h.
                - l.get_layer_params().size() != 0
                - have_same_dimensions(l.get_layer_params(), params_grad) == true.
                - When this function is invoked on a particular solver instance, it is
                  always supplied with the same LAYER_DETAILS object.
            ensures
                - Updates the parameters in l.  That is, l.get_layer_params() is modified
                  based on the parameter gradient vector stored in params_grad.
        !*/
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class sgd
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above.  It is a
                basic stochastic gradient descent solver which uses momentum and weight
                decay.  In particular, it performs the following update each time the
                solver is invoked:
                    v = momentum*v - weight_decay*learning_rate*l.get_layer_params() - learning_rate*params_grad;
                    l.get_layer_params() += v;
                Here v is a momentum term that is remembered by the solver from one
                invocation of operator() to the next.
        !*/
    public:

        sgd(
            float learning_rate = 0.001,
            float weight_decay = 0.0005,
            float momentum = 0.9 
        ); 
        /*!
            requires
                - learning_rate > 0
                - weight_decay >= 0
                - momentum >= 0
            ensures
                - #get_learning_rate() == learning_rate
                - #get_weight_decay()  == weight_decay 
                - #get_momentum()      == momentum 
        !*/

        float get_learning_rate () const; 
        float get_weight_decay () const;
        float get_momentum () const; 
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_ABSTRACT_H_

