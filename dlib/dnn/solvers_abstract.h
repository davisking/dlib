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
                neural network.  It takes a parameter gradient vector and the layer's
                parameters and tells you how the parameters should be updated.
                Importantly, each solver instance is used with only one layer in a network.
                This allows us to define solvers that have per layer state, for example, a
                solver may keep a momentum term and apply it to its update rule.

                Note that there is no dlib::EXAMPLE_SOLVER type.  It is shown here purely
                to document the interface a solver object must implement.
        !*/

    public:

        EXAMPLE_SOLVER(
        );

        const tensor& operator() (
            const tensor& params,
            const tensor& params_grad
        )
        /*!
            requires
                - params.size() != 0
                - have_same_dimensions(params, params_grad) == true.
                - When this function is invoked on a particular solver instance, it is
                  always supplied with parameters from the same layer instance.  That is,
                  the solver is allowed to remember things from one invocation to another
                  and to assume that it is being serially applied to optimize the same
                  parameters. 
            ensures
                - Returns a step vector V that is intended to be used to update the
                  parameters by adding V to params.
        !*/
    };

    void serialize(const EXAMPLE_SOLVER& item, std::ostream& out);
    void deserialize(EXAMPLE_SOLVER& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

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
            float learning_rate = 0.01,
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

    void serialize(const sgd& item, std::ostream& out);
    void deserialize(sgd& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

// ----------------------------------------------------------------------------------------

    class adam
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the EXAMPLE_SOLVER interface defined above.  In
                particular, it implements the ADAM parameter update method described in the
                paper:
                    Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic
                    optimization." International Conference on Learning Representation. 2015.
        !*/

    public:

        adam(
            float learning_rate = 0.001,
            float weight_decay = 0.0005,
            float momentum1 = 0.9, 
            float momentum2 = 0.999 
        ); 
        /*!
            requires
                - learning_rate > 0
                - weight_decay >= 0
                - 0 <= momentum1 < 1
                - 0 <= momentum2 < 1
            ensures
                - #get_learning_rate() == learning_rate
                - #get_weight_decay()  == weight_decay 
                - #get_momentum1()     == momentum1
                - #get_momentum2()     == momentum2
        !*/

        float get_learning_rate () const; 
        float get_weight_decay () const;
        float get_momentum1 () const; 
        float get_momentum2 () const; 
    };

    void serialize(const adam& item, std::ostream& out);
    void deserialize(adam& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_ABSTRACT_H_

