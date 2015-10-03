// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_TRAINER_ABSTRACT_H_
#ifdef DLIB_DNn_TRAINER_ABSTRACT_H_

#include "core_abstract.h"
#include "solvers_abstract.h"
#include <vector>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer
    {
        /*!
            REQUIREMENTS ON net_type
                - net_type is an add_loss_layer object.

            REQUIREMENTS ON solver_type
                - solver_type is an implementation of the EXAMPLE_SOLVER interface defined
                  in solvers_abstract.h

            WHAT THIS OBJECT REPRESENTS

        !*/

    public:

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer(
        );

        explicit dnn_trainer(
            const net_type& net
        );

        dnn_trainer(
            const net_type& net, 
            const solver_type& solver
        ); 

        const net_type& get_net (
        ) const; 

        void set_net (
            const net_type& net
        ); 

        void set_solver (
            const solver_type& solver_
        );

        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const; 

        sstack<solver_type,net_type::num_layers>& get_solvers (
        ); 

        unsigned long get_mini_batch_size (
        ) const; 

        void set_mini_batch_size (
            unsigned long batch_size 
        );

        unsigned long get_num_epochs (
        ) const; 

        void set_num_epochs (
            unsigned long num
        ) const;

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ); 
        /*!
            requires
                - data.size() == labels.size()
                - TODO: the net has a supervised loss layer.
        !*/

        const net_type& train (
            const std::vector<input_type>& data
        );
        /*!
            requires 
                - TODO: the net has an unsupervised loss layer.
            ensures
                - trains an auto-encoder
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_ABSTRACT_H_ 


