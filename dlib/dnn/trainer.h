// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TRAINER_H_
#define DLIB_DNn_TRAINER_H_

#include "trainer_abstract.h"
#include "core.h"
#include "solvers.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer
    {
    public:

        static_assert(is_loss_layer_type<net_type>::value, 
            "The last layer in a network must be a loss layer.");

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer(
        ) 
        {
            init();
        }

        explicit dnn_trainer(const net_type& net_) :  net(net_) 
        {
            init();
        }

        dnn_trainer(
            const net_type& net_, 
            const solver_type& solver_
        ) : net(net_), solvers(solver_) 
        {
            init();
        }

        const net_type& get_net (
        ) const { return net; }

        void set_net (
            const net_type& net_
        ) 
        { 
            return net = net_; 
        }

        void set_solver (
            const solver_type& solver_
        ) 
        { 
            solvers = solver_; 
        }

        unsigned long get_mini_batch_size (
        ) const { return mini_batch_size; }

        void set_mini_batch_size (
            unsigned long batch_size 
        )
        {
            DLIB_CASSERT(batch_size > 0,"");
            mini_batch_size = batch_size;
        }

        unsigned long get_num_epochs (
        ) const { return num_epochs; }

        void set_num_epochs (
            unsigned long num
        ) const 
        {
            DLIB_CASSERT(num > 0,"");
            num_epochs = num;
        }


        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const { return solvers; }

        sstack<solver_type,net_type::num_layers>& get_solvers (
        ) { return solvers; }

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ) 
        {
            DLIB_CASSERT(data.size() == labels.size() && data.size() > 0, "");

            resizable_tensor t1, t2;


            for (unsigned long epoch_iteration = 0; epoch_iteration < num_epochs; ++epoch_iteration)
            {
                unsigned long j = 0;

                // Load two tensors worth of data at once so we can overlap the computation
                // and data transfer between the host and the device.
                if (j < data.size())
                {
                    net.to_tensor(data.begin()+j, 
                                  data.begin()+std::min(j+mini_batch_size,data.size()), t1);
                    j += mini_batch_size;
                }
                if (j < data.size())
                {
                    net.to_tensor(data.begin()+j, 
                                  data.begin()+std::min(j+mini_batch_size,data.size()), t2);
                    j += mini_batch_size;
                }

                unsigned long i = 0;
                while (i < data.size())
                {
                    net.update(t1, labels.begin()+i, solvers);
                    i += mini_batch_size;
                    if (j < data.size())
                    {
                        net.to_tensor(data.begin()+j, 
                                      data.begin()+std::min(j+mini_batch_size,data.size()), t1);
                        j += mini_batch_size;
                    }

                    if (i < data.size())
                    {
                        net.update(t2, labels.begin()+i, solvers);
                        i += mini_batch_size;
                        if (j < data.size())
                        {
                            net.to_tensor(data.begin()+j, 
                                          data.begin()+std::min(j+mini_batch_size,data.size()), t2);
                            j += mini_batch_size;
                        }
                    }

                }
            }
            return net;
        }

        const net_type& train (
            const std::vector<input_type>& data
        ) 
        {
            DLIB_CASSERT(data.size() > 0, "");

            const bool has_unsupervised_loss = std::is_same<no_label_type, label_type>::value; 
            static_assert(has_unsupervised_loss, 
                "You can only call this version of train() when using an unsupervised loss.");

            resizable_tensor t1, t2;

            for (unsigned long epoch_iteration = 0; epoch_iteration < num_epochs; ++epoch_iteration)
            {
                unsigned long j = 0;

                // Load two tensors worth of data at once so we can overlap the computation
                // and data transfer between the host and the device.
                if (j < data.size())
                {
                    net.to_tensor(data.begin()+j, 
                                  data.begin()+std::min(j+mini_batch_size,data.size()), t1);
                    j += mini_batch_size;
                }
                if (j < data.size())
                {
                    net.to_tensor(data.begin()+j, 
                                  data.begin()+std::min(j+mini_batch_size,data.size()), t2);
                    j += mini_batch_size;
                }

                unsigned long i = 0;
                while (i < data.size())
                {
                    net.update(t1, solvers);
                    i += mini_batch_size;
                    if (j < data.size())
                    {
                        net.to_tensor(data.begin()+j, 
                                      data.begin()+std::min(j+mini_batch_size,data.size()), t1);
                        j += mini_batch_size;
                    }

                    if (i < data.size())
                    {
                        net.update(t2, solvers);
                        i += mini_batch_size;
                        if (j < data.size())
                        {
                            net.to_tensor(data.begin()+j, 
                                          data.begin()+std::min(j+mini_batch_size,data.size()), t2);
                            j += mini_batch_size;
                        }
                    }

                }
            }
            return net;
        }

    private:

        void init()
        {
            num_epochs = 300;
            mini_batch_size = 11;
        }

        unsigned long num_epochs;
        unsigned long mini_batch_size;

        net_type net;
        sstack<solver_type,net_type::num_layers> solvers;
    };

    // TODO, make dnn_trainer serializable. 

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_H_

