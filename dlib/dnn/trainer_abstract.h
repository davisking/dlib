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
                This object is a tool training a deep neural network. To use it you supply
                a neural network type and a solver, then you call train() with your
                training data and it will output a new network instance that has hopefully
                learned something useful from your training data.

        !*/

    public:

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer(
        );
        /*!
            ensures
                - #get_net() == a default initialized net_type object.
                - #get_solvers() == a set of default initialized solvers.
        !*/

        explicit dnn_trainer(
            const net_type& net
        );
        /*!
            ensures
                - #get_net() == net 
                - #get_solvers() == a set of default initialized solvers.
        !*/

        dnn_trainer(
            const net_type& net, 
            const solver_type& solver
        ); 
        /*!
            ensures
                - #get_net() == net 
                - #get_solvers() == a set of solvers that are all initialized with the
                  provided solver instance.
        !*/

        const net_type& get_net (
        ) const; 
        /*!
            ensures
                - returns the neural network object in this trainer.  This is the network
                  that is optimized when you call train().
        !*/

        void set_net (
            const net_type& net
        ); 
        /*!
            ensures
                - #get_net() == net
        !*/

        void set_solver (
            const solver_type& solver
        );
        /*!
            ensures
                - assigns solver to all the solvers in this object. I.e.  solver will be
                  assigned to each element in get_solvers(). 
        !*/

        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const; 
        /*!
            ensures
                - returns the solvers used to optimize each layer of the neural network
                  get_net().  In particular, the first layer's solver is
                  get_solvers().top(), the second layer's solver is
                  get_solvers().pop().top(), and so on.
        !*/

        sstack<solver_type,net_type::num_layers>& get_solvers (
        ); 
        /*!
            ensures
                - returns the solvers used to optimize each layer of the neural network
                  get_net().  In particular, the first layer's solver is
                  get_solvers().top(), the second layer's solver is
                  get_solvers().pop().top(), and so on.
        !*/

        unsigned long get_mini_batch_size (
        ) const; 
        /*!
            ensures
                - During training, we call the network's update() routine over and over
                  with training data.  The number of training samples we give to each call
                  to update is the "mini-batch size", which is defined by
                  get_mini_batch_size().
        !*/

        void set_mini_batch_size (
            unsigned long batch_size 
        );
        /*!
            requires
                - batch_size > 0
            ensures
                - #get_mini_batch_size() == batch_size
        !*/

        unsigned long get_num_epochs (
        ) const; 
        /*!
            ensures
                - Returns the number of passes over the training data we will execute when
                  train() is called. 
        !*/

        void set_num_epochs (
            unsigned long num
        );
        /*!
            requires
                - num > 0
            ensures
                - @get_num_epochs() == num
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that a 
                  user can observe the progress of the algorithm.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - This object will not print anything to standard out
        !*/

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ); 
        /*!
            requires
                - data.size() == labels.size()
                - net_type uses a supervised loss.  
                  i.e. net_type::label_type != no_label_type.
            ensures
                - Trains a supervised neural network based on the given training data.
                  The goal of training is to find the network parameters that minimize
                  get_net().compute_loss(data.begin(), data.end(), labels.begin()). 
                - The optimizer will run for get_num_epochs() epochs and each layer in the
                  network will be optimized by its corresponding solver in get_solvers().  
                - returns #get_net()
                  (i.e. the trained network can also be accessed by calling get_net() after
                  train() finishes executing)
                - Each call to train DOES NOT reinitialize the state of get_net() or
                  get_solvers().  That is, the state of the solvers and network contained
                  inside this trainer is the starting point for the optimization each time
                  train() is called.  For example, calling train() 1 time and having it
                  execute 100 epochs of training is equivalent to calling train() 10 times
                  and having it execute 10 epochs of training during each call.  This also
                  means you can serialize a trainer to disk and then, at a later date,
                  deserialize it and resume training your network where you left off.
                - You can obtain the average loss value during the final training epoch by
                  calling get_average_loss().
        !*/

        const net_type& train (
            const std::vector<input_type>& data
        );
        /*!
            requires 
                - net_type uses an unsupervised loss.  
                  i.e. net_type::label_type == no_label_type.
            ensures
                - Trains an unsupervised neural network based on the given training data.
                  The goal of training is to find the network parameters that minimize
                  get_net().compute_loss(data.begin(), data.end()). 
                - The optimizer will run for get_num_epochs() epochs and each layer in the
                  network will be optimized by its corresponding solver in get_solvers().  
                - returns #get_net()
                  (i.e. the trained network can also be accessed by calling get_net() after
                  train() finishes executing)
                - Each call to train DOES NOT reinitialize the state of get_net() or
                  get_solvers().  That is, the state of the solvers and network contained
                  inside this trainer is the starting point for the optimization each time
                  train() is called.  For example, calling train() 1 time and having it
                  execute 100 epochs of training is equivalent to calling train() 10 times
                  and having it execute 10 epochs of training during each call.  This also
                  means you can serialize a trainer to disk and then, at a later date,
                  deserialize it and resume training your network where you left off.
                - You can obtain the average loss value during the final training epoch by
                  calling get_average_loss().
        !*/

        void train_one_step (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        );
        /*!
            requires
                - data.size() == labels.size()
                - net_type uses a supervised loss.  
                  i.e. net_type::label_type != no_label_type.
            ensures
                - Performs one stochastic gradient update step based on the mini-batch of
                  data and labels supplied to this function.  In particular, calling
                  train_one_step() in a loop is equivalent to calling the train() method
                  defined above.  However, train_one_step() allows you to stream data from
                  disk into the training process while train() requires you to first load
                  all the training data into RAM.  Otherwise, these training methods are
                  equivalent.
                - You can observe the current average loss value by calling get_average_loss().
        !*/

        void train_one_step (
            const std::vector<input_type>& data
        );
        /*!
            requires
                - net_type uses an unsupervised loss.  
                  i.e. net_type::label_type == no_label_type.
            ensures
                - Performs one stochastic gradient update step based on the mini-batch of
                  data supplied to this function.  In particular, calling train_one_step()
                  in a loop is equivalent to calling the train() method defined above.
                  However, train_one_step() allows you to stream data from disk into the
                  training process while train() requires you to first load all the
                  training data into RAM.  Otherwise, these training methods are
                  equivalent.
                - You can observe the current average loss value by calling get_average_loss().
        !*/

        double get_average_loss (
        ) const;
        /*!
            ensures
                - returns the average loss value observed during previous calls to
                  train_one_step() or train().  That is, the average output of
                  net_type::update() during the previous mini-batch updates.
        !*/

        void clear_average_loss (
        );
        /*!
            ensures
                - #get_average_loss() == 0
                - get_average_loss() uses a dlib::running_stats object to keep a running
                  average of the loss values seen during the previous mini-batch updates
                  applied during training.  Calling clear_average_loss() resets the
                  running_stats object so it forgets about all previous loss values
                  observed.
        !*/

    };

    template <typename T, typename U>
    void serialize(const dnn_trainer<T,U>& item, std::ostream& out);
    template <typename T, typename U>
    void deserialize(dnn_trainer<T,U>& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_ABSTRACT_H_ 


