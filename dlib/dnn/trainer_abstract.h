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
                - #get_max_num_epochs() == 10000
                - #get_mini_batch_size() == 128
                - #get_step_size() == 1
                - #get_min_step_size() == 1e-4
                - #get_iterations_between_step_size_adjust() == 2000
                - #get_step_size_shrink() == 0.1
        !*/

        explicit dnn_trainer(
            const net_type& net
        );
        /*!
            ensures
                - #get_net() == net 
                - #get_solvers() == a set of default initialized solvers.
                - #get_max_num_epochs() == 10000
                - #get_mini_batch_size() == 128
                - #get_step_size() == 1
                - #get_min_step_size() == 1e-4
                - #get_iterations_between_step_size_adjust() == 2000
                - #get_step_size_shrink() == 0.1
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
                - #get_max_num_epochs() == 10000
                - #get_mini_batch_size() == 128
                - #get_step_size() == 1
                - #get_min_step_size() == 1e-4
                - #get_iterations_between_step_size_adjust() == 2000
                - #get_step_size_shrink() == 0.1
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

        const std::vector<solver_type>& get_solvers (
        ) const; 
        /*!
            ensures
                - returns the solvers used to optimize each layer of the neural network
                  get_net().  In particular, the first layer's solver is
                  get_solvers()[0], the second layer's solver is
                  get_solvers()[1], and so on.
        !*/

        std::vector<solver_type>& get_solvers (
        ); 
        /*!
            ensures
                - returns the solvers used to optimize each layer of the neural network
                  get_net().  In particular, the first layer's solver is
                  get_solvers()[0], the second layer's solver is
                  get_solvers()[1], and so on.
                - It should be noted that you should never change the number of elements in
                  the vector returned by get_solvers() (i.e. don't do something that
                  changes get_solvers().size()).  It will be set to net_type::num_layers by
                  this object and you should leave it at that.  The non-const version of
                  get_solvers() is provided only so you can tweak the parameters of a
                  particular solver.
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

        unsigned long get_max_num_epochs (
        ) const; 
        /*!
            ensures
                - train() will execute at most get_max_num_epochs() iterations over the
                  training data before returning.
        !*/

        void set_max_num_epochs (
            unsigned long num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_max_num_epochs() == num
        !*/

        void set_setep_size (
            double ss
        );
        /*!
            requires
                - ss > 0
            ensures
                - #get_step_size() == ss
        !*/

        double get_step_size(
        ) const;
        /*!
            ensures
                - During each training step, a solver tells us how to modify the parameters
                  of each layer in the network.  It does this by outputting a step vector,
                  that when added to the parameters, will hopefully result in improved
                  network performance.  In our case, at during each step, we multiply the
                  step vector from the solver by get_step_size() before adding it to the
                  parameters.  Therefore, get_step_size() controls the "learning rate" used
                  during training. 
        !*/

        void set_min_step_size (
            double ss
        );
        /*!
            requires
                - ss > 0
            ensures
                - #get_min_step_size() == ss
        !*/

        double get_min_step_size (
        ) const;
        /*!
            ensures
                - During training, this object will test if progress is still being made
                  and if it isn't then it will reduce get_step_size() by setting it to
                  get_step_size()*get_step_size_shrink().  However, it will not reduce it
                  below get_min_step_size().  Once this minimum step size is crossed the
                  training will terminate.
        !*/

        void set_iterations_between_step_size_adjust (
            unsigned long min_iter
        );
        /*!
            ensures
                - #get_iterations_between_step_size_adjust() == min_iter
        !*/

        unsigned long get_iterations_between_step_size_adjust (
        ) const;
        /*!
            ensures
                - This object monitors the progress of training and estimates if the
                  training error is being reduced.  It does this by looking at
                  get_iterations_between_step_size_adjust() mini-batch results and applying
                  the statistical test defined by the running_gradient object to see if the
                  training error is getting smaller.  

                  Therefore, get_iterations_between_step_size_adjust() should always be set
                  to something sensibly large so that this test can be done with reasonably
                  high confidence.
        !*/

        void set_step_size_shrink_amount (
            double shrink
        );
        /*!
            requires
                - 0 < shrink && shrink <= 1
            ensures
                - #get_step_size_shrink() == shrink
        !*/

        double get_step_size_shrink (
        ) const;
        /*!
            ensures
                - Whenever the training routine thinks it isn't making progress anymore it
                  will reduce get_step_size() by multiplying it by get_step_size_shrink().
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
                - The optimizer will run until get_step_size() < get_min_step_size() or
                  get_max_num_epochs() training epochs have been executes. 
                - Each layer in the network will be optimized by its corresponding solver
                  in get_solvers().  
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
                - The optimizer will run until get_step_size() < get_min_step_size() or
                  get_max_num_epochs() training epochs have been executes. 
                - Each layer in the network will be optimized by its corresponding solver
                  in get_solvers().  
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


