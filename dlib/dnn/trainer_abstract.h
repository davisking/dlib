// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_TRAINER_ABSTRACT_H_
#ifdef DLIB_DNn_TRAINER_ABSTRACT_H_

#include "core_abstract.h"
#include "solvers_abstract.h"
#include <vector>
#include <chrono>


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

                If you are compiling with CUDA then this object will use the GPU that is
                currently selected (i.e. the one indicated by cudaGetDevice()) when
                dnn_trainer is constructed.  It will continue to use that device even if
                you later change it by a call to cudaSetDevice().
        !*/

    public:

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;
        const static size_t num_computational_layers = net_type::num_computational_layers;

        dnn_trainer() = delete;
        dnn_trainer(const dnn_trainer&) = delete;
        dnn_trainer& operator=(const dnn_trainer&) = delete;

        dnn_trainer(
            net_type& net, 
            const solver_type& solver = solver_type(),
            const std::vector<int>& cuda_extra_devices = {}
        ); 
        /*!
            requires
                - for all valid i:
                    - 0 <= cuda_extra_devices[i] < dlib::cuda::get_num_devices()
            ensures
                - &#get_net() == &net 
                  (i.e. The dnn_trainer holds a reference to net, it does not copy it.
                  Therefore, you must ensure net has a lifetime at least as long as the
                  dnn_trainer).
                - #get_solvers() == a set of solvers that are all initialized with the
                  provided solver instance.
                - #get_max_num_epochs() == 10000
                - #get_mini_batch_size() == 128
                - #get_learning_rate() == 1e-2 
                - #get_min_learning_rate() == 1e-5
                - #get_iterations_without_progress_threshold() == 2000
                - #get_learning_rate_shrink_factor() == 0.1
                - #get_learning_rate_schedule().size() == 0
                - #get_train_one_step_calls() == 0
                - if (cuda_extra_devices.size() > 0) then
                    - This object will use multiple graphics cards to run the learning
                      algorithms.  In particular, it will always use whatever device is
                      currently selected on the calling thread (the device indicated by
                      cudaGetDevice()).  In addition, you can ask to use additional
                      devices, which you do by putting their device numbers into
                      cuda_extra_devices.
        !*/

       net_type& get_net (
        ) const; 
        /*!
            ensures
                - returns the neural network object used by this trainer.  This is the
                  network that is optimized when you call train() or train_one_step().
                  Recall that the dnn_trainer doesn't contain the net_type object but
                  simply holds a reference to an external network which was provided to the
                  dnn_trainer's constructor.
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

        const std::vector<solver_type>& get_solvers (
        ) const; 
        /*!
            ensures
                - returns the solvers used to optimize each layer of the neural network
                  get_net().  In particular, the first layer's solver is
                  get_solvers()[0], the second layer's solver is
                  get_solvers()[1], and so on.
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
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

        void set_learning_rate (
            double lr
        );
        /*!
            requires
                - lr > 0
            ensures
                - #get_learning_rate() == lr
                - #get_learning_rate_schedule().size() == 0
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

        double get_learning_rate(
        ) const;
        /*!
            ensures
                - During each training step, a solver tells us how to modify the parameters
                  of each layer in the network.  It does this by outputting a step vector
                  that, when added to the parameters, will hopefully result in improved
                  network performance.  The learning rate is one of the inputs to the
                  solver and influences the size of this step vector.  This function
                  returns the current learning rate, that is, the learning rate that will
                  be used during the next training step.
        !*/

        void set_min_learning_rate (
            double lr
        );
        /*!
            requires
                - lr > 0
            ensures
                - #get_min_learning_rate() == lr
                - #get_learning_rate_schedule().size() == 0
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

        double get_min_learning_rate (
        ) const;
        /*!
            ensures
                - During training via this->train(), this object will test if progress is
                  still being made and if it isn't then it will reduce get_learning_rate()
                  by setting it to get_learning_rate()*get_learning_rate_shrink_factor().
                  However, it will not reduce it below get_min_learning_rate().  Once this
                  minimum learning rate is crossed the training will terminate.
                - get_min_learning_rate() doesn't apply if you are using train_one_step().  
                  You can keep calling train_one_step() as many times as you want and the
                  learning rate will drop infinitely close to 0 if you run long enough.
        !*/

        template <typename EXP>
        void set_learning_rate_schedule (
            const matrix_exp<EXP>& schedule
        );
        /*!
            requires
                - schedule.size() > 0
                - min(schedule) > 0
            ensures
                - #get_learning_rate_schedule() == reshape_to_column_vector(schedule)
                - #get_learning_rate() == schedule(0,0)
                - #get_min_learning_rate() == min(schedule)
                - #set_learning_rate_shrink_factor() == 1
        !*/

        const matrix<double,0,1>& get_learning_rate_schedule (
        ) const;
        /*!
            ensures
                - if (this function returns a non-empty matrix) then
                    - This trainer will use an explicit learning rate schedule defined by
                      the learning rate values in get_learning_rate_schedule().  For
                      example, if get_learning_rate_schedule() returned {0.1, 0.09, 0.08,
                      0.07, 0.6} then the first training mini-batch would use a learning
                      rate of 0.1, then the next training mini-batch uses 0.09, and then
                      0.8, and so on until the end of the schedule is reached.  
                      
                      If you continue to run training after the end of the schedule has
                      been reached then the learning rate will be fixed to 0.99 times the
                      final value.  So in our example, eventually the learning rate would
                      be fixed to 0.99*0.6.  This allows you to test if we have reached the
                      end of the schedule by checking if get_learning_rate() >= 0.6.
        !*/

        unsigned long get_steps_without_progress (
        ) const;
        /*!
            ensures
                - if (get_learning_rate_shrink_factor() != 1) then
                    - returns an estimate of how many mini-batches have executed without us
                      observing a statistically significant decrease in the training error.
                - else
                    - returns 0
        !*/

        void set_iterations_without_progress_threshold (
            unsigned long thresh 
        );
        /*!
            ensures
                - #get_iterations_without_progress_threshold() == thresh
                - #get_learning_rate_schedule().size() == 0
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

        unsigned long get_iterations_without_progress_threshold (
        ) const;
        /*!
            ensures
                - This object monitors the progress of training and estimates if the
                  training error is being reduced.  It does this by looking at the previous
                  get_iterations_without_progress_threshold() mini-batch results and
                  applying the statistical test defined by the running_gradient object to
                  see if the training error is getting smaller.  If it isn't being reduced
                  then get_learning_rate() is made smaller by a factor of get_learning_rate_shrink_factor().

                  Therefore, get_iterations_without_progress_threshold() should always be
                  set to something sensibly large so that this test can be done with
                  reasonably high confidence.  Think of this test as saying "if the loss
                  hasn't decreased for the previous get_iterations_without_progress_threshold() 
                  then shrink the learning rate".
        !*/

        void set_learning_rate_shrink_factor (
            double shrink
        );
        /*!
            requires
                - 0 < shrink && shrink <= 1
            ensures
                - #get_learning_rate_shrink_factor() == shrink
                - #get_learning_rate_schedule().size() == 0
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

        double get_learning_rate_shrink_factor (
        ) const;
        /*!
            ensures
                - Whenever the training routine thinks it isn't making progress anymore it
                  will reduce get_learning_rate() by multiplying it by get_learning_rate_shrink_factor().
                - You can disable the automatic learning rate reduction by setting
                  get_learning_rate_shrink_factor() to 1.
        !*/

        unsigned long long get_train_one_step_calls (
        ) const;
        /*!
            ensures
                - returns the number of times train_one_step() has been called.
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

        void set_synchronization_file (
            const std::string& filename,
            std::chrono::seconds time_between_syncs = std::chrono::minutes(15)
        );
        /*!
            ensures
                - While training is running, either via train() or repeated calls to
                  train_one_step(), this object will save its entire state, including the
                  state of get_net(), to disk in the file named filename every
                  time_between_syncs seconds.
                - if the filename file already exists then the state of this trainer will
                  be loaded from that file by this call to set_synchronization_file().
                  This allows you to resume a training session which was previously
                  interrupted.
        !*/

        void train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ); 
        /*!
            requires
                - data.size() == labels.size()
                - data.size() > 0
                - net_type uses a supervised loss.  
                  i.e. net_type::label_type != no_label_type.
            ensures
                - Trains a supervised neural network based on the given training data.
                  The goal of training is to find the network parameters that minimize
                  get_net().compute_loss(data.begin(), data.end(), labels.begin()). 
                - The optimizer will run until get_learning_rate() < get_min_learning_rate() 
                  or get_max_num_epochs() training epochs have been executed. 
                - Each layer in the network will be optimized by its corresponding solver
                  in get_solvers().  
                - Each call to train DOES NOT reinitialize the state of get_net() or
                  get_solvers().  That is, the existing state of the solvers and network is
                  the starting point for the optimization each time train() is called.  In
                  particular, if you use the set_synchronization_file() method you can
                  resume an interrupted train() call by simply calling train() again and it
                  will pick up from the last synchronization point.  
                - You can obtain the average loss value during the final training epoch by
                  calling get_average_loss().
        !*/

        void train (
            const std::vector<input_type>& data
        );
        /*!
            requires 
                - data.size() > 0
                - net_type uses an unsupervised loss.  
                  i.e. net_type::label_type == no_label_type.
            ensures
                - Trains an unsupervised neural network based on the given training data.
                  The goal of training is to find the network parameters that minimize
                  get_net().compute_loss(data.begin(), data.end()). 
                - The optimizer will run until get_learning_rate() < get_min_learning_rate() 
                  or get_max_num_epochs() training epochs have been executed. 
                - Each layer in the network will be optimized by its corresponding solver
                  in get_solvers().  
                - Each call to train DOES NOT reinitialize the state of get_net() or
                  get_solvers().  That is, the existing state of the solvers and network is
                  the starting point for the optimization each time train() is called.  In
                  particular, if you use the set_synchronization_file() method you can
                  resume an interrupted train() call by simply calling train() again and it
                  will pick up from the last synchronization point.  
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
                - data.size() > 0
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
                - The network training will happen in another thread.  Therefore, after
                  calling this function you should call get_net() before you touch the net
                  object from the calling thread to ensure no other threads are still
                  accessing the network.
                - #get_train_one_step_calls() == get_train_one_step_calls() + 1.
        !*/

        void train_one_step (
            const std::vector<input_type>& data
        );
        /*!
            requires
                - data.size() > 0
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
                - The network training will happen in another thread.  Therefore, after
                  calling this function you should call get_net() before you touch the net
                  object from the calling thread to ensure no other threads are still
                  accessing the network.
                - #get_train_one_step_calls() == get_train_one_step_calls() + 1.
        !*/

        double get_average_loss (
        ) const;
        /*!
            ensures
                - returns the average loss value observed during previous calls to
                  train_one_step() or train().  That is, the average output of
                  net_type::update() during the previous mini-batch updates.
                - Note that, if be_verbose() has been called, then this object will
                  automatically call clear_average_loss() periodically when it logs the
                  loss to the console.
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
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
                - This function blocks until all threads inside the dnn_trainer have
                  stopped touching the net. 
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_ABSTRACT_H_ 


