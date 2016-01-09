// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TRAINER_H_
#define DLIB_DNn_TRAINER_H_

#include "trainer_abstract.h"
#include "core.h"
#include "solvers.h"
#include "../statistics.h"
#include <chrono>
#include "../serialize.h"

#include "../pipe.h"
#include "../threads.h"
#include "cuda_dlib.h"
#include "../statistics/running_gradient.h"
#include <atomic>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer : private threaded_object
    {
    public:

        static_assert(is_loss_layer_type<net_type>::value, 
            "The last layer in a network must be a loss layer.");

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer(
        ) : job_pipe(0), solvers(net_type::num_layers)
        {
            init();
        }

        explicit dnn_trainer(const net_type& net_) : job_pipe(0), net(net_), solvers(net_type::num_layers)
        {
            init();
        }

        dnn_trainer(
            const net_type& net_, 
            const solver_type& solver_
        ) : job_pipe(0), net(net_), solvers(net_type::num_layers, solver_) 
        {
            init();
        }

        ~dnn_trainer(
        )
        {
            job_pipe.disable();
            stop();
            wait();
        }

        const net_type& get_net (
        ) const 
        { 
            wait_for_thread_to_pause();
            return net; 
        }

        void set_net (
            const net_type& net_
        ) 
        { 
            wait_for_thread_to_pause();
            return net = net_; 
        }

        void set_solver (
            const solver_type& solver_
        ) 
        { 
            wait_for_thread_to_pause();
            solvers = std::vector<solver_type>(net_type::num_layers, solver_); 
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

        unsigned long get_max_num_epochs (
        ) const { return max_num_epochs; }

        void set_max_num_epochs (
            unsigned long num
        )  
        {
            DLIB_CASSERT(num > 0,"");
            max_num_epochs = num;
        }

        void be_verbose (
        )
        {
            verbose = true;
        }

        void be_quiet (
        )
        {
            verbose = false;
        }


        const std::vector<solver_type>& get_solvers (
        ) const 
        { 
            wait_for_thread_to_pause();
            return solvers; 
        }

        std::vector<solver_type>& get_solvers (
        ) 
        { 
            wait_for_thread_to_pause();
            return solvers; 
        }


        void train_one_step (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        )
        {
            job.labels = labels;
            net.to_tensor(data.begin(), data.end(), job.t);
            job_pipe.enqueue(job);
        }

        void train_one_step (
            const std::vector<input_type>& data
        )
        {
            net.to_tensor(data.begin(), data.end(), job.t);
            job_pipe.enqueue(job);
        }

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ) 
        {
            DLIB_CASSERT(data.size() == labels.size() && data.size() > 0, "");

            for (unsigned long epoch_iteration = 0; 
                epoch_iteration < max_num_epochs && step_size >= min_step_size; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                auto last_time = system_clock::now();
                clear_average_loss();
                for (size_t i = 0; i < data.size() && step_size >= min_step_size; i += mini_batch_size)
                {
                    net.to_tensor(data.begin()+i, 
                                  data.begin()+std::min(i+mini_batch_size,data.size()), 
                                  job.t);
                    job.labels.assign(labels.begin()+i,
                                      labels.begin()+std::min(i+mini_batch_size,data.size()));
                    job_pipe.enqueue(job);


                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + i/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) 
                                << std::endl;
                        }
                    }
                }

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) 
                              << std::endl;
                }
            }
            return get_net();
        }

        const net_type& train (
            const std::vector<input_type>& data
        ) 
        {
            DLIB_CASSERT(data.size() > 0, "");

            const bool has_unsupervised_loss = std::is_same<no_label_type, label_type>::value; 
            static_assert(has_unsupervised_loss, 
                "You can only call this version of train() when using an unsupervised loss.");

            for (unsigned long epoch_iteration = 0; 
                epoch_iteration < max_num_epochs && step_size >= min_step_size; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                auto last_time = system_clock::now();
                clear_average_loss();
                for (size_t i = 0; i < data.size() && step_size >= min_step_size; i += mini_batch_size)
                {
                    net.to_tensor(data.begin()+i, 
                                  data.begin()+std::min(i+mini_batch_size,data.size()), 
                                  job.t);
                    job_pipe.enqueue(job);


                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + i/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) 
                                << std::endl;
                        }
                    }
                }

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) 
                              << std::endl;
                }
            }
            return get_net();
        }

        friend void serialize(const dnn_trainer& item, std::ostream& out)
        {
            item.wait_for_thread_to_pause();
            int version = 3;
            serialize(version, out);

            serialize(item.rs, out);
            serialize(item.rg, out);
            serialize(item.max_num_epochs, out);
            serialize(item.mini_batch_size, out);
            serialize(item.verbose, out);
            serialize(item.net, out);
            serialize(item.solvers, out);
            serialize(item.step_size.load(), out);
            serialize(item.min_step_size, out);
            serialize(item.iter_between_step_size_adjust.load(), out);
            serialize(item.step_size_shrink.load(), out);
        }

        friend void deserialize(dnn_trainer& item, std::istream& in)
        {
            item.wait_for_thread_to_pause();
            int version = 0;
            deserialize(version, in);
            if (version != 3)
                throw serialization_error("Unexpected version found while deserializing dlib::dnn_trainer.");

            double temp;
            deserialize(item.rs, in);
            deserialize(item.rg, in);
            deserialize(item.max_num_epochs, in);
            deserialize(item.mini_batch_size, in);
            deserialize(item.verbose, in);
            deserialize(item.net, in);
            deserialize(item.solvers, in);
            deserialize(temp, in); item.step_size = temp;
            deserialize(item.min_step_size, in);
            deserialize(temp, in); item.iter_between_step_size_adjust = temp;
            deserialize(temp, in); item.step_size_shrink = temp;
        }

        double get_average_loss (
        ) const 
        { 
            wait_for_thread_to_pause();
            return rs.mean();
        }

        void clear_average_loss (
        )
        {
            wait_for_thread_to_pause();
            rs.clear();
        }

        void set_setep_size (
            double ss
        )
        {
            DLIB_CASSERT(ss > 0,"");
            wait_for_thread_to_pause();
            step_size = ss;
        }

        double get_step_size(
        ) const 
        {
            return step_size;
        }

        void set_min_step_size (
            double ss
        )
        {
            DLIB_CASSERT(ss > 0,"");
            min_step_size = ss;
        }

        double get_min_step_size (
        ) const
        {
            return min_step_size;
        }

        void set_iterations_between_step_size_adjust (
            unsigned long min_iter
        )
        {
            iter_between_step_size_adjust = min_iter;
        }

        unsigned long get_iterations_between_step_size_adjust (
        ) const
        {
            return iter_between_step_size_adjust;
        }

        void set_step_size_shrink_amount (
            double shrink
        )
        {
            DLIB_CASSERT(0 < shrink && shrink <= 1,"");
            step_size_shrink = shrink;
        }

        double get_step_size_shrink (
        ) const
        {
            return step_size_shrink;
        }

    private:
        struct job_t
        {
            std::vector<label_type> labels;
            resizable_tensor t;
        };

        template <typename T>
        void run_update(job_t& next_job, const T&)
        {
            double loss = net.update(next_job.t, next_job.labels.begin(), make_sstack(solvers),step_size);
            rs.add(loss);
            rg.add(loss);
        }

        void run_update(job_t& next_job, const no_label_type&)
        {
            no_label_type pick_wich_run_update;
            double loss = net.update(next_job.t, make_sstack(solvers), step_size);
            rs.add(loss);
            rg.add(loss);
        }

        void thread() try
        {
            // Make sure this thread uses the same cuda device as the thread that created
            // the dnn_trainer object.
            dlib::cuda::set_device(cuda_device_id);
            label_type pick_wich_run_update;
            job_t next_job;
            while(job_pipe.dequeue(next_job))
            {
                // call net.update() but pick the right version for unsupervised or
                // supervised training based on the type of label_type.
                run_update(next_job, pick_wich_run_update);

                // If we have been running for a while then check if the loss is still
                // dropping.  If it isn't then we will reduce the step size.
                if (rg.current_n() > iter_between_step_size_adjust)
                {
                    if (rg.probability_gradient_greater_than(0) > 0.45)
                    {
                        step_size = step_size_shrink*step_size;
                    }
                    rg.clear();
                }
            }
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }

        void wait_for_thread_to_pause() const
        {
            job_pipe.wait_for_num_blocked_dequeues(1);
        }

        const static long string_pad = 10;
        const static long epoch_string_pad = 4;
        const static long ss_string_pad = 4;

        void init()
        {
            max_num_epochs = 10000;
            mini_batch_size = 128;
            verbose = false;
            cuda_device_id = dlib::cuda::get_device();
            step_size = 1;
            min_step_size = 1e-4;
            iter_between_step_size_adjust = 2000;
            step_size_shrink = 0.1;
            start();
        }


        dlib::pipe<job_t> job_pipe;
        running_stats<double> rs;
        running_gradient rg;
        unsigned long max_num_epochs;
        size_t mini_batch_size;
        bool verbose;
        int cuda_device_id;
        net_type net;
        std::vector<solver_type> solvers;
        std::atomic<double> step_size;
        double min_step_size;
        std::atomic<long> iter_between_step_size_adjust;
        std::atomic<double> step_size_shrink;

        // The job object is not logically part of the state of this object. It is here
        // only to avoid reallocating it over and over.
        job_t job;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_H_

