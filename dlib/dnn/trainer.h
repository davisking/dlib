// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TRAINER_H_
#define DLIB_DNn_TRAINER_H_

#include "trainer_abstract.h"
#include "core.h"
#include "solvers.h"
#include "../statistics.h"
#include "../console_progress_indicator.h"
#include <chrono>
#include "../serialize.h"

#include "../pipe.h"
#include "../threads.h"
#include "cuda_dlib.h"

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
        ) : job_pipe(0)
        {
            init();
        }

        explicit dnn_trainer(const net_type& net_) : job_pipe(0), net(net_)
        {
            init();
        }

        dnn_trainer(
            const net_type& net_, 
            const solver_type& solver_
        ) : job_pipe(0), net(net_), solvers(solver_) 
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


        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const 
        { 
            wait_for_thread_to_pause();
            return solvers; 
        }

        sstack<solver_type,net_type::num_layers>& get_solvers (
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

            console_progress_indicator pbar(num_epochs);
            pbar.print_status(0);
            for (unsigned long epoch_iteration = 0; epoch_iteration < num_epochs; ++epoch_iteration)
            {
                using namespace std::chrono;
                auto last_time = system_clock::now();
                clear_average_loss();
                for (size_t i = 0; i < data.size(); i += mini_batch_size)
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
                            std::cout << "epoch: " << rpad(cast_to_string(iter),string_pad) << " " 
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                            pbar.print_status(iter, true);
                            std::cout << std::endl;
                        }
                    }
                }

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),string_pad) << " " 
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                    pbar.print_status(epoch_iteration+1, true);
                    std::cout << std::endl;
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

            console_progress_indicator pbar(num_epochs);
            pbar.print_status(0);
            for (unsigned long epoch_iteration = 0; epoch_iteration < num_epochs; ++epoch_iteration)
            {
                using namespace std::chrono;
                auto last_time = system_clock::now();
                clear_average_loss();
                for (size_t i = 0; i < data.size(); i += mini_batch_size)
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
                            std::cout << "epoch: " << rpad(cast_to_string(iter),string_pad) << " " 
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                            pbar.print_status(iter, true);
                            std::cout << std::endl;
                        }
                    }
                }

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),string_pad) << " " 
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                    pbar.print_status(epoch_iteration+1, true);
                    std::cout << std::endl;
                }
            }
            return get_net();
        }

        friend void serialize(const dnn_trainer& item, std::ostream& out)
        {
            item.wait_for_thread_to_pause();
            int version = 1;
            serialize(version, out);
            serialize(item.rs, out);
            serialize(item.num_epochs, out);
            serialize(item.mini_batch_size, out);
            serialize(item.verbose, out);
            serialize(item.net, out);
            serialize(item.solvers, out);
        }

        friend void deserialize(dnn_trainer& item, std::istream& in)
        {
            item.wait_for_thread_to_pause();
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::dnn_trainer.");
            deserialize(item.rs, in);
            deserialize(item.num_epochs, in);
            deserialize(item.mini_batch_size, in);
            deserialize(item.verbose, in);
            deserialize(item.net, in);
            deserialize(item.solvers, in);
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

    private:
        struct job_t
        {
            std::vector<label_type> labels;
            resizable_tensor t;
        };

        template <typename T>
        void run_update(job_t& next_job, const T&)
        {
            rs.add(net.update(next_job.t, next_job.labels.begin(), solvers));
        }

        void run_update(job_t& next_job, const no_label_type&)
        {
            no_label_type pick_wich_run_update;
            rs.add(net.update(next_job.t, solvers));
        }

        void thread()
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
            }
        }

        void wait_for_thread_to_pause() const
        {
            job_pipe.wait_for_num_blocked_dequeues(1);
        }

        const static long string_pad = 10;

        void init()
        {
            num_epochs = 300;
            mini_batch_size = 32;
            verbose = false;
            cuda_device_id = dlib::cuda::get_device();
            start();
        }

        // The job object is not logically part of the state of this object. It is here
        // only to avoid reallocating it over and over.
        job_t job;

        dlib::pipe<job_t> job_pipe;
        running_stats<double> rs;
        unsigned long num_epochs;
        size_t mini_batch_size;
        bool verbose;
        int cuda_device_id;

        net_type net;
        sstack<solver_type,net_type::num_layers> solvers;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_H_

