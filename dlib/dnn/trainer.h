// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TRAINER_H_
#define DLIB_DNn_TRAINER_H_

#include "trainer_abstract.h"
#include "core.h"
#include "solvers.h"
#include "../statistics.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "../serialize.h"

#include "../pipe.h"
#include "../threads.h"
#include "cuda_dlib.h"
#include "../statistics/running_gradient.h"
#include <atomic>
#include <cstdio>

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
        const static size_t num_layers = net_type::num_layers;

        dnn_trainer() = delete;
        dnn_trainer(const dnn_trainer&) = delete;

        explicit dnn_trainer(net_type& net_) : job_pipe(0), net(net_), solvers(num_layers)
        {
            init();
        }

        dnn_trainer(
            net_type& net_, 
            const solver_type& solver_
        ) : job_pipe(0), net(net_), solvers(num_layers, solver_) 
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

        net_type& get_net (
        ) const 
        { 
            wait_for_thread_to_pause();
            return net; 
        }

        void set_solver (
            const solver_type& solver_
        ) 
        { 
            wait_for_thread_to_pause();
            solvers = std::vector<solver_type>(num_layers, solver_); 
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
            if (verbose)
            {
                using namespace std::chrono;
                auto now_time = system_clock::now();
                if (now_time-last_time > seconds(40))
                {
                    last_time = now_time;
                    std::cout << "step#: " << rpad(cast_to_string(train_one_step_calls),epoch_string_pad) << "  " 
                        << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                        << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad)  << "  "
                        << "steps without apparent progress: " << steps_without_progress 
                        << std::endl;
                    clear_average_loss();
                }
            }
            sync_to_disk();
            job.labels = labels;
            net.to_tensor(data.begin(), data.end(), job.t);
            job_pipe.enqueue(job);
            ++train_one_step_calls;
        }

        void train_one_step (
            const std::vector<input_type>& data
        )
        {
            if (verbose)
            {
                using namespace std::chrono;
                auto now_time = system_clock::now();
                if (now_time-last_time > seconds(40))
                {
                    last_time = now_time;
                    std::cout << "step#: " << rpad(cast_to_string(train_one_step_calls),epoch_string_pad) << "  " 
                        << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                        << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  "
                        << "steps without apparent progress: " << steps_without_progress 
                        << std::endl;
                    clear_average_loss();
                }
            }
            sync_to_disk();
            net.to_tensor(data.begin(), data.end(), job.t);
            job_pipe.enqueue(job);
            ++train_one_step_calls;
        }

        void train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ) 
        {
            DLIB_CASSERT(data.size() == labels.size() && data.size() > 0, "");

            bool updated_the_network = false;
            // The reason these two loops don't initialize their counter variables but
            // instead use class members is so we can include the state of the loops in the
            // stuff written by sync_to_disk()
            for (; 
                epoch_iteration < max_num_epochs && step_size >= min_step_size; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                last_time = system_clock::now();
                clear_average_loss();
                for (; epoch_pos < data.size() && step_size >= min_step_size; epoch_pos += mini_batch_size)
                {
                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + epoch_pos/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  "
                                << "steps without apparent progress: " << steps_without_progress 
                                << std::endl;
                        }
                    }

                    sync_to_disk();
                    net.to_tensor(data.begin()+epoch_pos, 
                                  data.begin()+std::min(epoch_pos+mini_batch_size,data.size()), 
                                  job.t);
                    job.labels.assign(labels.begin()+epoch_pos,
                                      labels.begin()+std::min(epoch_pos+mini_batch_size,data.size()));
                    job_pipe.enqueue(job);
                    updated_the_network = true;
                }
                epoch_pos = 0;

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  "
                              << "steps without apparent progress: " << steps_without_progress 
                              << std::endl;
                }
            }
            wait_for_thread_to_pause();
            // if we modified the network at all then be sure to sync the final result.
            sync_to_disk(updated_the_network);
        }

        void train (
            const std::vector<input_type>& data
        ) 
        {
            DLIB_CASSERT(data.size() > 0, "");

            const bool has_unsupervised_loss = std::is_same<no_label_type, label_type>::value; 
            static_assert(has_unsupervised_loss, 
                "You can only call this version of train() when using an unsupervised loss.");

            bool updated_the_network = false;
            // The reason these two loops don't initialize their counter variables but
            // instead use class members is so we can include the state of the loops in the
            // stuff written by sync_to_disk()
            for (; 
                epoch_iteration < max_num_epochs && step_size >= min_step_size; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                last_time = system_clock::now();
                clear_average_loss();
                for (; epoch_pos < data.size() && step_size >= min_step_size; epoch_pos += mini_batch_size)
                {
                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + epoch_pos/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                                << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  "
                                << "steps without apparent progress: " << steps_without_progress 
                                << std::endl;
                        }
                    }

                    sync_to_disk();
                    net.to_tensor(data.begin()+epoch_pos, 
                                  data.begin()+std::min(epoch_pos+mini_batch_size,data.size()), 
                                  job.t);
                    job_pipe.enqueue(job);
                    updated_the_network = true;
                }
                epoch_pos = 0;

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "step size: " << rpad(cast_to_string(step_size),ss_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  "
                              << "steps without apparent progress: " << steps_without_progress 
                              << std::endl;
                }
            }
            wait_for_thread_to_pause();
            // if we modified the network at all then be sure to sync the final result.
            sync_to_disk(updated_the_network);
        }

        void set_synchronization_file (
            const std::string& filename,
            std::chrono::seconds time_between_syncs_ = std::chrono::minutes(15)
        )
        {
            last_sync_time = std::chrono::system_clock::now();
            sync_filename = filename;
            time_between_syncs = time_between_syncs_;

            // check if the sync file already exists, if it does we should load it.
            std::ifstream fin(filename, std::ios::binary);
            if (fin)
                deserialize(*this, fin);
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

        void set_step_size (
            double ss
        )
        {
            DLIB_CASSERT(ss > 0,"");
            wait_for_thread_to_pause();
            if (step_size != ss)
                previous_loss_values.clear();
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

        void set_iterations_without_progress_threshold (
            unsigned long thresh 
        )
        {
            iter_without_progress_thresh = thresh;
        }

        unsigned long get_iterations_without_progress_threshold (
        ) const
        {
            return iter_without_progress_thresh;
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

        void record_loss(double loss)
        {
            // Say that we will check if the gradient is bad 200 times during each
            // iter_without_progress_thresh interval of network updates.   This kind of
            // budgeting causes our gradient checking to use a fixed amount of
            // computational resources, regardless of the size of
            // iter_without_progress_thresh.
            gradient_check_budget += 200;

            rs.add(loss);
            previous_loss_values.push_back(loss);
            // discard really old loss values.
            while (previous_loss_values.size() > iter_without_progress_thresh)
                previous_loss_values.pop_front();
        }

        template <typename T>
        void run_update(job_t& next_job, const T&)
        {
            double loss = net.update(next_job.t, next_job.labels.begin(), make_sstack(solvers),step_size);
            record_loss(loss);
        }

        void run_update(job_t& next_job, const no_label_type&)
        {
            no_label_type pick_which_run_update;
            double loss = net.update(next_job.t, make_sstack(solvers), step_size);
            record_loss(loss);
        }

        void thread() try
        {
            // Make sure this thread uses the same cuda device as the thread that created
            // the dnn_trainer object.
            dlib::cuda::set_device(cuda_device_id);
            label_type pick_which_run_update;
            job_t next_job;
            while(job_pipe.dequeue(next_job))
            {
                // call net.update() but pick the right version for unsupervised or
                // supervised training based on the type of label_type.
                run_update(next_job, pick_which_run_update);

                // If we have been running for a while then check if the loss is still
                // dropping.  If it isn't then we will reduce the step size.  Note that we
                // have a "budget" that prevents us from calling
                // count_steps_without_decrease() every iteration.  We do this because
                // it can be expensive to compute when previous_loss_values is large.
                if (gradient_check_budget > iter_without_progress_thresh)
                {
                    gradient_check_budget = 0;
                    steps_without_progress = count_steps_without_decrease(previous_loss_values);
                    if (steps_without_progress >= iter_without_progress_thresh)
                    {
                        // optimization has flattened out, so drop the learning rate. 
                        step_size = step_size_shrink*step_size;
                        steps_without_progress = 0;
                        previous_loss_values.clear();
                    }
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
            min_step_size = 1e-3;
            iter_without_progress_thresh = 2000;
            steps_without_progress = 0;
            step_size_shrink = 0.1;
            epoch_iteration = 0;
            epoch_pos = 0;
            train_one_step_calls = 0;
            gradient_check_budget = 0;
            start();
        }

        // serialize and deserialize are private because we hold net by reference so
        // allowing someone to serialize this training object is weird and will likely
        // result in user errors.  However, we use these functions as part of the automatic
        // sync code in this object.
        friend void serialize(const dnn_trainer& item, std::ostream& out)
        {
            item.wait_for_thread_to_pause();
            int version = 5;
            serialize(version, out);

            size_t nl = dnn_trainer::num_layers;
            serialize(nl, out);
            serialize(item.rs, out);
            serialize(item.previous_loss_values, out);
            serialize(item.max_num_epochs, out);
            serialize(item.mini_batch_size, out);
            serialize(item.verbose, out);
            serialize(item.net, out);
            serialize(item.solvers, out);
            serialize(item.step_size.load(), out);
            serialize(item.min_step_size, out);
            serialize(item.iter_without_progress_thresh.load(), out);
            serialize(item.steps_without_progress.load(), out);
            serialize(item.step_size_shrink.load(), out);
            serialize(item.epoch_iteration, out);
            serialize(item.epoch_pos, out);
            serialize(item.train_one_step_calls, out);
        }
        friend void deserialize(dnn_trainer& item, std::istream& in)
        {
            item.wait_for_thread_to_pause();
            int version = 0;
            deserialize(version, in);
            if (version != 5)
                throw serialization_error("Unexpected version found while deserializing dlib::dnn_trainer.");

            size_t num_layers = 0;
            deserialize(num_layers, in);
            if (num_layers != dnn_trainer::num_layers)
            {
                std::ostringstream sout;
                sout << "Error deserializing dlib::dnn_trainer.  The saved sync file is for a network with " << std::endl;
                sout << "a different number of layers.  We expected the number of layers to be " << dnn_trainer::num_layers << " but" << std::endl;
                sout << "instead the file contains " << num_layers << " layers." << std::endl;
                throw serialization_error(sout.str());
            }

            double dtemp; long ltemp;
            deserialize(item.rs, in);
            deserialize(item.previous_loss_values, in);
            deserialize(item.max_num_epochs, in);
            deserialize(item.mini_batch_size, in);
            deserialize(item.verbose, in);
            deserialize(item.net, in);
            deserialize(item.solvers, in);
            deserialize(dtemp, in); item.step_size = dtemp;
            deserialize(item.min_step_size, in);
            deserialize(ltemp, in); item.iter_without_progress_thresh = ltemp;
            deserialize(ltemp, in); item.steps_without_progress = ltemp;
            deserialize(dtemp, in); item.step_size_shrink = dtemp;
            deserialize(item.epoch_iteration, in);
            deserialize(item.epoch_pos, in);
            deserialize(item.train_one_step_calls, in);
        }
        void sync_to_disk (
            bool do_it_now = false
        )
        {
            // If the sync file isn't set then don't do anything.
            if (sync_filename.size() == 0)
                return;

            // Only sync if it has been long enough since the last sync or we are being
            // explicitly forced to do it.
            if (std::chrono::system_clock::now() - last_sync_time > time_between_syncs ||
                do_it_now)
            {
                // save our state to a temp file
                std::string tempfile = sync_filename + ".tmp";
                std::ofstream fout(tempfile, std::ios::binary);
                // compact network before saving to disk.
                wait_for_thread_to_pause();
                this->net.clean(); 
                serialize(*this, fout);
                fout.close();

                // Now that we know the state is safely saved to disk, delete the old sync
                // file and move the .tmp file to it.
                std::remove(sync_filename.c_str());
                std::rename(tempfile.c_str(), sync_filename.c_str());

                last_sync_time = std::chrono::system_clock::now();
                if (verbose)
                    std::cout << "Saved state to " << sync_filename << std::endl;
            }
        }



        dlib::pipe<job_t> job_pipe;
        running_stats<double> rs;
        std::deque<double> previous_loss_values;
        unsigned long max_num_epochs;
        size_t mini_batch_size;
        bool verbose;
        int cuda_device_id;
        net_type& net;
        std::vector<solver_type> solvers;
        std::atomic<double> step_size;
        double min_step_size;
        std::atomic<unsigned long> iter_without_progress_thresh;
        std::atomic<unsigned long> steps_without_progress;
        std::atomic<double> step_size_shrink;
        std::chrono::time_point<std::chrono::system_clock> last_sync_time;
        std::string sync_filename;
        std::chrono::seconds time_between_syncs;
        unsigned long epoch_iteration;
        unsigned long epoch_pos;
        std::chrono::time_point<std::chrono::system_clock> last_time;
        unsigned long long train_one_step_calls;
        unsigned long gradient_check_budget;

        // The job object is not logically part of the state of this object. It is here
        // only to avoid reallocating it over and over.
        job_t job;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_H_

