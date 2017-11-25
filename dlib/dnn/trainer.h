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
#include <set>
#include <future>
#include <exception>
#include <mutex>
#include "../dir_nav.h"
#include "../md5.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename training_label_type>
        struct dnn_job_t
        {
            dnn_job_t() = default;
            dnn_job_t(const dnn_job_t&) = delete;
            dnn_job_t& operator=(const dnn_job_t&) = delete;

            std::vector<std::vector<training_label_type>> labels;
            std::vector<resizable_tensor> t;
            std::vector<int> have_data;  // have_data[i] is true if there is data in labels[i] and t[i].
            bool test_only = false;
        };

        template <typename training_label_type>
        void swap(dnn_job_t<training_label_type>& a, dnn_job_t<training_label_type>& b)
        {
            a.labels.swap(b.labels);
            a.t.swap(b.t);
            a.have_data.swap(b.have_data);
            std::swap(a.test_only,b.test_only);
        }
    }

    enum class force_flush_to_disk {
        no = 0,
        yes = 1
    };

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer : private threaded_object
    {
    public:

        static_assert(is_loss_layer_type<net_type>::value, 
            "The last layer in a network must be a loss layer.");

        typedef typename net_type::training_label_type training_label_type;
        typedef typename net_type::input_type input_type;
        const static size_t num_computational_layers = net_type::num_computational_layers;
        const static size_t num_layers = net_type::num_layers;
    private:
        typedef impl::dnn_job_t<training_label_type> job_t;
    public:

        dnn_trainer() = delete;
        dnn_trainer(const dnn_trainer&) = delete;
        dnn_trainer& operator=(const dnn_trainer&) = delete;

        explicit dnn_trainer(net_type& net_) : job_pipe(0), net(net_)
        {
            solver_type default_solver;
            devices.push_back(std::make_shared<device_data>(dlib::cuda::get_device(), net, default_solver));

            init();
        }

        dnn_trainer(
            net_type& net_, 
            const solver_type& solver_
        ) : job_pipe(0), net(net_) 
        {
            devices.push_back(std::make_shared<device_data>(dlib::cuda::get_device(), net, solver_));

            init();
        }

        dnn_trainer(
            net_type& net_, 
            const solver_type& solver_,
            const std::vector<int>& cuda_extra_devices
        ) : job_pipe(0), net(net_) 
        {
            devices.push_back(std::make_shared<device_data>(dlib::cuda::get_device(), net, solver_));

            const int total_devices = dlib::cuda::get_num_devices();

            // Make device contexts for the extra device ids but be careful to avoid any
            // duplicate ids.
            std::set<int> temp(cuda_extra_devices.begin(), cuda_extra_devices.end());
            temp.erase(devices[0]->device_id);
            for (auto id : temp)
            {
                DLIB_CASSERT(0 <= id && id < total_devices, "Invalid CUDA device id given to dnn_trainer.");
                // Switch to this device so that any tensor objects that get allocated when
                // we create the device context happen on this device.
                dlib::cuda::set_device(id);
                devices.push_back(std::make_shared<device_data>(id, net, solver_, clone_net()));
            }
            // Set the current device back to what it was before this constructor was
            // called.
            dlib::cuda::set_device(devices[0]->device_id);

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
            force_flush_to_disk force_flush = force_flush_to_disk::yes
        )  
        { 
            wait_for_thread_to_pause();
            sync_to_disk(force_flush == force_flush_to_disk::yes);
            propagate_exception();
            return net; 
        }


        unsigned long get_mini_batch_size (
        ) const { return mini_batch_size; }

        void set_mini_batch_size (
            unsigned long batch_size 
        )
        {
            DLIB_CASSERT(batch_size > 0);
            mini_batch_size = batch_size;
        }

        unsigned long get_max_num_epochs (
        ) const { return max_num_epochs; }

        void set_max_num_epochs (
            unsigned long num
        )  
        {
            DLIB_CASSERT(num > 0);
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
            propagate_exception();
            return devices[0]->solvers; 
        }

        void train_one_step (
            const std::vector<input_type>& data,
            const std::vector<training_label_type>& labels 
        )
        {
            DLIB_CASSERT(data.size() == labels.size());

            train_one_step(data.begin(), data.end(), labels.begin());
        }

        template <
            typename data_iterator,
            typename label_iterator
            >
        void train_one_step (
            data_iterator dbegin, 
            data_iterator dend,
            label_iterator lbegin
        )
        {
            DLIB_CASSERT(std::distance(dbegin, dend) > 0);

            print_periodic_verbose_status();
            sync_to_disk();
            send_job(false, dbegin, dend, lbegin);

            ++train_one_step_calls;
        }

        void train_one_step (
            const std::vector<input_type>& data
        )
        {
            train_one_step(data.begin(), data.end());
        }

        template <
            typename data_iterator
            >
        void train_one_step (
            data_iterator dbegin, 
            data_iterator dend
        )
        {
            DLIB_CASSERT(std::distance(dbegin, dend) > 0);
            print_periodic_verbose_status();
            sync_to_disk();
            send_job(false, dbegin, dend);
            ++train_one_step_calls;
        }

        void test_one_step (
            const std::vector<input_type>& data,
            const std::vector<training_label_type>& labels 
        )
        {
            DLIB_CASSERT(data.size() == labels.size());

            test_one_step(data.begin(), data.end(), labels.begin());
        }

        template <
            typename data_iterator,
            typename label_iterator
            >
        void test_one_step (
            data_iterator dbegin, 
            data_iterator dend,
            label_iterator lbegin
        )
        {
            DLIB_CASSERT(std::distance(dbegin, dend) > 0);

            print_periodic_verbose_status();
            sync_to_disk();
            send_job(true, dbegin, dend, lbegin);

            ++test_one_step_calls;
        }

        void test_one_step (
            const std::vector<input_type>& data
        )
        {
            test_one_step(data.begin(), data.end());
        }

        template <
            typename data_iterator
            >
        void test_one_step (
            data_iterator dbegin, 
            data_iterator dend
        )
        {
            DLIB_CASSERT(std::distance(dbegin, dend) > 0);
            print_periodic_verbose_status();
            sync_to_disk();
            send_job(true, dbegin, dend);
            ++test_one_step_calls;
        }

        void train (
            const std::vector<input_type>& data,
            const std::vector<training_label_type>& labels 
        ) 
        {
            DLIB_CASSERT(data.size() == labels.size() && data.size() > 0);

            // The reason these two loops don't initialize their counter variables but
            // instead use class members is so we can include the state of the loops in the
            // stuff written by sync_to_disk()
            for (; 
                epoch_iteration < max_num_epochs && learning_rate >= min_learning_rate; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                last_time = system_clock::now();
                clear_average_loss();
                for (; epoch_pos < data.size() && learning_rate >= min_learning_rate; epoch_pos += mini_batch_size)
                {
                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + epoch_pos/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                      << "learning rate: " << rpad(cast_to_string(learning_rate),lr_string_pad) << "  "
                                      << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                            print_progress();
                        }
                    }

                    sync_to_disk();
                    send_job(false, data.begin()+epoch_pos, 
                              data.begin()+std::min(epoch_pos+mini_batch_size,data.size()), 
                              labels.begin()+epoch_pos);
                }
                epoch_pos = 0;

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "learning rate: " << rpad(cast_to_string(learning_rate),lr_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                    print_progress();
                }
            }
            wait_for_thread_to_pause();
            // if we modified the network at all then be sure to sync the final result.
            sync_to_disk(true);
        }

        void train (
            const std::vector<input_type>& data
        ) 
        {
            DLIB_CASSERT(data.size() > 0);

            const bool has_unsupervised_loss = std::is_same<no_label_type, training_label_type>::value; 
            static_assert(has_unsupervised_loss, 
                "You can only call this version of train() when using an unsupervised loss.");

            // The reason these two loops don't initialize their counter variables but
            // instead use class members is so we can include the state of the loops in the
            // stuff written by sync_to_disk()
            for (; 
                epoch_iteration < max_num_epochs && learning_rate >= min_learning_rate; 
                ++epoch_iteration)
            {
                using namespace std::chrono;
                last_time = system_clock::now();
                clear_average_loss();
                for (; epoch_pos < data.size() && learning_rate >= min_learning_rate; epoch_pos += mini_batch_size)
                {
                    if (verbose)
                    {
                        auto now_time = system_clock::now();
                        if (now_time-last_time > seconds(20))
                        {
                            last_time = now_time;
                            auto iter = epoch_iteration + epoch_pos/(double)data.size();
                            std::cout << "epoch: " << rpad(cast_to_string(iter),epoch_string_pad) << "  " 
                                      << "learning rate: " << rpad(cast_to_string(learning_rate),lr_string_pad) << "  "
                                      << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                            print_progress();
                        }
                    }

                    sync_to_disk();
                    send_job(false, data.begin()+epoch_pos, 
                             data.begin()+std::min(epoch_pos+mini_batch_size,data.size()));
                }
                epoch_pos = 0;

                if (verbose)
                {
                    // Capitalize the E in Epoch so it's easy to grep out the lines that
                    // are for full epoch status statements.
                    std::cout << "Epoch: " << rpad(cast_to_string(epoch_iteration+1),epoch_string_pad) << "  " 
                              << "learning rate: " << rpad(cast_to_string(learning_rate),lr_string_pad) << "  "
                              << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                    print_progress();
                }
            }
            wait_for_thread_to_pause();
            // if we modified the network at all then be sure to sync the final result.
            sync_to_disk(true);
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
            std::ifstream fin(newest_syncfile(), std::ios::binary);
            if (fin)
                deserialize(*this, fin);
        }

        const std::string& get_synchronization_file (
        )
        {
            return sync_filename;
        }

        double get_average_loss (
        ) const 
        { 
            wait_for_thread_to_pause();
            return rs.mean();
        }

        double get_average_test_loss (
        ) const
        {
            wait_for_thread_to_pause();
            return rs_test.mean();
        }

        void clear_average_loss (
        )
        {
            wait_for_thread_to_pause();
            rs.clear();
        }

        void set_learning_rate (
            double lr
        )
        {
            DLIB_CASSERT(lr > 0);
            wait_for_thread_to_pause();
            if (learning_rate != lr)
            {
                steps_without_progress = 0;
                test_steps_without_progress = 0;
                previous_loss_values.clear();
                test_previous_loss_values.clear();
            }
            learning_rate = lr;
            lr_schedule.set_size(0);
        }

        double get_learning_rate(
        ) const 
        {
            return learning_rate;
        }

        void set_min_learning_rate (
            double lr
        )
        {
            DLIB_CASSERT(lr > 0);
            wait_for_thread_to_pause();
            lr_schedule.set_size(0);
            min_learning_rate = lr;
        }

        double get_min_learning_rate (
        ) const
        {
            return min_learning_rate;
        }

        template <typename EXP>
        void set_learning_rate_schedule (
            const matrix_exp<EXP>& schedule
        )
        {
            DLIB_CASSERT(schedule.size() > 0);
            DLIB_CASSERT(min(schedule) > 0);
            set_learning_rate(schedule(0,0));
            set_min_learning_rate(min(schedule));
            set_learning_rate_shrink_factor(1);
            lr_schedule = matrix_cast<double>(reshape_to_column_vector(schedule));
            lr_schedule_pos = 0;
        }

        const matrix<double,0,1>& get_learning_rate_schedule (
        ) const
        {
            return lr_schedule;
        }

        void set_iterations_without_progress_threshold (
            unsigned long thresh 
        )
        {
            wait_for_thread_to_pause();
            lr_schedule.set_size(0);
            iter_without_progress_thresh = thresh;
        }

        unsigned long get_iterations_without_progress_threshold (
        ) const
        {
            return iter_without_progress_thresh;
        }

        unsigned long get_steps_without_progress (
        ) const
        {
            return steps_without_progress;
        }

        void set_test_iterations_without_progress_threshold (
            unsigned long thresh 
        )
        {
            wait_for_thread_to_pause();
            lr_schedule.set_size(0);
            test_iter_without_progress_thresh = thresh;
        }

        unsigned long get_test_iterations_without_progress_threshold (
        ) const
        {
            return test_iter_without_progress_thresh;
        }

        unsigned long get_test_steps_without_progress (
        ) const
        {
            return test_steps_without_progress;
        }

        void set_learning_rate_shrink_factor (
            double shrink
        )
        {
            DLIB_CASSERT(0 < shrink && shrink <= 1);
            wait_for_thread_to_pause();
            lr_schedule.set_size(0);
            learning_rate_shrink = shrink;
            steps_without_progress = 0;
            test_steps_without_progress = 0;
        }

        double get_learning_rate_shrink_factor (
        ) const
        {
            return learning_rate_shrink;
        }

        unsigned long long get_train_one_step_calls (
        ) const
        {
            return train_one_step_calls;
        }

        unsigned long long get_test_one_step_calls (
        ) const
        {
            return test_one_step_calls;
        }

    private:

        void record_test_loss(double loss)
        {
            test_previous_loss_values.push_back(loss);
            if (is_finite(loss))
                rs_test.add(loss);
            // discard really old loss values.
            while (test_previous_loss_values.size() > test_iter_without_progress_thresh)
                test_previous_loss_values.pop_front();
        }

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
        double compute_parameter_gradients(size_t device, job_t& next_job, const T&)
        {
            if (next_job.have_data[device])
            {
                auto&& dev = *devices[device];
                dlib::cuda::set_device(dev.device_id);
                if (next_job.test_only)
                    return dev.net.compute_loss(next_job.t[device], next_job.labels[device].begin());
                else
                    return dev.net.compute_parameter_gradients(next_job.t[device], next_job.labels[device].begin());
            }
            else
            {
                return 0;
            }
        }

        double compute_parameter_gradients(size_t device, job_t& next_job, const no_label_type&)
        {
            if (next_job.have_data[device])
            {
                auto&& dev = *devices[device];
                dlib::cuda::set_device(dev.device_id);
                no_label_type pick_which_run_update;
                if (next_job.test_only)
                    return dev.net.compute_loss(next_job.t[device]);
                else
                    return dev.net.compute_parameter_gradients(next_job.t[device]);
            }
            else
            {
                return 0;
            }
        }

        void update_parameters(size_t device)
        {
            auto&& dev = *devices[device];
            dlib::cuda::set_device(dev.device_id);
            dev.net.update_parameters(make_sstack(dev.solvers), learning_rate);
        }

        void thread() try
        {
            training_label_type pick_which_run_update;
            job_t next_job;

            std::vector<dlib::future<double>> losses(devices.size());

            std::vector<tt::multi_device_tensor_averager> averagers;
            // An array of all the parameter tensors in the first network.  We will
            // periodically copy these tensors to all the other devices to make sure the
            // different GPUs don't go out of sync.
            std::vector<tensor*> reference_params;
            visit_layer_parameters(devices[0]->net, [&](size_t, tensor& t) { reference_params.push_back(&t); });

            // We make separate thread pools with just one thread in them because we want
            // to make sure each device is always executed on the same thread.  We care
            // about this because there are thread_local context variables for some cuda
            // components and they get allocated for each combination of thread and device.
            // So if we make sure the same device always uses the same thread this will
            // reduce the number of contexts we allocate from num_devices*num_devices to
            // just num_devices. 
            std::vector<std::shared_ptr<thread_pool>> tp;
            for (size_t i = 0; i < devices.size(); ++i)
                tp.push_back(std::make_shared<thread_pool>(1));


            main_iteration_counter = 0;
            while(job_pipe.dequeue(next_job))
            {
                if (next_job.test_only)
                {
                    // compute the testing loss
                    for (size_t i = 0; i < devices.size(); ++i)
                        tp[i]->add_task_by_value([&,i](double& loss){ loss = compute_parameter_gradients(i, next_job, pick_which_run_update); }, losses[i]);
                    // aggregate loss values from all the network computations.
                    double theloss = 0;
                    for (auto&& loss : losses)
                        theloss += loss.get();
                    record_test_loss(theloss/losses.size());

                    // Check if we should shrink the learning rate based on how the test
                    // error has been doing lately.
                    if (learning_rate_shrink != 1)
                    {
                        test_steps_without_progress = count_steps_without_decrease(test_previous_loss_values);
                        if (test_steps_without_progress >= test_iter_without_progress_thresh)
                        {
                            test_steps_without_progress = count_steps_without_decrease_robust(test_previous_loss_values);
                            if (test_steps_without_progress >= test_iter_without_progress_thresh)
                            {
                                // optimization has flattened out, so drop the learning rate. 
                                learning_rate = learning_rate_shrink*learning_rate;
                                test_steps_without_progress = 0;
                                // Empty out some of the previous loss values so that test_steps_without_progress 
                                // will decrease below test_iter_without_progress_thresh.  
                                for (unsigned long cnt = 0; cnt < test_previous_loss_values_dump_amount+test_iter_without_progress_thresh/10 && test_previous_loss_values.size() > 0; ++cnt)
                                    test_previous_loss_values.pop_front();
                            }
                        }
                    }
                    continue;
                }

                updated_net_since_last_sync = true;
                ++main_iteration_counter;
                // Call compute_parameter_gradients() and update_parameters() but pick the
                // right version for unsupervised or supervised training based on the type
                // of training_label_type.
                for (size_t i = 0; i < devices.size(); ++i)
                    tp[i]->add_task_by_value([&,i](double& loss){ loss = compute_parameter_gradients(i, next_job, pick_which_run_update); }, losses[i]);
                // aggregate loss values from all the network computations.
                double theloss = 0;
                for (auto&& loss : losses)
                    theloss += loss.get();
                record_loss(theloss/losses.size());

                // Now, if there is more than one active device we need to synchronize the
                // gradient updates between devices.  So we do that now.
                if (devices.size() > 1)
                {
                    // if this is the first iteration then we need to setup the averagers.
                    // We can't do this outside the loop because the tensors that get
                    // averaged need to be allocated to their devices before we call set()
                    // so that the averagers can determine how best to average them.
                    if (averagers.size() == 0 || sync_file_reloaded)
                    {
                        averagers = std::vector<tt::multi_device_tensor_averager>(net_type::num_computational_layers);
                        // setup the averagers to point to the tensors in the networks.
                        std::vector<std::vector<tensor*>> all_tensors(devices.size());
                        for (size_t i = 0; i < all_tensors.size(); ++i)
                        {
                            all_tensors[i].resize(net_type::num_computational_layers);
                            visit_layer_parameter_gradients(devices[i]->net, [&](size_t j, tensor& t){
                                all_tensors[i][j] = &t;
                            });
                        }
                        // Now set each averager to average the tensors at the same layer in each
                        // network.
                        for (size_t i = 0; i < net_type::num_computational_layers; ++i)
                        {
                            std::vector<tensor*> temp(all_tensors.size());
                            for (size_t j = 0; j < all_tensors.size(); ++j)
                                temp[j] = all_tensors[j][i];
                            // ignore layers that don't have parameters
                            if (temp[0]->size() != 0)
                                averagers[i].set(temp);
                        }

                        sync_file_reloaded = false;
                    }


                    for (auto&& d : devices)
                        cuda::device_synchronize(d->device_id);

                    for (auto&& avg : averagers)
                        avg.average();
                }


                // Now apply all the updates to each device.
                for (size_t i = 0; i < devices.size(); ++i)
                    tp[i]->add_task_by_value([&,i](){ if (next_job.have_data[i]) update_parameters(i); });
                // and wait for the updates to all happen.
                for (size_t i = 0; i < devices.size(); ++i)
                    tp[i]->wait_for_all_tasks();


                // Every now and then force all the parameters to be the same just to make
                // sure they aren't drifting apart due to any non-deterministic behavior on
                // the GPU.  It's also important to do this on the first iteration because
                // the different networks may be initialized differently when tensor data
                // is first passed through them.  So this code block deals with these
                // issues.
                if (devices.size() > 1 && main_iteration_counter%2000 == 1)
                {
                    for (size_t i = 1; i < devices.size(); ++i)
                    {
                        visit_layer_parameters(devices[i]->net, [&](size_t j, tensor& t) 
                        { 
                            memcpy(t, *reference_params[j]);
                        });
                    }
                }

                // If we have been running for a while then check if the loss is still
                // dropping.  If it isn't then we will reduce the learning rate.  Note that we
                // have a "budget" that prevents us from calling
                // count_steps_without_decrease() every iteration.  We do this because
                // it can be expensive to compute when previous_loss_values is large.
                if (gradient_check_budget > iter_without_progress_thresh && learning_rate_shrink != 1)
                {
                    gradient_check_budget = 0;
                    steps_without_progress = count_steps_without_decrease(previous_loss_values);
                    if (steps_without_progress >= iter_without_progress_thresh)
                    {
                        // Double check that we aren't seeing decrease.  This second check
                        // discards the top 10% largest values and checks again.  We do
                        // this because sometimes a mini-batch might be bad and cause the
                        // loss to suddenly jump up, making count_steps_without_decrease()
                        // return a large number.  But if we discard the top 10% of the
                        // values in previous_loss_values then we are robust to that kind
                        // of noise.  Another way of looking at it, if the reason
                        // count_steps_without_decrease() returns a large value is only
                        // because the most recent loss values have suddenly been large,
                        // then we shouldn't stop or lower the learning rate.  We should
                        // keep going until whatever disturbance we hit is damped down.  
                        steps_without_progress = count_steps_without_decrease_robust(previous_loss_values);
                        if (steps_without_progress >= iter_without_progress_thresh)
                        {
                            // optimization has flattened out, so drop the learning rate. 
                            learning_rate = learning_rate_shrink*learning_rate;
                            steps_without_progress = 0;
                            // Empty out some of the previous loss values so that steps_without_progress 
                            // will decrease below iter_without_progress_thresh.  
                            for (unsigned long cnt = 0; cnt < previous_loss_values_dump_amount+iter_without_progress_thresh/10 && previous_loss_values.size() > 0; ++cnt)
                                previous_loss_values.pop_front();
                        }
                    }
                }
                else if (lr_schedule.size() != 0) // or use the learning rate schedule if we have one.
                {
                    if (lr_schedule_pos < lr_schedule.size())
                        learning_rate = lr_schedule(lr_schedule_pos++);
                    else
                        learning_rate = lr_schedule(lr_schedule.size()-1)*0.99;
                }
            }
        }
        catch(...)
        {
            // If an exception happens then permanently disable the trainer object.
            job_pipe.disable();
            std::lock_guard<std::mutex> lock(eptr_mutex);
            eptr = std::current_exception();
        }

        void wait_for_thread_to_pause() const
        {
            job_pipe.wait_for_num_blocked_dequeues(1);
        }

        const static long string_pad = 11;
        const static long epoch_string_pad = 4;
        const static long lr_string_pad = 4;

        void init()
        {
            max_num_epochs = 10000;
            mini_batch_size = 128;
            verbose = false;
            learning_rate = 1e-2;
            min_learning_rate = 1e-5;
            iter_without_progress_thresh = 2000;
            steps_without_progress = 0;
            test_iter_without_progress_thresh = 500;
            test_steps_without_progress = 0;

            learning_rate_shrink = 0.1;
            epoch_iteration = 0;
            epoch_pos = 0;
            train_one_step_calls = 0;
            test_one_step_calls = 0;
            gradient_check_budget = 0;
            lr_schedule_pos = 0;

            main_iteration_counter = 0;
            main_iteration_counter_at_last_disk_sync = 0;
            prob_loss_increasing_thresh_default_value = 0.99;
            prob_loss_increasing_thresh_max_value = 0.99999;
            prob_loss_increasing_thresh = prob_loss_increasing_thresh_default_value;
            updated_net_since_last_sync = false;
            sync_file_reloaded = false;
            previous_loss_values_dump_amount = 400;
            test_previous_loss_values_dump_amount = 100;

            rs_test = running_stats_decayed<double>(200);

            start();
        }

        // serialize and deserialize are private because we hold net by reference so
        // allowing someone to serialize this training object is weird and will likely
        // result in user errors.  However, we use these functions as part of the automatic
        // sync code in this object.
        friend void serialize(const dnn_trainer& item, std::ostream& out)
        {
            item.wait_for_thread_to_pause();
            int version = 12;
            serialize(version, out);

            size_t nl = dnn_trainer::num_layers;
            serialize(nl, out);
            serialize(item.rs, out);
            serialize(item.rs_test, out);
            serialize(item.previous_loss_values, out);
            serialize(item.max_num_epochs, out);
            serialize(item.mini_batch_size, out);
            serialize(item.verbose, out);
            serialize(item.net, out);
            serialize(item.devices[0]->solvers, out);
            serialize(item.learning_rate.load(), out);
            serialize(item.min_learning_rate, out);
            serialize(item.iter_without_progress_thresh.load(), out);
            serialize(item.steps_without_progress.load(), out);
            serialize(item.learning_rate_shrink.load(), out);
            serialize(item.epoch_iteration, out);
            serialize(item.epoch_pos, out);
            serialize(item.train_one_step_calls, out);
            serialize(item.test_one_step_calls, out);
            serialize(item.lr_schedule, out);
            serialize(item.lr_schedule_pos, out);
            serialize(item.test_iter_without_progress_thresh.load(), out);
            serialize(item.test_steps_without_progress.load(), out);
            serialize(item.test_previous_loss_values, out);
            serialize(item.previous_loss_values_dump_amount, out);
            serialize(item.test_previous_loss_values_dump_amount, out);

        }
        friend void deserialize(dnn_trainer& item, std::istream& in)
        {
            item.wait_for_thread_to_pause();
            int version = 0;
            deserialize(version, in);
            if (version != 12)
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
            deserialize(item.rs_test, in);
            deserialize(item.previous_loss_values, in);
            deserialize(item.max_num_epochs, in);
            deserialize(item.mini_batch_size, in);
            deserialize(item.verbose, in);
            deserialize(item.net, in);
            deserialize(item.devices[0]->solvers, in);
            deserialize(dtemp, in); item.learning_rate = dtemp;
            deserialize(item.min_learning_rate, in);
            deserialize(ltemp, in); item.iter_without_progress_thresh = ltemp;
            deserialize(ltemp, in); item.steps_without_progress = ltemp;
            deserialize(dtemp, in); item.learning_rate_shrink = dtemp;
            deserialize(item.epoch_iteration, in);
            deserialize(item.epoch_pos, in);
            deserialize(item.train_one_step_calls, in);
            deserialize(item.test_one_step_calls, in);
            deserialize(item.lr_schedule, in);
            deserialize(item.lr_schedule_pos, in);
            deserialize(ltemp, in); item.test_iter_without_progress_thresh = ltemp;
            deserialize(ltemp, in); item.test_steps_without_progress = ltemp;
            deserialize(item.test_previous_loss_values, in);
            deserialize(item.previous_loss_values_dump_amount, in);
            deserialize(item.test_previous_loss_values_dump_amount, in);

            if (item.devices.size() > 1)
            {
                const auto prev_dev = dlib::cuda::get_device();
                // initialize all the other device networks and solver objects
                for (size_t i = 1; i < item.devices.size(); ++i)
                {
                    // Switch to this device so that any tensor objects that get allocated when
                    // we copy this stuff happen on this device.
                    dlib::cuda::set_device(item.devices[i]->device_id);
                    item.devices[i]->solvers = item.devices[0]->solvers;
                    item.devices[i]->net = item.devices[0]->net;
                }
                dlib::cuda::set_device(prev_dev);
            }
        }

        void sync_to_disk (
            bool do_it_now = false
        ) 
        {
            // don't sync anything if we haven't updated the network since the last sync
            if (!updated_net_since_last_sync)
                return;

            // If the sync file isn't set then don't do anything.
            if (sync_filename.size() == 0)
                return;

            // Only sync if it has been long enough since the last sync or we are being
            // explicitly forced to do it.
            if (std::chrono::system_clock::now() - last_sync_time > time_between_syncs ||
                do_it_now)
            {
                wait_for_thread_to_pause();

                // compact network before saving to disk.
                this->net.clean(); 

                // if the loss has actually been going up since the last time we saved our
                // state to disk then something has probably gone wrong in the
                // optimization.  So in this case we do the opposite and recall the
                // previously saved state in the hopes that the problem won't reoccur.
                if (loss_increased_since_last_disk_sync()) 
                {
                    std::ifstream fin(newest_syncfile(), std::ios::binary);
                    deserialize(*this, fin);
                    sync_file_reloaded = true;
                    if (verbose)
                        std::cout << "Loss has been increasing, reloading saved state from " << newest_syncfile() << std::endl;
                }
                else
                {

                    const std::string filename = oldest_syncfile();
                    serialize(filename) << *this;

                    if (verbose)
                        std::cout << "Saved state to " << filename << std::endl;
                }

                last_sync_time = std::chrono::system_clock::now();
                main_iteration_counter_at_last_disk_sync = main_iteration_counter;
                updated_net_since_last_sync = false;
            }
        }

        std::string newest_syncfile (
        )
        {
            return select_newest_file(sync_filename, sync_filename + "_");
        }

        std::string oldest_syncfile (
        )
        {
            return select_oldest_file(sync_filename, sync_filename + "_");
        }

        bool loss_increased_since_last_disk_sync() 
        {
            size_t gradient_updates_since_last_sync = main_iteration_counter - main_iteration_counter_at_last_disk_sync;

            // if we haven't synced anything to disk yet then return false.
            if (!std::ifstream(newest_syncfile(), std::ios::binary))
                return false;

            for (auto x : previous_loss_values)
            {
                // If we get a NaN value of loss assume things have gone horribly wrong and
                // we should reload the state of the trainer.
                if (std::isnan(x))
                    return true;
            }

            // if we haven't seen much data yet then just say false.  Or, alternatively, if
            // it's been too long since the last sync then don't reload either.
            if (gradient_updates_since_last_sync < 30 || previous_loss_values.size() < 2*gradient_updates_since_last_sync)
                return false;

            // Now look at the data since a little before the last disk sync.  We will
            // check if the loss is getting bettor or worse.
            running_gradient g;
            for (size_t i = previous_loss_values.size() - 2*gradient_updates_since_last_sync; i < previous_loss_values.size(); ++i)
                g.add(previous_loss_values[i]);

            // if the loss is very likely to be increasing then return true
            const double prob = g.probability_gradient_greater_than(0);
            if (prob > prob_loss_increasing_thresh && prob_loss_increasing_thresh <= prob_loss_increasing_thresh_max_value)
            {
                // Exponentially decay the threshold towards 1 so that if we keep finding
                // the loss to be increasing over and over we will make the test
                // progressively harder and harder until it fails, therefore ensuring we
                // can't get stuck reloading from a previous state over and over. 
                prob_loss_increasing_thresh = 0.1*prob_loss_increasing_thresh + 0.9*1;
                return true;
            }
            else
            {
                // decay back to the default threshold
                prob_loss_increasing_thresh = std::pow(prob_loss_increasing_thresh, 10.0);
                // but don't decay below the default value
                prob_loss_increasing_thresh = std::max(prob_loss_increasing_thresh, prob_loss_increasing_thresh_default_value);

                return false;
            }
        }


        struct clone_net{};

        // per device state.  All the containers have the same number of objects in them.
        struct device_data
        {
            device_data(
                int device_id_,
                net_type& net_,
                const solver_type& solver_
            ) : device_id(device_id_), net(net_), solvers(num_computational_layers, solver_) {}

            device_data(
                int device_id_,
                net_type& net_,
                const solver_type& solver_,
                clone_net
            ) : device_id(device_id_), net_copy(std::make_shared<net_type>(net_)), net(*net_copy), solvers(num_computational_layers, solver_) {}

            int device_id;
            std::shared_ptr<net_type> net_copy;
            net_type& net;
            std::vector<solver_type> solvers;
        };

        template <
            typename data_iterator,
            typename label_iterator
            >
        void send_job (
            bool test_only,
            data_iterator dbegin, 
            data_iterator dend,
            label_iterator lbegin
        )
        {
            propagate_exception();
            size_t num = std::distance(dbegin, dend);
            size_t devs = devices.size();
            job.t.resize(devs);
            job.labels.resize(devs);
            job.have_data.resize(devs);
            job.test_only = test_only;

            // chop the data into devs blocks, each of about block_size elements.
            size_t block_size = (num+devs-1)/devs;

            const auto prev_dev = dlib::cuda::get_device();
            for (size_t i = 0; i < devs; ++i)
            {
                dlib::cuda::set_device(devices[i]->device_id);

                size_t start = i*block_size;
                size_t stop  = std::min(num, start+block_size);

                if (start < stop)
                {
                    devices[i]->net.to_tensor(dbegin+start, dbegin+stop, job.t[i]);
                    job.labels[i].assign(lbegin+start, lbegin+stop);
                    job.have_data[i] = true;
                }
                else
                {
                    job.have_data[i] = false;
                }
            }

            dlib::cuda::set_device(prev_dev);
            job_pipe.enqueue(job);
        }

        template <
            typename data_iterator
            >
        void send_job (
            bool test_only,
            data_iterator dbegin, 
            data_iterator dend
        )
        {
            typename std::vector<training_label_type>::iterator nothing;
            send_job(test_only, dbegin, dend, nothing);
        }

        void print_progress()
        {
            if (lr_schedule.size() == 0)
            {
                if (test_previous_loss_values.size() == 0)
                    std::cout << "steps without apparent progress: " << steps_without_progress;
                else
                    std::cout << "steps without apparent progress: train=" << steps_without_progress << ", test=" << test_steps_without_progress;
            }
            else
            {
                std::ostringstream sout;
                sout << "percent complete: " << std::fixed << std::setprecision(2) << 100.0*lr_schedule_pos/(double)lr_schedule.size() << "%";
                std::cout << sout.str();
            }
            std::cout << std::endl;
        }

        void print_periodic_verbose_status()
        {
            if (verbose)
            {
                using namespace std::chrono;
                auto now_time = system_clock::now();
                if (now_time-last_time > seconds(40))
                {
                    last_time = now_time;
                    std::cout << "step#: " << rpad(cast_to_string(train_one_step_calls),epoch_string_pad) << "  " 
                              << "learning rate: " << rpad(cast_to_string(learning_rate),lr_string_pad) << "  ";
                    if (test_previous_loss_values.size() == 0)
                    {
                        std::cout << "average loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                    }
                    else
                    {
                        std::cout << "train loss: " << rpad(cast_to_string(get_average_loss()),string_pad) << "  ";
                        std::cout << "test loss: " << rpad(cast_to_string(get_average_test_loss()),string_pad) << "  ";
                    }
                    print_progress();
                    clear_average_loss();
                }
            }
        }

        std::vector<std::shared_ptr<device_data>> devices;
        dlib::pipe<job_t> job_pipe;
        job_t job;


        running_stats<double> rs;
        running_stats_decayed<double> rs_test;
        std::deque<double> previous_loss_values;
        unsigned long max_num_epochs;
        size_t mini_batch_size;
        bool verbose;
        net_type& net;
        std::atomic<double> learning_rate;
        double min_learning_rate;
        std::atomic<unsigned long> iter_without_progress_thresh;
        std::atomic<unsigned long> steps_without_progress;

        std::atomic<unsigned long> test_iter_without_progress_thresh;
        std::atomic<unsigned long> test_steps_without_progress;
        std::deque<double> test_previous_loss_values;

        std::atomic<double> learning_rate_shrink;
        std::chrono::time_point<std::chrono::system_clock> last_sync_time;
        std::string sync_filename;
        std::chrono::seconds time_between_syncs;
        unsigned long epoch_iteration;
        size_t epoch_pos;
        std::chrono::time_point<std::chrono::system_clock> last_time;
        unsigned long long train_one_step_calls;
        unsigned long long test_one_step_calls;
        matrix<double,0,1> lr_schedule;
        long lr_schedule_pos;
        unsigned long gradient_check_budget;

        std::exception_ptr eptr = nullptr;
        mutable std::mutex eptr_mutex;
        void propagate_exception() const
        {
            std::lock_guard<std::mutex> lock(eptr_mutex);
            if (eptr)
                std::rethrow_exception(eptr);
        }

        // These 5 variables are not serialized 
        size_t main_iteration_counter;
        size_t main_iteration_counter_at_last_disk_sync;
        double prob_loss_increasing_thresh_default_value;
        double prob_loss_increasing_thresh_max_value;
        double prob_loss_increasing_thresh;
        std::atomic<bool> updated_net_since_last_sync;

        bool sync_file_reloaded;
        unsigned long previous_loss_values_dump_amount;
        unsigned long test_previous_loss_values_dump_amount;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type 
        >
    std::ostream& operator<< (
        std::ostream& out,
        dnn_trainer<net_type,solver_type>& trainer
    )
    {
        using std::endl;
        out << "dnn_trainer details: \n";
        out << "  net_type::num_layers:  " << net_type::num_layers << endl;
        // figure out how big the net is in MB.
        std::ostringstream sout;
        net_type temp = trainer.get_net(); // make a copy so that we can clean it without mutating the trainer's net.
        temp.clean();
        serialize(temp, sout);
        out << "  net size: " << sout.str().size()/1024.0/1024.0 << "MB" << endl;
        // Don't include the loss params in the hash since we print them on the next line.
        // They also aren't really part of the "architecture" of the network.
        out << "  net architecture hash: " << md5(cast_to_string(trainer.get_net().subnet())) << endl;
        out << "  loss: " << trainer.get_net().loss_details() << endl;

        out << "  synchronization file:                       " << trainer.get_synchronization_file() << endl;
        out << "  trainer.get_solvers()[0]:                   " << trainer.get_solvers()[0] << endl;
        auto sched = trainer.get_learning_rate_schedule();
        if (sched.size() != 0)
        {
            out << "  using explicit user-supplied learning rate schedule" << endl;
        }
        else
        {
            out << "  learning rate:                              "<< trainer.get_learning_rate() << endl;
            out << "  learning rate shrink factor:                "<< trainer.get_learning_rate_shrink_factor() << endl;
            out << "  min learning rate:                          "<< trainer.get_min_learning_rate() << endl;
            out << "  iterations without progress threshold:      "<< trainer.get_iterations_without_progress_threshold() << endl;
            out << "  test iterations without progress threshold: "<< trainer.get_test_iterations_without_progress_threshold() << endl;
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TRAINER_H_

