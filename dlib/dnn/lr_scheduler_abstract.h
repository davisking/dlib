// Copyright (C) 2025  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_LR_SCHEDULER_ABSTRACT_H_
#ifdef DLIB_DNN_LR_SCHEDULER_ABSTRACT_H_

#include <cstddef>
#include <iostream>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class lr_decay_type
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                This enum specifies the type of learning rate decay to use after the
                warmup phase completes. The decay function determines how the learning
                rate decreases from peak_lr to min_lr over the remaining training steps.
        !*/

        COSINE,
        /*!
            Cosine annealing decay. The learning rate follows a cosine curve:
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
            
            This is the recommended decay type for transformer training as it provides
            smooth decay with a gradual slowdown near the end of training.
        !*/

        LINEAR,
        /*!
            Linear decay. The learning rate decreases linearly:
                lr = peak_lr - (peak_lr - min_lr) * progress
            
            Simple and predictable decay suitable for general deep learning tasks.
        !*/

        CONSTANT,
        /*!
            No decay after warmup. The learning rate remains at peak_lr:
                lr = peak_lr
            
            Useful when using external learning rate control or for debugging.
        !*/

        EXPONENTIAL
        /*!
            Exponential decay. The learning rate decreases exponentially:
                lr = peak_lr * (min_lr / peak_lr)^progress
            
            Provides rapid initial decay that slows down over time.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class lr_scheduler
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements a learning rate scheduler with warmup and decay
                phases, designed for training transformer-based neural networks. It is
                intended to be used alongside dnn_trainer to provide dynamic learning
                rate adjustment during training.

                The schedule consists of three phases:
                    1. WARMUP: Linear increase from initial_lr to peak_lr
                    2. HOLD (optional): Maintain peak_lr for hold_steps
                    3. DECAY: Decrease from peak_lr to min_lr using selected decay type

            MATHEMATICAL FORMULATION
                Warmup phase (step < warmup_steps):
                    lr = initial_lr + (peak_lr - initial_lr) * (step / warmup_steps)

                Hold phase (warmup_steps <= step < warmup_steps + hold_steps):
                    lr = peak_lr

                Decay phase (step >= warmup_steps + hold_steps):
                    progress = (step - warmup_steps - hold_steps) / decay_steps

                    For COSINE:
                        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))

                    For LINEAR:
                        lr = peak_lr - (peak_lr - min_lr) * progress

                    For EXPONENTIAL:
                        lr = peak_lr * (min_lr / peak_lr)^progress

                    For CONSTANT:
                        lr = peak_lr

            THREAD SAFETY
                This object is not thread-safe. Each trainer should have its own scheduler
                instance. If using multiple trainers in parallel, each should maintain its
                own lr_scheduler.

            SERIALIZATION
                This object supports serialization through serialize() and deserialize()
                functions, allowing training to be checkpointed and resumed.

            TYPICAL USAGE
                // Create scheduler
                lr_scheduler scheduler(
                    3e-4,       // peak_lr
                    2000,       // warmup_steps
                    100000,     // total_steps
                    1e-6,       // min_lr
                    lr_decay_type::COSINE
                );

                // Training loop
                while (!scheduler.is_training_complete()) {
                    trainer.set_learning_rate(scheduler.get_learning_rate());
                    trainer.train_one_step(data, labels);
                    scheduler.step();
                }
        !*/

    public:

        lr_scheduler(
        );
        /*!
            ensures
                - Constructs a default scheduler with reasonable defaults for transformer training
                - #get_peak_lr() == 3e-4
                - #get_min_lr() == 1e-6
                - #get_initial_lr() == 1e-7
                - #get_warmup_steps() == 2000
                - #get_hold_steps() == 0
                - #get_total_steps() == 100000
                - #get_decay_type() == lr_decay_type::COSINE
                - #get_current_step() == 0
        !*/

        lr_scheduler(
            double peak_lr,
            size_t warmup_steps,
            size_t total_steps,
            double min_lr = 1e-6,
            lr_decay_type decay_type = lr_decay_type::COSINE
        );
        /*!
            requires
                - peak_lr > 0
                - min_lr >= 0
                - min_lr < peak_lr
                - warmup_steps < total_steps
            ensures
                - #get_peak_lr() == peak_lr
                - #get_min_lr() == min_lr
                - #get_initial_lr() == min_lr
                - #get_warmup_steps() == warmup_steps
                - #get_hold_steps() == 0
                - #get_total_steps() == total_steps
                - #get_decay_type() == decay_type
                - #get_current_step() == 0
        !*/

        double get_learning_rate(
        ) const;
        /*!
            ensures
                - Returns the learning rate for the current step based on the schedule
                - The returned value is always >= get_min_lr()
                - The returned value is always <= get_peak_lr()
                - During warmup: returns a value linearly interpolated between
                  get_initial_lr() and get_peak_lr()
                - During hold: returns get_peak_lr()
                - During decay: returns a value determined by get_decay_type()
        !*/

        double get_learning_rate(
            size_t step
        ) const;
        /*!
            ensures
                - Returns the learning rate that would be used at the specified step
                - Does not modify the scheduler state
                - Equivalent to temporarily setting current_step to step and calling
                  get_learning_rate(), then restoring the original current_step
        !*/

        void step(
            size_t n = 1
        );
        /*!
            ensures
                - #get_current_step() == get_current_step() + n
                - Advances the scheduler by n steps
        !*/

        void reset(
        );
        /*!
            ensures
                - #get_current_step() == 0
                - Resets the scheduler to its initial state
        !*/

        void set_current_step(
            size_t step
        );
        /*!
            ensures
                - #get_current_step() == step
                - Useful for resuming training from a checkpoint
        !*/

        size_t get_current_step(
        ) const;
        /*!
            ensures
                - Returns the current training step
        !*/

        size_t get_warmup_steps(
        ) const;
        /*!
            ensures
                - Returns the number of warmup steps configured for this scheduler
                - During warmup, the learning rate increases linearly from
                  get_initial_lr() to get_peak_lr()
        !*/

        size_t get_hold_steps(
        ) const;
        /*!
            ensures
                - Returns the number of hold steps configured for this scheduler
                - During hold, the learning rate remains constant at get_peak_lr()
        !*/

        size_t get_total_steps(
        ) const;
        /*!
            ensures
                - Returns the total number of training steps configured for this scheduler
                - Training is considered complete when get_current_step() >= get_total_steps()
        !*/

        size_t get_decay_steps(
        ) const;
        /*!
            ensures
                - Returns the number of steps in the decay phase
                - Computed as: get_total_steps() - get_warmup_steps() - get_hold_steps()
        !*/

        double get_initial_lr(
        ) const;
        /*!
            ensures
                - Returns the initial learning rate at the start of warmup
                - This is the learning rate used at step 0
        !*/

        double get_peak_lr(
        ) const;
        /*!
            ensures
                - Returns the peak learning rate reached at the end of warmup
                - This is the maximum learning rate during training
        !*/

        double get_min_lr(
        ) const;
        /*!
            ensures
                - Returns the minimum learning rate at the end of training
                - The learning rate will never go below this value
        !*/

        lr_decay_type get_decay_type(
        ) const;
        /*!
            ensures
                - Returns the decay type used after warmup completes
        !*/

        void set_peak_lr(
            double lr
        );
        /*!
            requires
                - lr > 0
                - lr > get_min_lr()
            ensures
                - #get_peak_lr() == lr
        !*/

        void set_min_lr(
            double lr
        );
        /*!
            requires
                - lr >= 0
                - lr < get_peak_lr()
            ensures
                - #get_min_lr() == lr
        !*/

        void set_initial_lr(
            double lr
        );
        /*!
            requires
                - lr >= 0
                - lr <= get_peak_lr()
            ensures
                - #get_initial_lr() == lr
        !*/

        void set_warmup_steps(
            size_t steps
        );
        /*!
            requires
                - steps < get_total_steps()
            ensures
                - #get_warmup_steps() == steps
                - #get_decay_steps() is recomputed accordingly
        !*/

        void set_hold_steps(
            size_t steps
        );
        /*!
            ensures
                - #get_hold_steps() == steps
                - #get_decay_steps() is recomputed accordingly
        !*/

        void set_total_steps(
            size_t steps
        );
        /*!
            requires
                - steps > get_warmup_steps()
            ensures
                - #get_total_steps() == steps
                - #get_decay_steps() is recomputed accordingly
        !*/

        void set_decay_type(
            lr_decay_type type
        );
        /*!
            ensures
                - #get_decay_type() == type
        !*/

        bool is_warmup_complete(
        ) const;
        /*!
            ensures
                - Returns true if the warmup phase has completed
                - Equivalent to: get_current_step() >= get_warmup_steps()
        !*/

        bool is_training_complete(
        ) const;
        /*!
            ensures
                - Returns true if all training steps have been completed
                - Equivalent to: get_current_step() >= get_total_steps()
        !*/

        double get_warmup_progress(
        ) const;
        /*!
            ensures
                - Returns a value between 0.0 and 1.0 indicating progress through warmup
                - Returns 1.0 if warmup is complete
                - Computed as: min(1.0, get_current_step() / get_warmup_steps())
        !*/

        double get_total_progress(
        ) const;
        /*!
            ensures
                - Returns a value between 0.0 and 1.0 indicating overall training progress
                - Computed as: min(1.0, get_current_step() / get_total_steps())
        !*/

        std::string get_phase_name(
        ) const;
        /*!
            ensures
                - Returns "warmup" if in the warmup phase
                - Returns "hold" if in the hold phase
                - Returns "decay" if in the decay phase
        !*/
    };

// ----------------------------------------------------------------------------------------

    void serialize(
        const lr_scheduler& item,
        std::ostream& out
    );
    /*!
        ensures
            - Serializes the complete state of item to the output stream out
            - The serialized state includes: current_step, warmup_steps, hold_steps,
              total_steps, decay_steps, initial_lr, peak_lr, min_lr, and decay_type
    !*/

    void deserialize(
        lr_scheduler& item,
        std::istream& in
    );
    /*!
        ensures
            - Deserializes the state of item from the input stream in
            - Restores all configuration and progress state
        throws
            - serialization_error if the data in 'in' is not valid lr_scheduler data
    !*/

    std::ostream& operator<<(
        std::ostream& out,
        const lr_scheduler& item
    );
    /*!
        ensures
            - Prints a human-readable summary of the scheduler state to out
            - Includes: current step, current learning rate, phase name, and configuration
    !*/

// ----------------------------------------------------------------------------------------

    lr_scheduler make_transformer_scheduler(
        double peak_lr,
        size_t total_steps,
        double warmup_fraction = 0.02,
        double min_lr = 1e-6,
        lr_decay_type decay_type = lr_decay_type::COSINE
    );
    /*!
        requires
            - peak_lr > 0
            - total_steps > 0
            - 0 < warmup_fraction < 1
            - min_lr >= 0
            - min_lr < peak_lr
        ensures
            - Returns an lr_scheduler configured with common transformer training settings
            - The warmup_steps is computed as: max(100, total_steps * warmup_fraction)
            - returns a scheduler S such that:
                - S.get_peak_lr() == peak_lr
                - S.get_total_steps() == total_steps
                - S.get_min_lr() == min_lr
                - S.get_decay_type() == decay_type
                - S.get_warmup_steps() == max(100, total_steps * warmup_fraction)
    !*/

    size_t estimate_total_steps(
        size_t dataset_size,
        size_t batch_size,
        size_t num_epochs
    );
    /*!
        requires
            - batch_size > 0
        ensures
            - Returns an estimate of the total number of training steps
            - Computed as: ceil(dataset_size / batch_size) * num_epochs
            - Useful for configuring lr_scheduler when you know the dataset size,
              batch size, and desired number of epochs
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_LR_SCHEDULER_ABSTRACT_H_
