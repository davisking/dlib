// Copyright (C) 2025  Cydral (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_LR_SCHEDULER_H_
#define DLIB_DNN_LR_SCHEDULER_H_

#include "lr_scheduler_abstract.h"
#include "../serialize.h"
#include <cmath>
#include <algorithm>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        constexpr double lr_scheduler_pi = 3.14159265358979323846;
    }

// ----------------------------------------------------------------------------------------

    enum class lr_decay_type
    {
        COSINE,
        LINEAR,
        CONSTANT,
        EXPONENTIAL
    };

// ----------------------------------------------------------------------------------------

    class lr_scheduler
    {
    public:

        lr_scheduler(
        ) :
            current_step_(0),
            warmup_steps_(2000),
            hold_steps_(0),
            total_steps_(100000),
            initial_lr_(1e-7),
            peak_lr_(3e-4),
            min_lr_(1e-6),
            decay_type_(lr_decay_type::COSINE)
        {
            compute_decay_steps();
        }

        lr_scheduler(
            double peak_lr,
            size_t warmup_steps,
            size_t total_steps,
            double min_lr = 1e-6,
            lr_decay_type decay_type = lr_decay_type::COSINE
        ) :
            current_step_(0),
            warmup_steps_(warmup_steps),
            hold_steps_(0),
            total_steps_(total_steps),
            initial_lr_(min_lr),
            peak_lr_(peak_lr),
            min_lr_(min_lr),
            decay_type_(decay_type)
        {
            DLIB_CASSERT(peak_lr > 0, "peak_lr must be positive");
            DLIB_CASSERT(min_lr >= 0, "min_lr must be non-negative");
            DLIB_CASSERT(min_lr < peak_lr, "min_lr must be less than peak_lr");
            DLIB_CASSERT(warmup_steps < total_steps, "warmup_steps must be less than total_steps");
            compute_decay_steps();
        }

        double get_learning_rate(
        ) const
        {
            // Phase 1: Warmup
            if (current_step_ < warmup_steps_)
            {
                if (warmup_steps_ == 0)
                    return peak_lr_;
                const double progress = static_cast<double>(current_step_) / warmup_steps_;
                return initial_lr_ + (peak_lr_ - initial_lr_) * progress;
            }

            // Phase 2: Hold (optional)
            const size_t post_warmup = current_step_ - warmup_steps_;
            if (post_warmup < hold_steps_)
                return peak_lr_;

            // Phase 3: Decay
            if (decay_steps_ == 0)
                return peak_lr_;

            const size_t decay_step = post_warmup - hold_steps_;
            const double progress = std::min(1.0, static_cast<double>(decay_step) / decay_steps_);

            switch (decay_type_)
            {
            case lr_decay_type::COSINE:
                return min_lr_ + 0.5 * (peak_lr_ - min_lr_) * (1.0 + std::cos(impl::lr_scheduler_pi * progress));

            case lr_decay_type::LINEAR:
                return peak_lr_ - (peak_lr_ - min_lr_) * progress;

            case lr_decay_type::EXPONENTIAL:
                return peak_lr_ * std::pow(min_lr_ / peak_lr_, progress);

            case lr_decay_type::CONSTANT:
            default:
                return peak_lr_;
            }
        }

        double get_learning_rate(
            size_t step
        ) const
        {
            lr_scheduler temp = *this;
            temp.current_step_ = step;
            return temp.get_learning_rate();
        }

        void step(
            size_t n = 1
        )
        {
            current_step_ += n;
        }

        void reset(
        )
        {
            current_step_ = 0;
        }

        void set_current_step(
            size_t step
        )
        {
            current_step_ = step;
        }

        size_t get_current_step(
        ) const { return current_step_; }

        size_t get_warmup_steps(
        ) const { return warmup_steps_; }

        size_t get_hold_steps(
        ) const { return hold_steps_; }

        size_t get_total_steps(
        ) const { return total_steps_; }

        size_t get_decay_steps(
        ) const { return decay_steps_; }

        double get_initial_lr(
        ) const { return initial_lr_; }

        double get_peak_lr(
        ) const { return peak_lr_; }

        double get_min_lr(
        ) const { return min_lr_; }

        lr_decay_type get_decay_type(
        ) const { return decay_type_; }

        void set_peak_lr(
            double lr
        )
        {
            DLIB_CASSERT(lr > 0 && lr > min_lr_);
            peak_lr_ = lr;
        }

        void set_min_lr(
            double lr
        )
        {
            DLIB_CASSERT(lr >= 0 && lr < peak_lr_);
            min_lr_ = lr;
        }

        void set_initial_lr(
            double lr
        )
        {
            DLIB_CASSERT(lr >= 0 && lr <= peak_lr_);
            initial_lr_ = lr;
        }

        void set_warmup_steps(
            size_t steps
        )
        {
            DLIB_CASSERT(steps < total_steps_);
            warmup_steps_ = steps;
            compute_decay_steps();
        }

        void set_hold_steps(
            size_t steps
        )
        {
            hold_steps_ = steps;
            compute_decay_steps();
        }

        void set_total_steps(
            size_t steps
        )
        {
            DLIB_CASSERT(steps > warmup_steps_);
            total_steps_ = steps;
            compute_decay_steps();
        }

        void set_decay_type(
            lr_decay_type type
        )
        {
            decay_type_ = type;
        }

        bool is_warmup_complete(
        ) const { return current_step_ >= warmup_steps_; }

        bool is_training_complete(
        ) const { return current_step_ >= total_steps_; }

        double get_warmup_progress(
        ) const
        {
            if (warmup_steps_ == 0)
                return 1.0;
            return std::min(1.0, static_cast<double>(current_step_) / warmup_steps_);
        }

        double get_total_progress(
        ) const
        {
            if (total_steps_ == 0)
                return 1.0;
            return std::min(1.0, static_cast<double>(current_step_) / total_steps_);
        }

        std::string get_phase_name(
        ) const
        {
            if (current_step_ < warmup_steps_)
                return "warmup";
            else if (current_step_ < warmup_steps_ + hold_steps_)
                return "hold";
            else
                return "decay";
        }

    private:

        void compute_decay_steps(
        )
        {
            const size_t non_decay = warmup_steps_ + hold_steps_;
            decay_steps_ = (total_steps_ > non_decay) ? (total_steps_ - non_decay) : 0;
        }

        size_t current_step_;
        size_t warmup_steps_;
        size_t hold_steps_;
        size_t total_steps_;
        size_t decay_steps_;
        double initial_lr_;
        double peak_lr_;
        double min_lr_;
        lr_decay_type decay_type_;
    };

// ----------------------------------------------------------------------------------------

    inline void serialize(
        const lr_scheduler& item,
        std::ostream& out
    )
    {
        serialize("lr_scheduler", out);
        serialize(item.get_current_step(), out);
        serialize(item.get_warmup_steps(), out);
        serialize(item.get_hold_steps(), out);
        serialize(item.get_total_steps(), out);
        serialize(item.get_decay_steps(), out);
        serialize(item.get_initial_lr(), out);
        serialize(item.get_peak_lr(), out);
        serialize(item.get_min_lr(), out);
        serialize(static_cast<int>(item.get_decay_type()), out);
    }

    inline void deserialize(
        lr_scheduler& item,
        std::istream& in
    )
    {
        std::string version;
        deserialize(version, in);
        if (version != "lr_scheduler")
            throw serialization_error("Unexpected version '" + version +
                "' found while deserializing lr_scheduler.");

        size_t current_step, warmup_steps, hold_steps, total_steps, decay_steps;
        double initial_lr, peak_lr, min_lr;
        int decay_type_int;

        deserialize(current_step, in);
        deserialize(warmup_steps, in);
        deserialize(hold_steps, in);
        deserialize(total_steps, in);
        deserialize(decay_steps, in);
        deserialize(initial_lr, in);
        deserialize(peak_lr, in);
        deserialize(min_lr, in);
        deserialize(decay_type_int, in);

        item = lr_scheduler(peak_lr, warmup_steps, total_steps, min_lr,
            static_cast<lr_decay_type>(decay_type_int));
        item.set_initial_lr(initial_lr);
        item.set_hold_steps(hold_steps);
        item.set_current_step(current_step);
    }

    inline std::ostream& operator<<(
        std::ostream& out,
        const lr_scheduler& item
    )
    {
        out << "lr_scheduler ("
            << "step=" << item.get_current_step()
            << ", lr=" << item.get_learning_rate()
            << ", phase=" << item.get_phase_name()
            << ", warmup=" << item.get_warmup_steps()
            << ", total=" << item.get_total_steps()
            << ", peak=" << item.get_peak_lr()
            << ", min=" << item.get_min_lr()
            << ")";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline lr_scheduler make_transformer_scheduler(
        double peak_lr,
        size_t total_steps,
        double warmup_fraction = 0.02,
        double min_lr = 1e-6,
        lr_decay_type decay_type = lr_decay_type::COSINE
    )
    {
        DLIB_CASSERT(peak_lr > 0, "peak_lr must be positive");
        DLIB_CASSERT(total_steps > 0, "total_steps must be positive");
        DLIB_CASSERT(warmup_fraction > 0 && warmup_fraction < 1, "warmup_fraction must be in (0, 1)");
        DLIB_CASSERT(min_lr >= 0 && min_lr < peak_lr, "min_lr must be in [0, peak_lr)");

        size_t warmup_steps = static_cast<size_t>(total_steps * warmup_fraction);
        warmup_steps = std::max(size_t(100), warmup_steps);
        return lr_scheduler(peak_lr, warmup_steps, total_steps, min_lr, decay_type);
    }

    inline size_t estimate_total_steps(
        size_t dataset_size,
        size_t batch_size,
        size_t num_epochs
    )
    {
        DLIB_CASSERT(batch_size > 0, "batch_size must be positive");
        const size_t steps_per_epoch = (dataset_size + batch_size - 1) / batch_size;
        return steps_per_epoch * num_epochs;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNN_LR_SCHEDULER_H_
