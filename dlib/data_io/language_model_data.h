#ifndef DLIB_LANGUAGE_MODEL_DATA_H_
#define DLIB_LANGUAGE_MODEL_DATA_H_

#include "language_model_data_abstract.h"

#include <iostream>
#include <string>
#include <vector>
#include "../matrix.h"
#include "../serialize.h"

namespace dlib
{
    class inference_context
    {
    public:
        inference_context(
            long window_size = 256,
            long context_multiplier = 10,
            long padding_token = 0
        ) : window_size_(window_size),
            context_capacity_(window_size * context_multiplier),
            padding_token_(padding_token),
            current_size_(0)
        {
            DLIB_CASSERT(window_size > 0, "Window size must be positive");
            DLIB_CASSERT(context_multiplier > 0, "Context multiplier must be positive");
            context_.reserve(context_capacity_);
        }

        void add_token(unsigned long token)
        {
            if (current_size_ == context_capacity_)
            {
                // FIFO: remove oldest, add newest
                context_.erase(context_.begin());
                context_.push_back(static_cast<int>(token));
            }
            else
            {
                // Still room in context
                context_.push_back(static_cast<int>(token));
                current_size_++;
            }
        }

        void add_tokens(const std::vector<unsigned long>& tokens)
        {
            for (unsigned long token : tokens) add_token(token);
        }

        void add_tokens(const std::vector<int>& tokens)
        {
            for (int token : tokens) add_token(static_cast<unsigned long>(token));
        }

        matrix<int, 0, 1> get_input_window(long custom_window_size = -1) const
        {
            long win_size = (custom_window_size > 0) ? custom_window_size : window_size_;
            matrix<int, 0, 1> window(win_size, 1);

            if (current_size_ >= win_size)
            {
                // Context has enough tokens - take last win_size tokens
                for (long i = 0; i < win_size; ++i)
                    window(i) = context_[current_size_ - win_size + i];
            }
            else
            {
                // Context has fewer tokens - left pad
                long padding_needed = win_size - current_size_;

                for (long i = 0; i < padding_needed; ++i)
                    window(i) = padding_token_;
                for (long i = 0; i < current_size_; ++i)
                    window(padding_needed + i) = context_[i];
            }

            return window;
        }

        void reset()
        {
            context_.clear();
            current_size_ = 0;
        }

        void resize_context(long new_capacity)
        {
            DLIB_CASSERT(new_capacity > 0, "New capacity must be positive");

            if (new_capacity < current_size_)
            {
                // Keep only the last new_capacity tokens
                context_.erase(context_.begin(), context_.begin() + (current_size_ - new_capacity));
                current_size_ = new_capacity;
            }

            context_capacity_ = new_capacity;
            context_.reserve(context_capacity_);
        }

        long size() const { return current_size_; }
        long capacity() const { return context_capacity_; }
        long window_size() const { return window_size_; }
        bool is_full() const { return current_size_ >= context_capacity_; }
        const std::vector<int>& get_full_context() const { return context_; }

        std::string to_string(bool show_all = false) const
        {
            std::ostringstream ss;
            ss << "InferenceContext[size=" << current_size_
                << "/" << context_capacity_
                << ", window=" << window_size_ << "]\n";

            if (show_all && current_size_ > 0)
            {
                ss << "Tokens: [";
                long display_count = show_all ? current_size_ : std::min(20L, current_size_);
                for (long i = 0; i < display_count; ++i)
                {
                    ss << context_[i];
                    if (i < display_count - 1) ss << ", ";
                }
                if (current_size_ > display_count)
                {
                    ss << " ... +" << (current_size_ - display_count) << " more";
                }
                ss << "]";
            }

            return ss.str();
        }

        friend void serialize(const inference_context& item, std::ostream& out)
        {
            serialize("inference_context", out);
            serialize(item.window_size_, out);
            serialize(item.context_capacity_, out);
            serialize(item.padding_token_, out);
            serialize(item.current_size_, out);
            serialize(item.context_, out);
        }

        friend void deserialize(inference_context& item, std::istream& in)
        {
            std::string name;
            deserialize(name, in);
            if (name != "inference_context")
            {
                throw serialization_error("Error deserializing object of type 'inference_context': "
                    "expected 'inference_context' but got '" + name + "'");
            }

            deserialize(item.window_size_, in);
            deserialize(item.context_capacity_, in);
            deserialize(item.padding_token_, in);
            deserialize(item.current_size_, in);
            deserialize(item.context_, in);
        }

    private:
        std::vector<int> context_;      // Full context history
        long window_size_;               // Window size for model input
        long context_capacity_;          // Maximum context size
        long padding_token_;             // Token used for left padding
        long current_size_;              // Current number of tokens
    };

    inline void build_single_token_prediction_dataset(
        const std::vector<std::vector<int>>& token_sequences,
        long window_len,
        long padding_token,
        bool use_left_padding,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<unsigned long>& Y)
    {
        X.clear();
        Y.clear();

        for (const auto& seq : token_sequences)
        {
            const long len = static_cast<long>(seq.size());
            if (len <= 1) continue;

            long start = 0;
            if (len < window_len)
            {
                if (!use_left_padding) continue;
                start = (len - window_len);
            }

            // Generate initial padded samples for sequences >= window_len
            if (use_left_padding && len >= window_len)
            {
                for (long pos = 1; pos < window_len; ++pos)
                {
                    matrix<int, 0, 1> window(window_len, 1);
                    long pad = window_len - pos;

                    for (long i = 0; i < pad; ++i) window(i) = padding_token;
                    for (long i = 0; i < pos; ++i) window(pad + i) = seq[i];

                    X.push_back(window);
                    Y.push_back(seq[pos]);
                }
            }

            // Slide window through sequence
            for (long pos = start; pos < len - 1; ++pos)
            {
                matrix<int, 0, 1> window(window_len, 1);

                for (long i = 0; i < window_len; ++i)
                {
                    long idx = pos + i;
                    window(i) = (idx >= 0 && idx < len) ? seq[idx] : padding_token;
                }

                long target_idx = pos + window_len;
                if (target_idx >= 0 && target_idx < len)
                {
                    X.push_back(window);
                    Y.push_back(seq[target_idx]);
                }
            }
        }
    }

    inline void build_multi_token_prediction_dataset(
        const std::vector<std::vector<int>>& source_sequences,
        const std::vector<std::vector<int>>& target_sequences,
        long src_window_len,
        long tgt_window_len,
        long padding_token,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<matrix<unsigned long, 0, 1>>& Y)
    {
        DLIB_CASSERT(source_sequences.size() == target_sequences.size(),
            "Source and target must have same size");

        X.clear();
        Y.clear();

        for (size_t i = 0; i < source_sequences.size(); ++i)
        {
            const auto& src = source_sequences[i];
            const auto& tgt = target_sequences[i];

            const long src_len = static_cast<long>(src.size());
            const long tgt_len = static_cast<long>(tgt.size());

            if (src_len == 0 || tgt_len == 0) continue;

            long src_pos = (src_len < src_window_len) ? (src_len - src_window_len) : 0;
            long tgt_pos = 0;

            while (true)
            {
                // Build source window
                matrix<int, 0, 1> src_window(src_window_len, 1);
                long src_real = 0;

                for (long j = 0; j < src_window_len; ++j)
                {
                    long idx = src_pos + j;
                    if (idx >= 0 && idx < src_len)
                    {
                        src_window(j) = src[idx];
                        src_real++;
                    }
                    else
                    {
                        src_window(j) = padding_token;
                    }
                }

                // Build target window
                matrix<unsigned long, 0, 1> tgt_window(tgt_window_len, 1);
                long tgt_real = 0;

                for (long j = 0; j < tgt_window_len; ++j)
                {
                    long idx = tgt_pos + j;
                    if (idx < tgt_len)
                    {
                        tgt_window(j) = tgt[idx];
                        tgt_real++;
                    }
                    else
                    {
                        tgt_window(j) = padding_token;
                    }
                }

                // Stop if no real tokens in either window
                if (src_real == 0 || tgt_real == 0) break;

                X.push_back(src_window);
                Y.push_back(tgt_window);

                // Stop if both sequences fully consumed
                if (src_pos + src_window_len >= src_len &&
                    tgt_pos + tgt_window_len >= tgt_len) break;

                src_pos++;
                tgt_pos++;
            }
        }
    }

    template <typename sample_type, typename label_type>
    void shuffle_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        unsigned long seed = 0)
    {
        DLIB_CASSERT(samples.size() == labels.size(),
            "samples and labels must have the same size");

        const size_t dataset_size = samples.size();
        if (dataset_size <= 1) return;

        dlib::rand rng;
        if (seed != 0) rng = dlib::rand(seed);

        // Fisher-Yates shuffle algorithm
        for (size_t i = dataset_size - 1; i > 0; --i)
        {
            size_t j = rng.get_random_32bit_number() % (i + 1);

            // Swap samples[i] with samples[j]
            std::swap(samples[i], samples[j]);

            // Swap labels[i] with labels[j]
            std::swap(labels[i], labels[j]);
        }
    }

    template <typename sample_type, typename label_type>
    void augment_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        int unk_token,
        int padding_token,
        double augmentation_ratio = 0.2,
        long min_noise_tokens = 1,
        long max_noise_tokens = 3,
        unsigned long seed = 0)
    {
        DLIB_CASSERT(samples.size() == labels.size(),
            "samples and labels must have the same size");
        DLIB_CASSERT(augmentation_ratio >= 0.0 && augmentation_ratio <= 2.0,
            "augmentation_ratio must be between 0.0 and 2.0");
        DLIB_CASSERT(min_noise_tokens >= 0 && max_noise_tokens >= min_noise_tokens,
            "Invalid noise token range: min=" << min_noise_tokens << ", max=" << max_noise_tokens);

        const size_t original_size = samples.size();
        if (original_size == 0 || augmentation_ratio == 0.0) return;

        // Calculate number of augmented samples to create
        const size_t num_augmented = static_cast<size_t>(original_size * augmentation_ratio);
        if (num_augmented == 0) return;

        // Reserve space to avoid multiple reallocations
        samples.reserve(original_size + num_augmented);
        labels.reserve(original_size + num_augmented);

        dlib::rand rng;
        if (seed != 0) rng = dlib::rand(seed);

        for (size_t aug_idx = 0; aug_idx < num_augmented; ++aug_idx)
        {
            // Select a random sample to augment
            const size_t source_idx = rng.get_random_32bit_number() % original_size;

            // Create a copy of the sample and its label
            auto augmented_sample = samples[source_idx];
            auto augmented_label = labels[source_idx];

            // Identify non-padding positions in the sample
            std::vector<long> valid_positions;
            const long sample_length = augmented_sample.nr();

            for (long i = 0; i < sample_length; ++i)
            {
                if (augmented_sample(i) != padding_token)
                    valid_positions.push_back(i);
            }

            // Skip if no valid positions to add noise
            if (valid_positions.empty()) continue;

            // Determine number of tokens to replace with noise
            const long num_valid = static_cast<long>(valid_positions.size());
            const long effective_max = std::min(max_noise_tokens, num_valid);
            const long effective_min = std::min(min_noise_tokens, effective_max);

            long num_noise = effective_min;
            if (effective_max > effective_min)
            {
                num_noise = effective_min +
                    (rng.get_random_32bit_number() % (effective_max - effective_min + 1));
            }

            // Ensure noise ratio is reasonable (max 30% of non-padding tokens)
            const long max_reasonable = std::max(1L, static_cast<long>(num_valid * 0.3));
            num_noise = std::min(num_noise, max_reasonable);

            // Randomly select positions to replace with UNK
            std::vector<long> noise_positions = valid_positions;

            // Fisher-Yates shuffle to select random positions
            for (long i = static_cast<long>(noise_positions.size()) - 1; i > 0; --i)
            {
                long j = rng.get_random_32bit_number() % (i + 1);
                std::swap(noise_positions[i], noise_positions[j]);
            }

            // Apply noise to the first num_noise positions
            for (long i = 0; i < num_noise; ++i)
            {
                augmented_sample(noise_positions[i]) = unk_token;
            }

            // Add augmented sample and label to the dataset
            samples.push_back(std::move(augmented_sample));
            labels.push_back(std::move(augmented_label));
        }
    }

} // namespace dlib

#endif // DLIB_LANGUAGE_MODEL_DATA_H_