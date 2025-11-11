#ifndef DLIB_LANGUAGE_MODEL_DATA_H_
#define DLIB_LANGUAGE_MODEL_DATA_H_

#include "language_model_data_abstract.h"

#include <iostream>
#include <string>
#include <vector>
#include <dlib/matrix.h>
#include <dlib/serialize.h>
//#include "../matrix.h"
//#include "../serialize.h"

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

        dlib::matrix<int, 0, 1> get_input_window(long custom_window_size = -1) const
        {
            long win_size = (custom_window_size > 0) ? custom_window_size : window_size_;
            dlib::matrix<int, 0, 1> window(win_size, 1);

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
        std::vector<dlib::matrix<int, 0, 1>>& X,
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
                    dlib::matrix<int, 0, 1> window(window_len, 1);
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
                dlib::matrix<int, 0, 1> window(window_len, 1);

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
        std::vector<dlib::matrix<int, 0, 1>>& X,
        std::vector<dlib::matrix<unsigned long, 0, 1>>& Y)
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
                dlib::matrix<int, 0, 1> src_window(src_window_len, 1);
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
                dlib::matrix<unsigned long, 0, 1> tgt_window(tgt_window_len, 1);
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
} // namespace dlib

#endif // DLIB_LANGUAGE_MODEL_DATA_H_