// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BPE_TOKENIZER_H
#define DLIB_BPE_TOKENIZER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>
#include <sstream>
#include <list>

#include "../base64.h"
#include "../serialize.h"
#include "bpe_tokenizer_abstract.h"

namespace dlib
{

    class bpe_tokenizer
    {
    public:
        bpe_tokenizer()
        {
            vocab_size = BPE_BASE_VOCAB_SIZE + special_token_list.size();

            // Initialize base vocabulary (bytes 0-255)
            for (int i = 0; i < BPE_BASE_VOCAB_SIZE; ++i) {
                Merge m;
                m.token_id = i;
                m.left = i;
                m.right = i;
                m.pattern.push_back(static_cast<uint8_t>(i));
                merges.push_back(m);
            }

            // Initialize special tokens
            initialize_special_tokens();
        }

        // Train the tokenizer on input data
        void train(const std::string& text, size_t max_vocab_size, size_t max_bytes = 0, bool verbose = false)
        {
            if (text.empty()) return;

            // Convert text to bytes
            std::vector<uint8_t> data(text.begin(), text.end());
            if (max_bytes > 0 && data.size() > max_bytes) data.resize(max_bytes);

            // Calculate available merges (reserving space for special tokens)
            size_t num_merges = max_vocab_size - BPE_BASE_VOCAB_SIZE - special_token_list.size();

            if (num_merges <= 0) {
                if (verbose) {
                    std::cout << "Warning: max_vocab_size too small for any merges. Need at least "
                        << (BPE_BASE_VOCAB_SIZE + special_token_list.size() + 1) << " tokens." << std::endl;
                }
                return;
            }

            if (verbose) {
                std::cout << "Training BPE tokenizer on " << data.size() << " bytes..." << std::endl;
                std::cout << "Target vocabulary size: " << max_vocab_size << std::endl;
                std::cout << "Base vocabulary: " << BPE_BASE_VOCAB_SIZE << " tokens" << std::endl;
                std::cout << "Special tokens: " << special_token_list.size() << " tokens" << std::endl;
                std::cout << "Available merges: " << num_merges << std::endl;
            }

            // Reset merges beyond base vocabulary
            merges.clear();
            for (int i = 0; i < BPE_BASE_VOCAB_SIZE; ++i) {
                Merge m;
                m.token_id = i;
                m.left = i;
                m.right = i;
                m.pattern.push_back(static_cast<uint8_t>(i));
                merges.push_back(m);
            }

            // Tokenize input into segments (split on whitespace and newlines)
            std::vector<std::list<uint16_t>> segments;
            std::vector<uint32_t> segment_counts;
            tokenize(data, segments, segment_counts);

            if (verbose) {
                std::cout << "Created " << segments.size() << " segments for training" << std::endl;
            }

            // Initialize pair counting
            std::unordered_map<uint32_t, std::pair<int32_t, std::pair<uint16_t, uint16_t>>> pair_counts;
            std::unordered_map<uint32_t, std::unordered_set<uint32_t>> where_to_update;

            // Count initial pairs
            for (uint32_t i = 0; i < segments.size(); i++) {
                countInSegment(segments[i], i, segment_counts[i], pair_counts, where_to_update);
            }

            // Main training loop
            size_t merges_performed = 0;

            for (size_t merge_idx = 0; merge_idx < num_merges; merge_idx++) {
                // Find most frequent pair
                int32_t max_count = 0;
                std::pair<uint16_t, uint16_t> max_pair = findMaxPair(pair_counts, max_count);

                if (max_count <= 0) {
                    if (verbose) {
                        std::cout << "\nNo more pairs to merge at iteration " << merge_idx << std::endl;
                    }
                    break;
                }

                uint16_t new_token = BPE_BASE_VOCAB_SIZE + merge_idx;

                // Create merge entry
                Merge m;
                m.token_id = new_token;
                m.left = max_pair.first;
                m.right = max_pair.second;

                // Build pattern for new token
                m.pattern = merges[max_pair.first].pattern;
                const auto& right_pattern = merges[max_pair.second].pattern;
                m.pattern.insert(m.pattern.end(), right_pattern.begin(), right_pattern.end());

                merges.push_back(m);

                // Store mapping for fast encoding
                uint32_t pair_key = (static_cast<uint32_t>(max_pair.first) << 16) | max_pair.second;

                if (verbose && (merge_idx % 1000 == 0 || merge_idx < 10)) {
                    std::cout << "Merge " << merge_idx << ": (" << max_pair.first << ", " << max_pair.second
                        << ") -> " << new_token << " (occurrences: " << max_count
                        << ", pattern length: " << m.pattern.size() << ")" << std::endl;
                }

                // Apply merge to all affected segments
                auto affected_segments = where_to_update[pair_key];
                applyMerge(max_pair, new_token, segments, segment_counts, affected_segments,
                    pair_counts, where_to_update);

                // Clear this pair's count
                pair_counts[pair_key].first = 0;

                merges_performed++;
            }

            // Update vocabulary size: base + special tokens + actual merges performed
            vocab_size = merges.size() + special_token_list.size();
            initialize_special_tokens();

            if (verbose) {
                std::cout << "\nTraining complete!" << std::endl;
                std::cout << "Base vocabulary: " << BPE_BASE_VOCAB_SIZE << " tokens" << std::endl;
                std::cout << "Merges performed: " << merges_performed << std::endl;
                std::cout << "Special tokens: " << special_token_list.size() << " tokens" << std::endl;
                std::cout << "Total vocabulary size: " << vocab_size << std::endl;
            }
        }

        // Encode text into tokens
        std::vector<int> encode(const std::string& text) const
        {
            if (text.empty()) return {};

            // Convert to initial tokens
            std::vector<int> tokens;
            tokens.reserve(text.size());
            for (unsigned char byte : text) {
                tokens.push_back(static_cast<int>(byte));
            }

            // Apply all merges in order
            for (size_t merge_idx = BPE_BASE_VOCAB_SIZE; merge_idx < merges.size(); merge_idx++) {
                const Merge& m = merges[merge_idx];

                std::vector<int> new_tokens;
                new_tokens.reserve(tokens.size());

                for (size_t i = 0; i < tokens.size(); ) {
                    if (i < tokens.size() - 1 && tokens[i] == m.left && tokens[i + 1] == m.right) {
                        new_tokens.push_back(m.token_id);
                        i += 2;
                    }
                    else {
                        new_tokens.push_back(tokens[i]);
                        i++;
                    }
                }

                tokens = std::move(new_tokens);
            }

            return tokens;
        }

        // Decode tokens back to text
        std::string decode(const std::vector<int>& tokens, bool display_special_tokens = true) const
        {
            std::vector<uint8_t> result;
            result.reserve(tokens.size() * 4); // Estimate

            for (int token : tokens) {
                if (token >= 0 && token < static_cast<int>(merges.size())) {
                    // Base token or merge token
                    const std::vector<uint8_t>& pattern = merges[token].pattern;
                    result.insert(result.end(), pattern.begin(), pattern.end());
                }
                else if (token >= static_cast<int>(merges.size()) &&
                    token < static_cast<int>(get_vocab_size())) {
                    // Special token
                    if (display_special_tokens) {
                        auto it = special_token_map.find(token);
                        if (it != special_token_map.end()) {
                            const std::string& special_str = it->second;
                            result.insert(result.end(), special_str.begin(), special_str.end());
                        }
                    }
                }
                else {
                    const std::string tok_unk = "<unk>";
                    result.insert(result.end(), tok_unk.begin(), tok_unk.end());
                }
            }

            return std::string(result.begin(), result.end());
        }

        // Decode single token (for compatibility)
        std::string decode(int token, bool display_special_tokens = true) const
        {
            std::vector<int> tokens = { token };
            return decode(tokens, display_special_tokens);
        }

        // Get special token ID
        int get_special_token_id(const std::string& token) const
        {
            auto it = special_tokens.find(token);
            if (it != special_tokens.end()) {
                // Special tokens come after base vocab and merges
                return it->second;
            }
            throw std::runtime_error("Special token not found: " + token);
        }

        // Get vocabulary size
        size_t get_specials_size() const { return special_token_list.size(); }
        size_t get_vocab_size() const { return vocab_size; }
        size_t get_vocab_without_specials_size() const { return (vocab_size - special_token_list.size()); }

        // Serialization
        friend void serialize(const bpe_tokenizer& item, std::ostream& out)
        {
            serialize("bpe_tokenizer_", out);
            serialize(item.vocab_size, out);

            // Serialize only the merge entries beyond base vocabulary
            size_t num_merges = item.merges.size() - bpe_tokenizer::BPE_BASE_VOCAB_SIZE;
            serialize(num_merges, out);

            for (size_t i = bpe_tokenizer::BPE_BASE_VOCAB_SIZE; i < item.merges.size(); ++i) {
                const auto& m = item.merges[i];
                serialize(m.left, out);
                serialize(m.right, out);
                serialize(m.pattern, out);
            }
        }

        // Deserialization
        friend void deserialize(bpe_tokenizer& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "bpe_tokenizer_")
                throw serialization_error("Unexpected version found while deserializing dlib::bpe_tokenizer.");
            deserialize(item.vocab_size, in);

            // Initialize base vocabulary
            item.merges.clear();
            for (int i = 0; i < bpe_tokenizer::BPE_BASE_VOCAB_SIZE; ++i) {
                Merge m;
                m.token_id = i;
                m.left = i;
                m.right = i;
                m.pattern.push_back(static_cast<uint8_t>(i));
                item.merges.push_back(m);
            }

            // Deserialize merge entries
            size_t num_merges;
            deserialize(num_merges, in);
            for (size_t i = 0; i < num_merges; ++i) {
                Merge m;
                m.token_id = bpe_tokenizer::BPE_BASE_VOCAB_SIZE + i;
                deserialize(m.left, in);
                deserialize(m.right, in);
                deserialize(m.pattern, in);
                item.merges.push_back(m);
            }

            // Initialize special tokens
            item.initialize_special_tokens();
        }

    private:
        // Define special tokens
        const std::vector<std::string> special_token_list = {
            "<text>", "</text>", "<url>", "</url>",
            "<image>", "</image>", "<video>", "</video>",
            "<audio>", "</audio>", "<file>", "</file>",
            "<code>", "</code>", "<summary>", "</summary>",
            "<think>", "</think>", "<start>", "<end>",
            "<user>", "<bot>", "<system>", "<question>",
            "<answer>", "<search>", "<unk>", "<pad>"
        };
        static const int BPE_BASE_VOCAB_SIZE = 256;

        // Merge structure
        struct Merge {
            int token_id;
            int left;
            int right;
            std::vector<uint8_t> pattern;
        };

        // Data members
        size_t vocab_size;
        std::vector<Merge> merges;

        // Special tokens handling
        std::map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> special_token_map;

        void initialize_special_tokens()
        {
            special_tokens.clear();
            special_token_map.clear();

            // Initialize special tokens with sequential IDs
            int next_id = get_vocab_without_specials_size();

            for (const auto& token : special_token_list) {
                special_tokens[token] = next_id;
                special_token_map[next_id] = token;
                next_id++;
            }
        }

        // Segment-based training functions from BPETokenizer
        void tokenize(const std::vector<uint8_t>& data,
            std::vector<std::list<uint16_t>>& segments,
            std::vector<uint32_t>& segment_counts)
        {
            segments.clear();
            segment_counts.clear();

            // Split on whitespace and newlines to create meaningful segments
            std::list<uint16_t> current_segment;

            for (size_t i = 0; i < data.size(); i++) {
                uint8_t byte = data[i];

                // Split on whitespace and newlines
                if (byte == ' ' || byte == '\n' || byte == '\t' || byte == '\r') {
                    if (!current_segment.empty()) {
                        segments.push_back(current_segment);
                        segment_counts.push_back(1);
                        current_segment.clear();
                    }
                    // Add the delimiter as its own segment
                    segments.push_back({ static_cast<uint16_t>(byte) });
                    segment_counts.push_back(1);
                }
                else {
                    current_segment.push_back(static_cast<uint16_t>(byte));
                }
            }

            // Add the last segment if not empty
            if (!current_segment.empty()) {
                segments.push_back(current_segment);
                segment_counts.push_back(1);
            }
        }

        // Count pairs in a segment and update global pair counts
        void countInSegment(const std::list<uint16_t>& segment, uint32_t segment_idx,
            int32_t count_delta,
            std::unordered_map<uint32_t, std::pair<int32_t, std::pair<uint16_t, uint16_t>>>& pair_counts,
            std::unordered_map<uint32_t, std::unordered_set<uint32_t>>& where_to_update)
        {
            if (segment.size() < 2) return;

            auto it = segment.begin();
            uint16_t prev_token = *it;
            ++it;

            while (it != segment.end()) {
                uint16_t curr_token = *it;
                uint32_t pair_key = (static_cast<uint32_t>(prev_token) << 16) | curr_token;

                auto& entry = pair_counts[pair_key];
                entry.first += count_delta;
                entry.second = { prev_token, curr_token };

                if (count_delta > 0) {
                    where_to_update[pair_key].insert(segment_idx);
                }
                else {
                    where_to_update[pair_key].erase(segment_idx);
                }

                prev_token = curr_token;
                ++it;
            }
        }

        // Find the most frequent pair
        std::pair<uint16_t, uint16_t> findMaxPair(
            const std::unordered_map<uint32_t, std::pair<int32_t, std::pair<uint16_t, uint16_t>>>& pair_counts,
            int32_t& max_count)
        {
            max_count = 0;
            std::pair<uint16_t, uint16_t> max_pair = { 0, 0 };

            for (const auto& entry : pair_counts) {
                int32_t count = entry.second.first;
                const auto& pair = entry.second.second;

                if (count > max_count || (count == max_count && pair < max_pair)) {
                    max_count = count;
                    max_pair = pair;
                }
            }

            return max_pair;
        }

        // Apply a merge to all affected segments
        void applyMerge(const std::pair<uint16_t, uint16_t>& merge_pair, uint16_t new_token_id,
            std::vector<std::list<uint16_t>>& segments,
            const std::vector<uint32_t>& segment_counts,
            const std::unordered_set<uint32_t>& affected_segments,
            std::unordered_map<uint32_t, std::pair<int32_t, std::pair<uint16_t, uint16_t>>>& pair_counts,
            std::unordered_map<uint32_t, std::unordered_set<uint32_t>>& where_to_update)
        {
            for (uint32_t segment_idx : affected_segments) {
                auto& segment = segments[segment_idx];
                int32_t count = segment_counts[segment_idx];

                // Remove old pair counts for this segment
                countInSegment(segment, segment_idx, -count, pair_counts, where_to_update);

                // Apply merge in this segment
                auto it = segment.begin();
                while (it != segment.end()) {
                    auto next_it = std::next(it);
                    if (next_it != segment.end() && *it == merge_pair.first && *next_it == merge_pair.second) {
                        *it = new_token_id;
                        segment.erase(next_it);
                        // Don't increment it, check the same position again
                    }
                    else {
                        ++it;
                    }
                }

                // Add new pair counts for this segment
                countInSegment(segment, segment_idx, count, pair_counts, where_to_update);
            }
        }
    };

}

#endif // DLIB_BPE_TOKENIZER_H