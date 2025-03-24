// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BPE_TOKENIZER_H
#define DLIB_BPE_TOKENIZER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <future>
#include <mutex>
#include <thread>
#include <map>
#include <unordered_map>
#include <queue>

#include "../base64.h"
#include "../serialize.h"
#include "bpe_tokenizer_abstract.h"

namespace dlib
{

    class bpe_tokenizer
    {
    public:
        bpe_tokenizer() : vocab_size(BASE_VOCAB_SIZE)
        {
            // Initialize the base vocabulary with single bytes
            for (int i = 0; i < BASE_VOCAB_SIZE; ++i)
                vocab[i] = std::vector<uint8_t>{ static_cast<uint8_t>(i) };
            
            // Initialize special tokens with sequential IDs
            special_tokens =
            {
                {"<text>",      BASE_VOCAB_SIZE},
                {"</text>",     BASE_VOCAB_SIZE + 1},
                {"<url>",       BASE_VOCAB_SIZE + 2},
                {"</url>",      BASE_VOCAB_SIZE + 3},
                {"<image>",     BASE_VOCAB_SIZE + 4},
                {"</image>",    BASE_VOCAB_SIZE + 5},
                {"<video>",     BASE_VOCAB_SIZE + 6},
                {"</video>",    BASE_VOCAB_SIZE + 7},
                {"<audio>",     BASE_VOCAB_SIZE + 8},
                {"</audio>",    BASE_VOCAB_SIZE + 9},
                {"<file>",      BASE_VOCAB_SIZE + 10},
                {"</file>",     BASE_VOCAB_SIZE + 11},
                {"<code>",      BASE_VOCAB_SIZE + 12},
                {"</code>",     BASE_VOCAB_SIZE + 13},
                {"<summary>",   BASE_VOCAB_SIZE + 14},
                {"</summary>",  BASE_VOCAB_SIZE + 15},
                {"<think>",     BASE_VOCAB_SIZE + 16},
                {"</think>",    BASE_VOCAB_SIZE + 17},
                {"<start>",     BASE_VOCAB_SIZE + 18},
                {"<end>",       BASE_VOCAB_SIZE + 19},
                {"<user>",      BASE_VOCAB_SIZE + 20},
                {"<bot>",       BASE_VOCAB_SIZE + 21},
                {"<system>",    BASE_VOCAB_SIZE + 22},
                {"<question>",  BASE_VOCAB_SIZE + 23},
                {"<answer>",    BASE_VOCAB_SIZE + 24},
                {"<search>",    BASE_VOCAB_SIZE + 25},
                {"<unk>",       BASE_VOCAB_SIZE + 26},
                {"<pad>",       BASE_VOCAB_SIZE + 27}
            };

            // Initialize the vector of special token IDs
            for (const auto& token : special_tokens)
                special_token_map[token.second] = token.first;
        }

        // Train the tokenizer on the given text
        void train(const std::string& text, int vocab_size, bool verbose = false)
        {
            DLIB_CASSERT(vocab_size >= BASE_VOCAB_SIZE);
            this->vocab_size = vocab_size;
            int num_merges = vocab_size - BASE_VOCAB_SIZE;

            // Convert text to byte IDs
            std::vector<int> ids;
            for (char c : text) ids.push_back(static_cast<uint8_t>(c));

            // Perform BPE merges
            for (int i = 0; i < num_merges; ++i) {
                auto stats = get_stats(ids);
                if (stats.empty()) break;

                // Find the most frequent pair that does not exceed MAX_TOKEN_LENGTH
                auto pair = get_most_frequent_pair(stats);

                // Check if the resulting token would exceed MAX_TOKEN_LENGTH
                size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
                if (new_token_length > MAX_TOKEN_LENGTH) {
                    if (verbose)
                    {
                        std::cout << "\r"
                            << std::setw(100) << std::flush
                            << "\rskipping merge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                            << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> new token length "
                            << std::to_string(new_token_length) << " exceeds limit of " << std::to_string(MAX_TOKEN_LENGTH)
                            << std::flush;
                    }
                    continue; // Skip this merge
                }

                int idx = (BASE_VOCAB_SIZE + (int)special_tokens.size()) + i;
                ids = merge(ids, pair, idx);
                merges[pair] = idx;
                vocab[idx].insert(vocab[idx].end(), vocab[pair.first].begin(), vocab[pair.first].end());
                vocab[idx].insert(vocab[idx].end(), vocab[pair.second].begin(), vocab[pair.second].end());

                if (verbose)
                {
                    std::cout << "\r"
                        << std::setw(100) << std::flush
                        << "\rmerge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                        << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> " << std::to_string(idx)
                        << " (" << bytes_to_string(vocab[idx]) << ") had "
                        << std::to_string(stats[pair]) << " occurrences"
                        << std::endl;
                }
            }
        }

        // Encode the given text into subword tokens
        std::vector<int> encode(const std::string& text) const
        {
            std::vector<int> result_ids;
            std::mutex result_mutex;

            // Split the text into paragraphs based on newline characters
            std::vector<std::string> paragraphs;
            size_t start = 0, end = text.find('\n');
            while (end != std::string::npos) {
                std::string paragraph = text.substr(start, end - start);
                if (!paragraph.empty()) paragraphs.push_back(paragraph);
                start = end + 1;
                end = text.find('\n', start);
            }
            // Add the last paragraph (if any) and only if it's not empty
            if (start < text.size()) {
                std::string paragraph = text.substr(start);
                if (!paragraph.empty()) paragraphs.push_back(paragraph);
            }

            // Function to encode a single paragraph
            auto encode_paragraph = [this](const std::string& paragraph) -> std::vector<int> {
                std::vector<int> ids;
                ids.reserve(paragraph.size());
                for (char c : paragraph) ids.push_back(static_cast<uint8_t>(c));

                auto stats = get_stats(ids);
                std::priority_queue<std::pair<int, std::pair<int, int>>> pq;
                for (const auto& stat : stats) {
                    const std::pair<int, int>& pair = stat.first;
                    if (merges.count(pair)) pq.push({ merges.at(pair), pair });
                }

                while (!pq.empty()) {
                    const auto& top_element = pq.top();
                    const std::pair<int, int>& pair = top_element.second;
                    pq.pop();

                    bool pair_found = false;
                    for (size_t i = 0; i < ids.size() - 1; ++i) {
                        if (ids[i] == pair.first && ids[i + 1] == pair.second) {
                            pair_found = true;
                            break;
                        }
                    }
                    if (!pair_found) continue;

                    int idx = merges.at(pair);
                    ids = merge(ids, pair, idx);

                    stats = get_stats(ids);
                    for (const auto& stat : stats) {
                        const std::pair<int, int>& new_pair = stat.first;
                        if (merges.count(new_pair)) pq.push({ merges.at(new_pair), new_pair });
                    }
                }

                return ids;
            };

            // Special case: if there's only one paragraph, no need for threads
            int sot_tok = get_special_token_id("<text>");
            int eot_tok = get_special_token_id("</text>");
            if (paragraphs.size() == 1) {
                std::vector<int> paragraph_ids = encode_paragraph(paragraphs[0]);
                result_ids.push_back(sot_tok);
                result_ids.insert(result_ids.end(), paragraph_ids.begin(), paragraph_ids.end());
                result_ids.push_back(eot_tok);
                return result_ids;
            }

            // Launch encoding tasks in parallel for multiple paragraphs
            std::vector<std::future<std::vector<int>>> futures;
            for (const auto& paragraph : paragraphs)
                futures.push_back(std::async(std::launch::async, encode_paragraph, paragraph));

            // Collect results in order
            for (auto& future : futures) {
                std::vector<int> paragraph_ids = future.get();
                std::lock_guard<std::mutex> lock(result_mutex);
                result_ids.push_back(sot_tok);
                result_ids.insert(result_ids.end(), paragraph_ids.begin(), paragraph_ids.end());
                result_ids.push_back(eot_tok);
            }
            return result_ids;
        }

        // Decode a single token ID back into text
        std::string decode(int id, bool display_special_tokens = true) const
        {
            return decode(std::vector<int>({ id }), display_special_tokens);
        }

        // Decode a sequence of token IDs back into text
        std::string decode(const std::vector<int>& ids, bool display_special_tokens = true) const
        {
            std::vector<uint8_t> bytes;
            int vocab_size = static_cast<int>(get_vocab_size());
            for (int id : ids)
            {
                if (id < vocab_size)
                {
                    // Check if the ID is a special token
                    auto it = special_token_map.find(id);
                    if (it != special_token_map.end())
                    {
                        // It's a special token, get the corresponding string
                        if (display_special_tokens) bytes.insert(bytes.end(), it->second.begin(), it->second.end());
                    }
                    else
                    {
                        // It's a regular token, get the bytes from the vocabulary
                        auto& token = vocab.at(id);
                        bytes.insert(bytes.end(), token.begin(), token.end());
                    }
                }
            }
            return std::string(bytes.begin(), bytes.end());
        }

        // Save the tokenizer model and vocabulary to file
        friend void serialize(const bpe_tokenizer& tok, std::ostream& out)
        {
            serialize("bpe_tokenizer2_", out);
            serialize(tok.special_tokens, out);
            serialize(tok.special_token_map, out);
            serialize(tok.merges, out);
            serialize(tok.vocab, out);
            serialize(tok.vocab_size, out);
        }

        // Load the tokenizer model and vocabulary from file
        friend void deserialize(bpe_tokenizer& tok, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "bpe_tokenizer2_")
                throw dlib::serialization_error("Unexpected version '" + version + "' found while deserializing dlib::bpe_tokenizer_.");
            deserialize(tok.special_tokens, in);
            deserialize(tok.special_token_map, in);
            deserialize(tok.merges, in);
            deserialize(tok.vocab, in);
            deserialize(tok.vocab_size, in);
        }

        // Get the ID of a special token
        int get_special_token_id(const std::string& token) const
        {
            auto it = special_tokens.find(token);
            if (it != special_tokens.end()) return it->second;
            throw std::runtime_error("Special token not found: " + token);
        }

        // Get the total vocabulary size
        size_t get_vocab_size() const
        {
            return (vocab.size() + special_tokens.size());
        }

    private:
        std::map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> special_token_map;
        std::map<std::pair<int, int>, int> merges;
        std::map<int, std::vector<uint8_t>> vocab;
        int vocab_size;

        static const size_t MAX_TOKEN_LENGTH = 8;
        static const int BASE_VOCAB_SIZE = 256;

        // Get frequency statistics of adjacent token pairs
        struct pair_hash {
            template <class T1, class T2>
            std::size_t operator()(const std::pair<T1, T2>& p) const
            {
                auto hash1 = std::hash<T1>{}(p.first);
                auto hash2 = std::hash<T2>{}(p.second);
                return hash1 ^ (hash2 << 1);
            }
        };
        std::unordered_map<std::pair<int, int>, int, pair_hash> get_stats(const std::vector<int>& ids) const
        {
            std::unordered_map<std::pair<int, int>, int, pair_hash> global_stats;
            std::mutex global_stats_mutex;

            auto worker = [&](size_t start, size_t end) {
                std::unordered_map<std::pair<int, int>, int, pair_hash> local_stats;
                for (size_t i = start; i < end - 1 && i + 1 < ids.size(); ++i)
                    local_stats[{ids[i], ids[i + 1]}]++;

                std::lock_guard<std::mutex> lock(global_stats_mutex);
                for (const auto& pair : local_stats)
                    global_stats[pair.first] += pair.second;
            };

            size_t num_threads = std::thread::hardware_concurrency();
            size_t segment_size = ids.size() / num_threads;
            std::vector<std::thread> threads;

            for (size_t t = 0; t < num_threads; ++t)
            {
                size_t start = t * segment_size;
                size_t end = (t == num_threads - 1) ? ids.size() : start + segment_size;
                threads.emplace_back(worker, start, end);
            }

            for (auto& thread : threads) thread.join();

            return global_stats;
        }

        // Finds the most frequent pair of tokens in the given statistics map that does not exceed the maximum token length
        std::pair<int, int> get_most_frequent_pair(const std::unordered_map<std::pair<int, int>, int, pair_hash>& stats) const 
        {
            std::pair<int, int> best_pair = { -1, -1 }; // Initialize the best pair to an invalid value
            double max_score = 0; // Initialize the maximum score to 0

            // Iterate over all pairs in the statistics map
            for (const auto& stat : stats) {
                const std::pair<int, int>& pair = stat.first; // Extract the token pair
                int count = stat.second; // Extract the frequency count

                // Check if the new token formed by merging the pair would exceed the maximum allowed length
                size_t new_token_length = vocab.at(pair.first).size() + vocab.at(pair.second).size();
                if (new_token_length > MAX_TOKEN_LENGTH) continue; // Skip this pair if it exceeds the maximum token length

                // Calculate the score for this pair (frequency * length_penalty)
                double score = (size_t)count * (new_token_length > (MAX_TOKEN_LENGTH / 2) ? 1.75 : 1.0);

                // Update the best pair if the current pair has a higher score
                if (score > max_score)
                {
                    best_pair = pair;
                    max_score = score;
                }
            }

            return best_pair; // Return the pair with the highest score
        }

        // Merge the most frequent pair in the token sequence
        std::vector<int> merge(std::vector<int>& ids, const std::pair<int, int>& pair, int idx) const
        {
            std::vector<int> new_ids;
            new_ids.reserve(ids.size()); // Reserve space to avoid reallocations

            for (size_t i = 0; i < ids.size(); ++i)
            {
                if (i < ids.size() - 1 && ids[i] == pair.first && ids[i + 1] == pair.second)
                {
                    new_ids.push_back(idx); // Replace the pair with the new token ID
                    i++; // Skip the next token
                }
                else new_ids.push_back(ids[i]); // Keep the current token
            }

            return new_ids;
        }

        static std::string base64_encode(const std::string& input) {
            dlib::base64 encoder;
            std::istringstream sin(input);
            std::ostringstream sout;
            encoder.encode(sin, sout);
            return sout.str();
        }

        // Convert a sequence of bytes to a readable string
        static std::string bytes_to_string(const std::vector<uint8_t>& bytes)
        {
            std::string data(bytes.begin(), bytes.end());
            return base64_encode(data);
        }

    };

}


#endif // DLIB_BPE_TOKENIZER_H
