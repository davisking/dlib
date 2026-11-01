// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FEATURE_HASHING_Hh_
#define DLIB_FEATURE_HASHING_Hh_

#include "feature_hashing_abstract.h"
#include "../algs.h"
#include "../general_hash/murmur_hash3.h"
#include "../sparse_vector.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename sample_type>
        void add_ngram (
            const char* data,
            std::size_t size,
            unsigned long num_buckets,
            sample_type& counts
        )
        {
            using index_type = typename sample_type::value_type::first_type;
            const auto bucket = murmur_hash3(data, static_cast<int>(size), 0) % num_buckets;
            counts.emplace_back(static_cast<index_type>(bucket), 1);
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type = std::vector<std::pair<unsigned long,double>>,
        typename string_type
        >
    sample_type hash_word_ngrams (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n,
        unsigned long num_buckets
    )
    {
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0,
            "\t sample_type hash_word_ngrams()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
        sample_type counts;
        std::string ngram;
        for (unsigned long n = min_n; n <= std::min<unsigned long>(max_n, tokens.size()); ++n)
        {
            for (unsigned long i = 0; i + n <= tokens.size(); ++i)
            {
                ngram.assign(tokens[i].data(), tokens[i].size());
                for (unsigned long j = i + 1; j < i + n; ++j)
                {
                    ngram += ' ';
                    ngram.append(tokens[j].data(), tokens[j].size());
                }
                impl::add_ngram(ngram.data(), ngram.size(), num_buckets, counts);
            }
        }
        make_sparse_vector_inplace(counts);
        return counts;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type = std::vector<std::pair<unsigned long,double>>,
        typename string_type
        >
    sample_type hash_char_ngrams (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n,
        unsigned long num_buckets
    )
    {
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0,
            "\t sample_type hash_char_ngrams()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
        sample_type counts;
        std::string padded;
        for (const auto& token : tokens)
        {
            padded.assign(1, ' ');
            padded.append(token.data(), token.size());
            padded += ' ';
            if (padded.size() < min_n)
            {
                impl::add_ngram(padded.data(), padded.size(), num_buckets, counts);
                continue;
            }
            for (unsigned long n = min_n; n <= std::min<unsigned long>(max_n, padded.size()); ++n)
            {
                for (unsigned long i = 0; i + n <= padded.size(); ++i)
                    impl::add_ngram(padded.data() + i, n, num_buckets, counts);
            }
        }
        make_sparse_vector_inplace(counts);
        return counts;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FEATURE_HASHING_Hh_
