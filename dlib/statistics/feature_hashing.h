// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FEATURE_HASHING_Hh_
#define DLIB_FEATURE_HASHING_Hh_

#include "feature_hashing_abstract.h"
#include "../algs.h"
#include "../error.h"
#include "../general_hash/murmur_hash3.h"
#include "../sparse_vector.h"
#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline unsigned long ngram_bucket (
            const char* data,
            std::size_t size,
            unsigned long num_buckets
        )
        /*!
            requires
                - data points to size bytes.
                - num_buckets > 0
            ensures
                - Returns murmur_hash3(data,size,0) % num_buckets.
            throws
                - dlib::error if size exceeds the length supported by murmur_hash3().
        !*/
        {
            if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw error("N-gram length exceeds the length supported by murmur_hash3().");
            return murmur_hash3(data, static_cast<int>(size), 0) % num_buckets;
        }

        template <typename sample_type>
        void check_index_type ()
        {
            using index_type = typename sample_type::value_type::first_type;
            static_assert(std::is_integral_v<index_type> && std::is_unsigned_v<index_type> &&
                          !std::is_same_v<index_type,bool>, "N-gram indices must be unsigned integers.");
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename string_type,
        typename callback_type
        >
    void for_each_word_ngram (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n,
        unsigned long num_buckets,
        callback_type&& f
    )
    {
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0,
            "\t void for_each_word_ngram()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
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
                f(impl::ngram_bucket(ngram.data(), ngram.size(), num_buckets),
                  std::string_view(ngram));
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename string_type,
        typename callback_type
        >
    void for_each_char_ngram (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n,
        unsigned long num_buckets,
        callback_type&& f
    )
    {
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0,
            "\t void for_each_char_ngram()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
        std::string padded;
        for (const auto& token : tokens)
        {
            padded.assign(1, ' ');
            padded.append(token.data(), token.size());
            padded += ' ';
            if (padded.size() < min_n)
            {
                f(impl::ngram_bucket(padded.data(), padded.size(), num_buckets),
                  std::string_view(padded));
                continue;
            }
            for (unsigned long n = min_n; n <= std::min<unsigned long>(max_n, padded.size()); ++n)
            {
                for (unsigned long i = 0; i + n <= padded.size(); ++i)
                {
                    f(impl::ngram_bucket(padded.data() + i, n, num_buckets),
                      std::string_view(padded.data() + i, n));
                }
            }
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
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0 &&
                    num_buckets-1 <= std::numeric_limits<typename sample_type::value_type::first_type>::max(),
            "\t sample_type hash_word_ngrams()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
        impl::check_index_type<sample_type>();
        using index_type = typename sample_type::value_type::first_type;
        sample_type counts;
        for_each_word_ngram(tokens, min_n, max_n, num_buckets, [&](unsigned long b, std::string_view)
                            { counts.emplace_back(static_cast<index_type>(b), 1); });
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
        DLIB_ASSERT(1 <= min_n && min_n <= max_n && num_buckets > 0 &&
                    num_buckets-1 <= std::numeric_limits<typename sample_type::value_type::first_type>::max(),
            "\t sample_type hash_char_ngrams()"
            << "\n\t Invalid arguments were given to this function."
            << "\n\t min_n:       " << min_n
            << "\n\t max_n:       " << max_n
            << "\n\t num_buckets: " << num_buckets
        );
        impl::check_index_type<sample_type>();
        using index_type = typename sample_type::value_type::first_type;
        sample_type counts;
        for_each_char_ngram(tokens, min_n, max_n, num_buckets, [&](unsigned long b, std::string_view)
                            { counts.emplace_back(static_cast<index_type>(b), 1); });
        make_sparse_vector_inplace(counts);
        return counts;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FEATURE_HASHING_Hh_
