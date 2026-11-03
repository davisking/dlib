// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FEATURE_HASHING_ABSTRACT_Hh_
#ifdef DLIB_FEATURE_HASHING_ABSTRACT_Hh_

#include "../general_hash/murmur_hash3_abstract.h"
#include "../error.h"
#include <string_view>
#include <utility>
#include <vector>

namespace dlib
{

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
    );
    /*!
        requires
            - string_type provides data() and size() giving a const char* and the
              number of bytes in the token, e.g. std::string or std::string_view.
            - sample_type is a std::vector<std::pair<I,V>> where I is an unsigned
              integral type and V is an integral or floating point type. Neither is bool.
            - I can represent num_buckets-1.
            - V can represent the number of n-grams assigned to each bucket.
            - 1 <= min_n <= max_n
            - num_buckets > 0
        ensures
            - Counts the word n-grams of a document, hashed into num_buckets buckets,
              without needing a vocabulary.  The result is suitable for
              tfidf_transformer or a sparse linear trainer.
            - An n-gram is n adjacent tokens joined by a single space, for each n in
              [min_n, max_n].
            - The n-gram s goes to bucket murmur_hash3(s.data(), s.size(), 0) % num_buckets.
            - Returns a sorted sparse vector: (b,c) means c n-grams went to bucket b.
        throws
            - dlib::error if an n-gram exceeds the length supported by murmur_hash3()
              (std::numeric_limits<int>::max() bytes).
    !*/

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
    );
    /*!
        requires
            - The same as hash_word_ngrams().
        ensures
            - Counts the character n-grams within each token, hashed and returned
              as hash_word_ngrams() does.
            - Each token is padded with a space on both sides.  Its n-grams are the
              runs of n consecutive bytes of the padded token, for each n in
              [min_n, max_n] no longer than it.  A padded token shorter than min_n is
              one n-gram, whole.  This uses the boundary padding and short-token rule
              of scikit-learn's analyzer='char_wb', applied to bytes.  For UTF-8 text,
              n measures bytes and a window may split a multi-byte character.
        throws
            - The same as hash_word_ngrams().
    !*/

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
    );
    /*!
        requires
            - string_type satisfies the requirements of hash_word_ngrams().
            - 1 <= min_n <= max_n
            - num_buckets > 0
            - f is callable as f(unsigned long bucket, std::string_view ngram).
        ensures
            - Calls f once for each n-gram that hash_word_ngrams(tokens,min_n,max_n,num_buckets)
              counts, with the n-gram's text and the bucket it goes to.  So the buckets f
              receives, counted, are that function's result.  This lets a caller find
              which n-grams a bucket stands for.
            - ngram is valid only during the call to f.
        throws
            - The same as hash_word_ngrams(), and any exception f throws.
    !*/

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
    );
    /*!
        requires
            - The same as for_each_word_ngram().
        ensures
            - for_each_word_ngram() for the n-grams that hash_char_ngrams() counts.
        throws
            - The same as for_each_word_ngram().
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FEATURE_HASHING_ABSTRACT_Hh_
