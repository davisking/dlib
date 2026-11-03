// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FEATURE_HASHING_ABSTRACT_Hh_
#ifdef DLIB_FEATURE_HASHING_ABSTRACT_Hh_

#include "../general_hash/murmur_hash3_abstract.h"
#include "../error.h"
#include <string>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename string_type
        >
    std::vector<std::string> word_ngrams (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n
    );
    /*!
        requires
            - string_type provides data() and size() giving a const char* and the
              number of bytes in the token, e.g. std::string or std::string_view.
            - 1 <= min_n <= max_n
        ensures
            - Returns the word n-grams of a document.  An n-gram is n adjacent tokens
              joined by a single space, for each n in [min_n, max_n].
            - The n-grams are listed by increasing n, and for each n in the order they
              occur in tokens.  An n-gram that occurs more than once is listed each time.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename string_type
        >
    std::vector<std::string> char_ngrams (
        const std::vector<string_type>& tokens,
        unsigned long min_n,
        unsigned long max_n
    );
    /*!
        requires
            - The same as word_ngrams().
        ensures
            - Returns the character n-grams within each token.
            - Each token is padded with a space on both sides.  Its n-grams are the
              runs of n consecutive bytes of the padded token, for each n in
              [min_n, max_n] no longer than it.  A padded token shorter than min_n is
              one n-gram, whole.  This uses the boundary padding and short-token rule
              of scikit-learn's analyzer='char_wb', applied to bytes.  For UTF-8 text,
              n measures bytes and a window may split a multi-byte character.
            - The n-grams are listed token by token, and within a token by increasing
              n and then by position.  An n-gram that occurs more than once is listed
              each time.
    !*/

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
            - string_type satisfies the requirements of word_ngrams().
            - sample_type is a std::vector<std::pair<I,V>> where I is an unsigned
              integral type and V is an integral or floating point type. Neither is bool.
            - I can represent num_buckets-1.
            - V can represent the number of n-grams assigned to each bucket.
            - 1 <= min_n <= max_n
            - num_buckets > 0
        ensures
            - Counts the n-grams in word_ngrams(tokens,min_n,max_n), hashed into
              num_buckets buckets, without needing a vocabulary.  The result is
              suitable for tfidf_transformer or a sparse linear trainer.
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
            - Counts the n-grams in char_ngrams(tokens,min_n,max_n), hashed and
              returned as hash_word_ngrams() does.
        throws
            - The same as hash_word_ngrams().
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FEATURE_HASHING_ABSTRACT_Hh_
