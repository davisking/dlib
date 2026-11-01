// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/statistics.h>
#include <dlib/sparse_vector.h>
#include "tester.h"
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.feature_hashing");

    using sparse_type = std::vector<std::pair<unsigned long,double>>;

    unsigned long bucket (const std::string& s, unsigned long num_buckets)
    {
        return murmur_hash3(s.data(), s.size(), 0) % num_buckets;
    }

    // The expected counts, from the n-grams written out by hand.
    sparse_type expected (const std::vector<std::string>& ngrams, unsigned long num_buckets)
    {
        std::map<unsigned long,double> counts;
        for (const auto& s : ngrams)
            counts[bucket(s, num_buckets)] += 1;
        return sparse_type(counts.begin(), counts.end());
    }

// ----------------------------------------------------------------------------------------

    void test_word_ngrams()
    {
        const unsigned long num_buckets = 1 << 20;
        const std::vector<std::string> tokens = {"a", "b", "a", "b"};
        DLIB_TEST(hash_word_ngrams(tokens, 1, 2, num_buckets) ==
                  expected({"a", "b", "a", "b", "a b", "b a", "a b"}, num_buckets));
        DLIB_TEST(hash_word_ngrams(tokens, 3, 3, num_buckets) ==
                  expected({"a b a", "b a b"}, num_buckets));
        DLIB_TEST(hash_word_ngrams(tokens, 5, 6, num_buckets).empty());
        DLIB_TEST(hash_word_ngrams(std::vector<std::string>(), 1, 2, num_buckets).empty());
    }

// ----------------------------------------------------------------------------------------

    void test_char_ngrams()
    {
        const unsigned long num_buckets = 1 << 20;

        // " ab " has two 3-grams, is itself the only 4-gram, and is too short for 5.
        DLIB_TEST(hash_char_ngrams(std::vector<std::string>{"ab"}, 3, 5, num_buckets) ==
                  expected({" ab", "ab ", " ab "}, num_buckets));

        // " a " is shorter than min_n, so it counts once, whole.
        DLIB_TEST(hash_char_ngrams(std::vector<std::string>{"a"}, 4, 5, num_buckets) ==
                  expected({" a "}, num_buckets));

        // N-grams stay inside their token.
        DLIB_TEST(hash_char_ngrams(std::vector<std::string>{"abc", "de"}, 3, 3, num_buckets) ==
                  expected({" ab", "abc", "bc ", " de", "de "}, num_buckets));

        DLIB_TEST(hash_char_ngrams(std::vector<std::string>{"abcd"}, 2, 3, num_buckets) ==
                  expected({" a", "ab", "bc", "cd", "d ", " ab", "abc", "bcd", "cd "}, num_buckets));

        DLIB_TEST(hash_char_ngrams(std::vector<std::string>(), 3, 5, num_buckets).empty());
    }

// ----------------------------------------------------------------------------------------

    void test_collisions()
    {
        // With one bucket, every n-gram lands in bucket 0.
        DLIB_TEST(hash_word_ngrams(std::vector<std::string>{"a", "b", "c"}, 1, 2, 1) ==
                  sparse_type({{0, 5}}));
        DLIB_TEST(hash_char_ngrams(std::vector<std::string>{"ab"}, 3, 5, 1) ==
                  sparse_type({{0, 3}}));
    }

// ----------------------------------------------------------------------------------------

    void test_types()
    {
        const unsigned long num_buckets = 1000;
        const std::vector<std::string> words = {"to", "be", "or", "not", "to", "be"};
        const std::vector<std::string_view> views(words.begin(), words.end());

        DLIB_TEST(hash_word_ngrams(words, 1, 2, num_buckets) ==
                  hash_word_ngrams(views, 1, 2, num_buckets));
        const auto from_strings = hash_char_ngrams(words, 2, 4, num_buckets);
        DLIB_TEST(from_strings == hash_char_ngrams(views, 2, 4, num_buckets));

        using integer_type = std::vector<std::pair<unsigned int,unsigned int>>;
        const auto integer_counts = hash_char_ngrams<integer_type>(words, 2, 4, num_buckets);
        DLIB_TEST(integer_counts.size() == from_strings.size());
        for (unsigned long i = 0; i < integer_counts.size(); ++i)
        {
            DLIB_TEST(integer_counts[i].first == from_strings[i].first);
            DLIB_TEST(integer_counts[i].second == from_strings[i].second);
        }
    }

// ----------------------------------------------------------------------------------------

    void test_with_tfidf()
    {
        // The intended use: hashed counts in, TF-IDF weighted vectors out.
        const std::vector<std::vector<std::string>> documents = {
            {"the", "cat", "sat"},
            {"the", "dog", "sat"},
            {"the", "cat", "ran"}
        };
        std::vector<sparse_type> counts;
        for (const auto& document : documents)
            counts.push_back(hash_word_ngrams(document, 1, 1, 1 << 20));

        tfidf_transformer<sparse_type> tfidf;
        tfidf.train(counts);
        DLIB_TEST(tfidf.vocabulary_size() == 5);
        DLIB_TEST(std::abs(length(tfidf(counts[0])) - 1) < 1e-12);
    }

// ----------------------------------------------------------------------------------------

    class feature_hashing_tester : public tester
    {
    public:
        feature_hashing_tester (
        ) :
            tester (
                "test_feature_hashing",
                "Run tests on hash_word_ngrams() and hash_char_ngrams().",
                0
            )
        {}

        void perform_test (
        )
        {
            test_word_ngrams();
            test_char_ngrams();
            test_collisions();
            test_types();
            test_with_tfidf();
        }
    } a;

}
