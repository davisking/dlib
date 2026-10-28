// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/statistics.h>
#include <dlib/sparse_vector.h>
#include "tester.h"
#include <cmath>
#include <sstream>
#include <type_traits>
#include <vector>

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.tfidf");

    using sparse_type = std::vector<std::pair<unsigned long,double>>;

    bool near (double a, double b) { return std::abs(a-b) < 1e-12; }

    // Three documents over terms 0 to 3:
    //   term:          0  1  2  3
    //   documents in:  1  3  1  1
    //   total count:   1  4  3  1
    std::vector<sparse_type> corpus()
    {
        return {
            {{0,1}, {1,2}},
            {{1,1}, {2,3}},
            {{1,1}, {3,1}}
        };
    }

// ----------------------------------------------------------------------------------------

    void test_defaults()
    {
        tfidf_transformer<sparse_type> tfidf;
        DLIB_TEST(tfidf.vocabulary_size() == 0);
        DLIB_TEST(tfidf.min_document_frequency() == 1);
        DLIB_TEST(tfidf.uses_smooth_idf());
        DLIB_TEST(!tfidf.uses_sublinear_tf());
        DLIB_TEST(tfidf.normalizes_to_unit_length());

        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary() == std::vector<unsigned long>({0,1,2,3}));
        // smooth idf: log((1+n)/(1+df)) + 1 with n == 3
        DLIB_TEST(near(tfidf.idf()[0], std::log(2.0) + 1));
        DLIB_TEST(near(tfidf.idf()[1], 1));
        DLIB_TEST(near(tfidf.idf()[2], std::log(2.0) + 1));
        DLIB_TEST(near(tfidf.idf()[3], std::log(2.0) + 1));

        const sparse_type r = tfidf(corpus()[0]);
        const double a = std::log(2.0) + 1;
        const double b = 2;
        const double norm = std::sqrt(a*a + b*b);
        DLIB_TEST(r.size() == 2);
        DLIB_TEST(r[0].first == 0 && near(r[0].second, a/norm));
        DLIB_TEST(r[1].first == 1 && near(r[1].second, b/norm));
        DLIB_TEST(near(length(r), 1));
    }

// ----------------------------------------------------------------------------------------

    void test_term_selection()
    {
        tfidf_transformer<sparse_type> tfidf;
        tfidf.set_min_document_frequency(2);
        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary() == std::vector<unsigned long>({1}));
        const sparse_type r = tfidf(corpus()[0]);
        DLIB_TEST(r.size() == 1 && r[0].first == 0 && near(r[0].second, 1));

        // The most frequent terms by total count: term 1 (4), then term 2 (3).
        tfidf.set_min_document_frequency(1);
        tfidf.set_max_vocabulary_size(2);
        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary() == std::vector<unsigned long>({1,2}));

        // Terms 0 and 3 tie with a total count of 1; the smaller term wins.
        tfidf.set_max_vocabulary_size(3);
        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary() == std::vector<unsigned long>({0,1,2}));

        tfidf.set_max_vocabulary_size(0);
        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary_size() == 0);
        DLIB_TEST(tfidf(corpus()[0]).empty());
    }

// ----------------------------------------------------------------------------------------

    void test_weighting_options()
    {
        tfidf_transformer<sparse_type> tfidf;
        tfidf.normalize_to_unit_length(false);
        tfidf.train(corpus());

        sparse_type r = tfidf(corpus()[1]);
        DLIB_TEST(r.size() == 2);
        DLIB_TEST(r[0].first == 1 && near(r[0].second, 1*1.0));
        DLIB_TEST(r[1].first == 2 && near(r[1].second, 3*(std::log(2.0)+1)));

        // Weighting options apply to operator() right away.
        tfidf.use_sublinear_tf(true);
        r = tfidf(corpus()[1]);
        DLIB_TEST(near(r[0].second, (1 + std::log(1.0))*1));
        DLIB_TEST(near(r[1].second, (1 + std::log(3.0))*(std::log(2.0)+1)));

        // idf smoothing applies on the next train().
        tfidf.use_smooth_idf(false);
        DLIB_TEST(near(tfidf.idf()[0], std::log(2.0) + 1));
        tfidf.train(corpus());
        DLIB_TEST(near(tfidf.idf()[0], std::log(3.0) + 1));
        DLIB_TEST(near(tfidf.idf()[1], 1));

        // So does term selection.
        tfidf.set_min_document_frequency(2);
        DLIB_TEST(tfidf.vocabulary_size() == 4);
        tfidf.train(corpus());
        DLIB_TEST(tfidf.vocabulary_size() == 1);
    }

// ----------------------------------------------------------------------------------------

    void test_edge_cases()
    {
        tfidf_transformer<sparse_type> tfidf;
        tfidf.train(corpus());

        // Zero counts and terms unseen in training are ignored.
        const sparse_type r = tfidf({{0,0}, {1,1}, {9,5}});
        DLIB_TEST(r.size() == 1 && r[0].first == 1 && near(r[0].second, 1));

        DLIB_TEST(tfidf(sparse_type()).empty());

        // A term whose only counts are zero is not kept.
        tfidf.train({{{0,1}, {5,0}}, {{0,2}}});
        DLIB_TEST(tfidf.vocabulary() == std::vector<unsigned long>({0}));

        tfidf.train({});
        DLIB_TEST(tfidf.vocabulary_size() == 0);
        DLIB_TEST(tfidf(corpus()[0]).empty());
    }

// ----------------------------------------------------------------------------------------

    void test_map_samples()
    {
        using map_type = std::map<unsigned long,double>;
        std::vector<map_type> documents;
        for (const auto& d : corpus())
            documents.emplace_back(d.begin(), d.end());

        tfidf_transformer<map_type> by_map;
        tfidf_transformer<sparse_type> by_vector;
        by_map.train(documents);
        by_vector.train(corpus());
        DLIB_TEST(by_map.vocabulary() == by_vector.vocabulary());
        for (unsigned long i = 0; i < documents.size(); ++i)
        {
            DLIB_TEST(by_map(documents[i]) == by_vector(corpus()[i]));
        }
    }

// ----------------------------------------------------------------------------------------

    void test_integer_counts()
    {
        using int_sparse_type = std::vector<std::pair<unsigned long,unsigned int>>;
        static_assert(std::is_same_v<tfidf_transformer<int_sparse_type>::scalar_type, double>);
        std::vector<int_sparse_type> documents;
        for (const auto& d : corpus())
            documents.emplace_back(d.begin(), d.end());

        tfidf_transformer<int_sparse_type> by_int;
        tfidf_transformer<sparse_type> by_double;
        by_int.use_sublinear_tf(true);
        by_double.use_sublinear_tf(true);
        by_int.train(documents);
        by_double.train(corpus());
        DLIB_TEST(by_int.vocabulary() == by_double.vocabulary());
        DLIB_TEST(by_int.idf() == by_double.idf());
        for (unsigned long i = 0; i < documents.size(); ++i)
        {
            DLIB_TEST(by_int(documents[i]) == by_double(corpus()[i]));
        }
    }

// ----------------------------------------------------------------------------------------

    void test_serialization()
    {
        tfidf_transformer<sparse_type> tfidf;
        tfidf.set_max_vocabulary_size(3);
        tfidf.use_sublinear_tf(true);
        tfidf.train(corpus());

        std::ostringstream out;
        serialize(tfidf, out);
        tfidf_transformer<sparse_type> copy;
        std::istringstream in(out.str());
        deserialize(copy, in);

        DLIB_TEST(copy.max_vocabulary_size() == 3);
        DLIB_TEST(copy.uses_sublinear_tf());
        DLIB_TEST(copy.vocabulary() == tfidf.vocabulary());
        DLIB_TEST(copy.idf() == tfidf.idf());
        for (const auto& d : corpus())
            DLIB_TEST(copy(d) == tfidf(d));
    }

// ----------------------------------------------------------------------------------------

    class tfidf_tester : public tester
    {
    public:
        tfidf_tester (
        ) :
            tester (
                "test_tfidf",
                "Run tests on the tfidf_transformer.",
                0
            )
        {}

        void perform_test (
        )
        {
            test_defaults();
            test_term_selection();
            test_weighting_options();
            test_edge_cases();
            test_map_samples();
            test_integer_counts();
            test_serialization();
        }
    } a;

}
