// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TFIDF_Hh_
#define DLIB_TFIDF_Hh_

#include "tfidf_abstract.h"
#include "../algs.h"
#include "../serialize.h"
#include "../sparse_vector.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename sample_type
        >
    class tfidf_transformer
    {
        /*!
            CONVENTION
                - vocabulary_ is sorted, with no repeated terms.
                - idf_.size() == vocabulary_.size()
        !*/

    public:
        using index_type = std::remove_const_t<typename sample_type::value_type::first_type>;
        using count_type = typename sample_type::value_type::second_type;
        using scalar_type = std::conditional_t<std::is_floating_point_v<count_type>, count_type, double>;
        using result_type = std::vector<std::pair<unsigned long, scalar_type>>;

        void set_min_document_frequency (
            unsigned long min_df
        )
        {
            DLIB_ASSERT(min_df >= 1,
                "\t void tfidf_transformer::set_min_document_frequency()"
                << "\n\t min_df must be at least 1"
                << "\n\t min_df: " << min_df
            );
            min_df_ = min_df;
        }

        unsigned long min_document_frequency (
        ) const
        {
            return min_df_;
        }

        void set_max_vocabulary_size (
            unsigned long num
        )
        {
            max_vocabulary_size_ = num;
        }

        unsigned long max_vocabulary_size (
        ) const
        {
            return max_vocabulary_size_;
        }

        void use_sublinear_tf (
            bool enabled
        )
        {
            sublinear_tf_ = enabled;
        }

        bool uses_sublinear_tf (
        ) const
        {
            return sublinear_tf_;
        }

        void use_smooth_idf (
            bool enabled
        )
        {
            smooth_idf_ = enabled;
        }

        bool uses_smooth_idf (
        ) const
        {
            return smooth_idf_;
        }

        void normalize_to_unit_length (
            bool enabled
        )
        {
            normalize_ = enabled;
        }

        bool normalizes_to_unit_length (
        ) const
        {
            return normalize_;
        }

        void train (
            const std::vector<sample_type>& documents
        )
        {
            DLIB_ASSERT(
                std::all_of(documents.begin(), documents.end(), is_nonnegative_sparse_vector),
                "\t void tfidf_transformer::train()"
                << "\n\t each document must be a sparse vector of finite counts >= 0"
            );
            // term -> (documents it occurs in, sum of its counts)
            std::unordered_map<index_type, std::pair<unsigned long, double>> stats;
            for (const auto& document : documents)
            {
                for (const auto& [term, count] : document)
                {
                    if (count > 0)
                    {
                        auto& [df, total] = stats[term];
                        ++df;
                        total += count;
                    }
                }
            }

            std::vector<std::pair<index_type, std::pair<unsigned long, double>>> kept;
            kept.reserve(stats.size());
            for (const auto& entry : stats)
            {
                if (entry.second.first >= min_df_)
                    kept.push_back(entry);
            }
            // The max_vocabulary_size_ terms with the largest totals, ties to the smaller term.
            const auto keep = std::min<unsigned long>(kept.size(), max_vocabulary_size_);
            std::nth_element(kept.begin(), kept.begin() + keep, kept.end(),
                [](const auto& a, const auto& b)
                {
                    return a.second.second != b.second.second ? a.second.second > b.second.second
                                                              : a.first < b.first;
                });
            kept.resize(keep);
            std::sort(kept.begin(), kept.end());

            const double n = documents.size();
            vocabulary_.clear();
            idf_.clear();
            for (const auto& [term, s] : kept)
            {
                const double df = s.first;
                vocabulary_.push_back(term);
                idf_.push_back(std::log(smooth_idf_ ? (1 + n) / (1 + df) : n / df) + 1);
            }
        }

        unsigned long vocabulary_size (
        ) const
        {
            return vocabulary_.size();
        }

        const std::vector<index_type>& vocabulary (
        ) const
        {
            return vocabulary_;
        }

        const std::vector<scalar_type>& idf (
        ) const
        {
            return idf_;
        }

        result_type operator() (
            const sample_type& counts
        ) const
        {
            DLIB_ASSERT(is_nonnegative_sparse_vector(counts),
                "\t result_type tfidf_transformer::operator()"
                << "\n\t counts must be a sparse vector of finite counts >= 0"
            );
            // counts and vocabulary_ are both sorted by term, so each search starts
            // where the last one ended and out comes out sorted.
            result_type out;
            auto at = vocabulary_.begin();
            for (const auto& [term, count] : counts)
            {
                if (count <= 0)
                    continue;
                at = std::lower_bound(at, vocabulary_.end(), term);
                if (at == vocabulary_.end() || *at != term)
                    continue;
                const unsigned long j = at - vocabulary_.begin();
                const scalar_type tf = sublinear_tf_ ? 1 + std::log(count) : count;
                out.emplace_back(j, tf * idf_[j]);
            }
            const scalar_type norm = normalize_ ? length(out) : 0;
            if (norm > 0)
            {
                for (auto& entry : out)
                    entry.second /= norm;
            }
            return out;
        }

        friend void serialize (
            const tfidf_transformer& item,
            std::ostream& out
        )
        {
            serialize(1, out);
            serialize_these(out, item.min_df_, item.max_vocabulary_size_, item.sublinear_tf_,
                item.smooth_idf_, item.normalize_, item.vocabulary_, item.idf_);
        }

        friend void deserialize (
            tfidf_transformer& item,
            std::istream& in
        )
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error(
                    "Unexpected version found while deserializing dlib::tfidf_transformer.");
            deserialize_these(in, item.min_df_, item.max_vocabulary_size_, item.sublinear_tf_,
                item.smooth_idf_, item.normalize_, item.vocabulary_, item.idf_);
        }

    private:

        static bool is_nonnegative_sparse_vector (
            const sample_type& sample
        )
        /*!
            ensures
                - Returns true if and only if sample's keys are strictly increasing and
                  its values are all finite and >= 0.
        !*/
        {
            const auto unordered = [](const auto& a, const auto& b) { return a.first >= b.first; };
            const auto valid = [](const auto& entry)
            { return std::isfinite(entry.second) && entry.second >= 0; };
            return std::adjacent_find(sample.begin(), sample.end(), unordered) == sample.end() &&
                   std::all_of(sample.begin(), sample.end(), valid);
        }

        unsigned long min_df_ = 1;
        unsigned long max_vocabulary_size_ = std::numeric_limits<unsigned long>::max();
        bool sublinear_tf_ = false;
        bool smooth_idf_ = true;
        bool normalize_ = true;
        std::vector<index_type> vocabulary_;
        std::vector<scalar_type> idf_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TFIDF_Hh_
