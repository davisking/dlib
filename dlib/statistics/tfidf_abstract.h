// Copyright (C) 2026  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_TFIDF_ABSTRACT_Hh_
#ifdef DLIB_TFIDF_ABSTRACT_Hh_

#include "../svm/sparse_vector_abstract.h"
#include "../serialize.h"
#include <limits>
#include <type_traits>
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
            REQUIREMENTS ON sample_type
                - sample_type is a sparse vector type, as defined in
                  dlib/svm/sparse_vector_abstract.h, e.g.
                  std::vector<std::pair<unsigned long,double>> or
                  std::map<unsigned long,double>.
                - Its values, the counts, may be integers or floating point numbers.
                  The weights, idf() and the values operator() returns, are of
                  scalar_type: the count type if it is floating point, double otherwise.

            INITIAL VALUE
                - vocabulary_size() == 0
                - min_document_frequency() == 1
                - max_vocabulary_size() == std::numeric_limits<unsigned long>::max()
                - uses_sublinear_tf() == false
                - uses_smooth_idf() == true
                - normalizes_to_unit_length() == true

            WHAT THIS OBJECT REPRESENTS
                This object learns TF-IDF weights from a collection of documents and
                applies them to term counts.  A document is a sparse vector of term
                counts: the entry (t,c) says that term t occurs c times in it.  TF-IDF
                weighs each count by how rare its term is across the training documents,
                so terms that appear in most documents count for less than terms that
                appear in few.

                train() chooses the vocabulary, the terms it keeps, and computes each
                one's inverse document frequency (idf).  operator() then maps a
                document's counts to a sparse vector over the vocabulary: entry j is the
                weighted count of term vocabulary()[j], times idf()[j], so its indices
                are in [0,vocabulary_size()).
        !*/

    public:
        using index_type = std::remove_const_t<typename sample_type::value_type::first_type>;
        using count_type = typename sample_type::value_type::second_type;
        using scalar_type = std::conditional_t<std::is_floating_point_v<count_type>, count_type, double>;
        using result_type = std::vector<std::pair<unsigned long, scalar_type>>;

        void set_min_document_frequency (
            unsigned long min_df
        );
        /*!
            requires
                - min_df >= 1
            ensures
                - #min_document_frequency() == min_df
                - Takes effect on the next call to train().
        !*/

        unsigned long min_document_frequency (
        ) const;
        /*!
            ensures
                - Returns the number of training documents a term must occur in for
                  train() to keep it.
        !*/

        void set_max_vocabulary_size (
            unsigned long num
        );
        /*!
            ensures
                - #max_vocabulary_size() == num
                - Takes effect on the next call to train().
        !*/

        unsigned long max_vocabulary_size (
        ) const;
        /*!
            ensures
                - Returns the largest number of terms train() keeps.
        !*/

        void use_sublinear_tf (
            bool enabled
        );
        /*!
            ensures
                - #uses_sublinear_tf() == enabled
                - Takes effect on the next call to operator().
        !*/

        bool uses_sublinear_tf (
        ) const;
        /*!
            ensures
                - Returns true if operator() weighs a count c as 1 + log(c), and false
                  if it uses c itself.  The logarithm keeps a term that occurs many times
                  in a document from dominating it.
        !*/

        void use_smooth_idf (
            bool enabled
        );
        /*!
            ensures
                - #uses_smooth_idf() == enabled
                - Takes effect on the next call to train().
        !*/

        bool uses_smooth_idf (
        ) const;
        /*!
            ensures
                - Returns true if train() computes idf()[j] as log((1+n)/(1+df)) + 1,
                  and false if it uses log(n/df) + 1, where n is the number of training
                  documents and df the number of them the term occurs in.  Smoothing
                  counts one extra document containing every term.  A term found in
                  every training document has IDF weight 1 with either formula.
        !*/

        void normalize_to_unit_length (
            bool enabled
        );
        /*!
            ensures
                - #normalizes_to_unit_length() == enabled
                - Takes effect on the next call to operator().
        !*/

        bool normalizes_to_unit_length (
        ) const;
        /*!
            ensures
                - Returns true if operator() scales each nonzero result to length() 1,
                  so that long and short documents are comparable.
        !*/

        void train (
            const std::vector<sample_type>& documents
        );
        /*!
            requires
                - Each documents[i] is a sparse vector whose values are all finite and
                  >= 0.
            ensures
                - A term's document frequency, df, is the number of documents with a
                  count above 0 for it.
                - #vocabulary() lists the kept terms in increasing order: every term
                  with df >= min_document_frequency() or, if there are more than
                  max_vocabulary_size() of them, the max_vocabulary_size() of them with
                  the largest sum of counts over documents, ties going to the smaller term.
                - #vocabulary_size() == #vocabulary().size()
                - #idf().size() == #vocabulary_size(), and #idf()[j] is the inverse
                  document frequency of term #vocabulary()[j], as uses_smooth_idf()
                  defines it, with n == documents.size().
        !*/

        unsigned long vocabulary_size (
        ) const;
        /*!
            ensures
                - Returns the number of terms in the vocabulary, i.e. the dimension of
                  the vectors operator() returns.
        !*/

        const std::vector<index_type>& vocabulary (
        ) const;
        /*!
            ensures
                - Returns the vocabulary: the terms the last train() kept.
        !*/

        const std::vector<scalar_type>& idf (
        ) const;
        /*!
            ensures
                - Returns the inverse document frequency of each term in vocabulary().
        !*/

        result_type operator() (
            const sample_type& counts
        ) const;
        /*!
            requires
                - counts is a sparse vector whose values are all finite and >= 0.
            ensures
                - Returns the TF-IDF vector of the document counts.  Let V be the sparse
                  vector with one entry (j, w(c)*idf()[j]) for each entry (t,c) of counts
                  with c > 0 and t == vocabulary()[j], where w(c) is 1 + log(c) if
                  uses_sublinear_tf() and c otherwise.  If normalizes_to_unit_length()
                  and V is nonzero, returns V/length(V).  Otherwise returns V.
        !*/
    };

    template <typename sample_type>
    void serialize (
        const tfidf_transformer<sample_type>& item,
        std::ostream& out
    );
    /*!
        provides serialization support
    !*/

    template <typename sample_type>
    void deserialize (
        tfidf_transformer<sample_type>& item,
        std::istream& in
    );
    /*!
        provides deserialization support
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_TFIDF_ABSTRACT_Hh_
