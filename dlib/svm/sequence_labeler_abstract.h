// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SEQUENCE_LAbELER_ABSTRACT_H___
#ifdef DLIB_SEQUENCE_LAbELER_ABSTRACT_H___

#include "../matrix.h"
#include <vector>
#include "../optimization/find_max_factor_graph_viterbi_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class example_feature_extractor
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

    public:
        typedef word_type sample_type;

        example_feature_extractor (
        ); 

        unsigned long num_features (
        ) const;

        unsigned long order(
        ) const; 

        unsigned long num_labels(
        ) const; 

        template <typename EXP>
        bool reject_labeling (
            const std::vector<sample_type>& sequence,
            const matrix_exp<EXP>& candidate_labeling,
            unsigned long position
        ) const;
        /*!
            requires
                - EXP::type == unsigned long
                  (i.e. candidate_labeling contains unsigned longs)
                - position < sequence.size()
                - candidate_labeling.size() == min(position, order) + 1
                - is_vector(candidate_labeling) == true
                - max(candidate_labeling) < num_labels() 
            ensures
                - if (the given candidate_labeling for sequence[position] is
                  always the wrong labeling) then
                    - returns true
                      (note that reject_labeling() is just an optional tool to allow 
                      you to overrule the learning algorithm.  You don't have to use
                      it.  So if you prefer you can set reject_labeling() to always
                      return false.)
                - else
                    - returns false
        !*/

        template <typename feature_setter, typename EXP>
        void get_features (
            feature_setter& set_feature,
            const std::vector<sample_type>& sequence,
            const matrix_exp<EXP>& candidate_labeling,
            unsigned long position
        ) const;
        /*!
            requires
                - EXP::type == unsigned long
                  (i.e. candidate_labeling contains unsigned longs)
                - position < sequence.size()
                - candidate_labeling.size() == min(position, order) + 1
                - is_vector(candidate_labeling) == true
                - max(candidate_labeling) < num_labels() 
                - set_feature is a function object which allows expressions of the form:
                    - set_features((unsigned long)feature_index, (double)feature_value);
                    - set_features((unsigned long)feature_index);
            ensures
        !*/

    };

// ----------------------------------------------------------------------------------------

    void serialize(
        const example_feature_extractor& item,
        std::ostream& out
    );
    /*!
        provides serialization support 
    !*/

    void deserialize(
        example_feature_extractor& item, 
        std::istream& in
    );
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class sequence_labeler
    {
        /*!
            REQUIREMENTS ON feature_extractor
                It must be an object that implements an interface compatible with 
                the example_feature_extractor discussed above.

            WHAT THIS OBJECT REPRESENTS

        !*/

    public:
        typedef typename feature_extractor::sample_type sample_type;
        typedef std::vector<sample_type> sample_sequence_type;
        typedef std::vector<unsigned long> labeled_sequence_type;

        sequence_labeler() {}

        sequence_labeler(
            const feature_extractor& fe,
            const matrix<double,0,1>& weights
        ); 
        /*!
            requires
                - fe.num_features() == weights.size()
            ensures
                - #get_feature_extractor() == fe
                - #get_weights() == weights
        !*/

        const feature_extractor& get_feature_extractor (
        ) const; 

        const matrix<double,0,1>& get_weights (
        ) const;
        /*!
            ensures
                - returns a vector of length get_feature_extractor().num_features()
        !*/

        unsigned long num_labels (
        ) const { return get_feature_extractor().num_labels(); }

        labeled_sequence_type operator() (
            const sample_sequence_type& x
        ) const;

        void label_sequence (
            const sample_sequence_type& x,
            labeled_sequence_type& y
        ) const;

    };

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void serialize (
        const sequence_labeler<feature_extractor>& item,
        std::ostream& out
    );
    /*!
        provides serialization support 
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    void deserialize (
        sequence_labeler<feature_extractor>& item,
        std::istream& in 
    );
    /*!
        provides deserialization support 
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SEQUENCE_LAbELER_ABSTRACT_H___


