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

    template <
        typename feature_extractor
        >
    class sequence_labeler
    {
        /*!
            WHAT THIS OBJECT REPRESENTS

        !*/

    public:
        typedef typename feature_extractor::sample_type sample_type;
        typedef std::vector<sample_type> sample_sequence_type;
        typedef std::vector<unsigned long> labeled_sequence_type;

        sequence_labeler() {}

        sequence_labeler(
            const feature_extractor& fe_,
            const matrix<double,0,1>& weights_
        ); 

        const feature_extractor& get_feature_extractor (
        ) const; 

        const matrix<double,0,1>& get_weights (
        ) const;

        unsigned long num_labels (
        ) const { return fe.num_labels(); }

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


