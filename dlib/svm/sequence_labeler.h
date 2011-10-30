// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SEQUENCE_LAbELER_H___
#define DLIB_SEQUENCE_LAbELER_H___

#include "sequence_labeler_abstract.h"
#include "../matrix.h"
#include <vector>
#include "../optimization/find_max_factor_graph_viterbi.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace fe_helpers
    {
        template <typename EXP>
        struct dot_functor
        {
            dot_functor(const matrix_exp<EXP>& lambda_) : lambda(lambda_), value(0) {}

            inline void operator() (
                unsigned long feat_index
            )
            {
                value += lambda(feat_index);
            }

            inline void operator() (
                unsigned long feat_index,
                double feat_value
            )
            {
                value += feat_value*lambda(feat_index);
            }

            const matrix_exp<EXP>& lambda;
            double value;
        };

        template <typename feature_extractor, typename EXP, typename sample_type, typename EXP2> 
        double dot(
            const matrix_exp<EXP>& lambda,
            const feature_extractor& fe,
            unsigned long position,
            const matrix_exp<EXP2>& label_states,
            const std::vector<sample_type>& x
        )
        {
            dot_functor<EXP> dot(lambda);
            fe.get_features(dot, position, label_states, x);
            return dot.value;
        }

    }

// ----------------------------------------------------------------------------------------

    template <typename feature_extractor>
    class sequence_labeler
    {
    public:
        typedef typename feature_extractor::sample_type sample_type;
        typedef std::vector<sample_type> sample_sequence_type;
        typedef std::vector<unsigned long> labeled_sequence_type;

    private:
        class map_prob
        {
        public:
            unsigned long order() const { return fe.order(); }
            unsigned long num_states() const { return fe.num_labels(); }

            map_prob(
                const sample_sequence_type& x_,
                const feature_extractor& fe_,
                const matrix<double,0,1>& weights_
            ) :
                x(x_),
                fe(fe_),
                weights(weights_)
            {
            }

            unsigned long number_of_nodes(
            ) const
            {
                return x.size();
            }

            template <
                typename EXP 
                >
            double factor_value (
                unsigned long node_id,
                const matrix_exp<EXP>& node_states
            ) const
            {
                if (fe.reject_labeling(node_id, node_states, x))
                    return -std::numeric_limits<double>::infinity();

                return fe_helpers::dot(weights, fe, node_id, node_states, x);
            }

            const sample_sequence_type& x;
            const feature_extractor& fe;
            const matrix<double,0,1>& weights;
        };
    public:

        sequence_labeler()
        {}

        sequence_labeler(
            const feature_extractor& fe_,
            const matrix<double,0,1>& weights_
        ) :
            fe(fe_),
            weights(weights_)
        {}

        const feature_extractor& get_feature_extractor (
        ) const { return fe; }

        const matrix<double,0,1>& get_weights (
        ) const { return weights; }

        unsigned long num_labels (
        ) const { return fe.num_labels(); }

        labeled_sequence_type operator() (
            const sample_sequence_type& x
        ) const
        {
            labeled_sequence_type y;
            find_max_factor_graph_viterbi(map_prob(x,fe,weights), y);
            return y;
        }

        void label_sequence (
            const sample_sequence_type& x,
            labeled_sequence_type& y
        ) const
        {
            find_max_factor_graph_viterbi(map_prob(x,fe,weights), y);
        }

    private:

        feature_extractor fe;
        matrix<double,0,1> weights;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SEQUENCE_LAbELER_H___

