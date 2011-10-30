// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_H__
#define DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_H__


#include "structural_svm_sequence_labeling_problem_abstract.h"
#include "../matrix.h"
#include "sequence_labeler.h"
#include <vector>
#include "structural_svm_problem_threaded.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{

    namespace fe_helpers
    {

    // ----------------------------------------------------------------------------------------

        struct get_feats_functor 
        {
            get_feats_functor(std::vector<std::pair<unsigned long, double> >& feats_) : feats(feats_) {}

            inline void operator() (
                unsigned long feat_index,
                double feat_value
            )
            {
                feats.push_back(std::make_pair(feat_index, feat_value));
            }

            inline void operator() (
                unsigned long feat_index
            )
            {
                feats.push_back(std::make_pair(feat_index, 1));
            }

            std::vector<std::pair<unsigned long, double> >& feats;
        };

    // ----------------------------------------------------------------------------------------

        template <typename feature_extractor, typename sample_type, typename EXP2> 
        void get_feature_vector(
            std::vector<std::pair<unsigned long, double> >& feats,
            const feature_extractor& fe,
            unsigned long position,
            const matrix_exp<EXP2>& label_states,
            const std::vector<sample_type>& x
        )
        {
            get_feats_functor funct(feats);
            fe.get_features(funct, position, label_states, x);
        }

    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class structural_svm_sequence_labeling_problem : 
        public structural_svm_problem_threaded<matrix<double,0,1>, std::vector<std::pair<unsigned long,double> > >
    {
    public:
        typedef matrix<double,0,1> matrix_type;
        typedef std::vector<std::pair<unsigned long, double> > feature_vector_type;

        typedef typename feature_extractor::sample_type sample_type;

        structural_svm_sequence_labeling_problem(
            const std::vector<std::vector<sample_type> >& samples_,
            const std::vector<std::vector<unsigned long> >& labels_,
            const feature_extractor& fe_        
        ) :
            structural_svm_problem_threaded<matrix_type,feature_vector_type>(4),
            samples(samples_),
            labels(labels_),
            fe(fe_)
        {
        }

    private:
        virtual long get_num_dimensions (
        ) const 
        {
            return fe.num_features();
        }

        virtual long get_num_samples (
        ) const 
        {
            return samples.size();
        }

        void get_joint_feature_vector (
            const std::vector<sample_type>& sample, 
            const std::vector<unsigned long>& label,
            feature_vector_type& psi
        ) const 
        {
            psi.clear();

            const int order = fe.order();

            matrix<unsigned long,0,1> label_states; 
            for (unsigned long i = 0; i < sample.size(); ++i)
            {
                label_states = rowm(vector_to_matrix(label), range(i, std::max((int)i-order,0)));

                fe_helpers::get_feature_vector(psi,fe,i,label_states, sample);
            }
        }

        virtual void get_truth_joint_feature_vector (
            long idx,
            feature_vector_type& psi 
        ) const 
        {
            get_joint_feature_vector(samples[idx], labels[idx], psi);
        }

        class map_prob
        {
        public:
            unsigned long order() const { return fe.order(); }
            unsigned long num_states() const { return fe.num_labels(); }

            map_prob(
                const std::vector<sample_type>& sample_,
                const std::vector<unsigned long>& label_,
                const feature_extractor& fe_,
                const matrix<double,0,1>& weights_
            ) :
                sample(sample_),
                label(label_),
                fe(fe_),
                weights(weights_)
            {
            }

            unsigned long number_of_nodes(
            ) const
            {
                return sample.size();
            }

            template <
                typename EXP 
                >
            double factor_value (
                unsigned long node_id,
                const matrix_exp<EXP>& node_states
            ) const
            {
                // Note that it is intentional that we don't call fe.reject_labeling() here
                // because doing so would break the structural svm optimizer. 

                double loss = 0;
                if (node_states(0) != label[node_id])
                    loss = 1;

                return fe_helpers::dot(weights, fe, node_id, node_states, sample) + loss;
            }

            const std::vector<sample_type>& sample;
            const std::vector<unsigned long>& label;
            const feature_extractor& fe;
            const matrix<double,0,1>& weights;
        };

        virtual void separation_oracle (
            const long idx,
            const matrix_type& current_solution,
            scalar_type& loss,
            feature_vector_type& psi
        ) const
        {
            std::vector<unsigned long> y;
            find_max_factor_graph_viterbi(map_prob(samples[idx],labels[idx],fe,current_solution), y);

            loss = 0;
            for (unsigned long i = 0; i < y.size(); ++i)
            {
                if (y[i] != labels[idx][i])
                    loss += 1;
            }

            get_joint_feature_vector(samples[idx], y, psi);
        }

        const std::vector<std::vector<sample_type> >& samples;
        const std::vector<std::vector<unsigned long> >& labels;
        const feature_extractor& fe;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_STRUCTURAL_SVM_SEQUENCE_LaBELING_PROBLEM_H__

