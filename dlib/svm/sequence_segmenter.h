// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SEQUENCE_SeGMENTER_H___
#define DLIB_SEQUENCE_SeGMENTER_H___

#include "sequence_segmenter_abstract.h"
#include "../matrix.h"
#include "sequence_labeler.h"
#include <vector>

namespace dlib
{
    // This namespace contains implementation details for the sequence_segmenter.
    namespace impl_ss
    {

    // ------------------------------------------------------------------------------------

        const unsigned int BEGIN   = 0;
        const unsigned int INSIDE  = 1;
        const unsigned int OUTSIDE = 2;

    // ------------------------------------------------------------------------------------

        template <typename ss_feature_extractor>
        class feature_extractor
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    This is a feature extractor for a sequence_labeler.  It serves to map
                    the interface defined by a sequence_labeler into the kind of interface
                    defined for a sequence_segmenter.
            !*/

        public:
            typedef typename ss_feature_extractor::sequence_type sequence_type;

            ss_feature_extractor fe;

            feature_extractor() {}
            feature_extractor(const ss_feature_extractor& ss_fe_) : fe(ss_fe_) {}

            friend void serialize(const feature_extractor& item, std::ostream& out) 
            {
                serialize(item.fe, out);
            }

            friend void deserialize(feature_extractor& item, std::istream& in) 
            {
                deserialize(item.fe, in);
            }

            unsigned long num_features() const
            {

                const int base_dims = fe.num_features();
                return num_labels()*(
                    num_labels() +  // previous and current label
                    base_dims*fe.window_size() + // window around current element
                    num_labels()*base_dims*fe.window_size() // window around current element in conjunction with previous label
                );
            }

            unsigned long order() const 
            { 
                return 1; 
            }

            unsigned long num_labels() const 
            { 
                return 3; 
            }

        private:

            template <typename feature_setter>
            struct dot_functor
            {
                /*!
                    WHAT THIS OBJECT REPRESENTS
                        This class wraps the feature_setter used by a sequence_labeler
                        and turns it into the kind needed by a sequence_segmenter.
                !*/

                dot_functor(feature_setter& set_feature_, unsigned long offset_) : 
                    set_feature(set_feature_), offset(offset_) {}

                feature_setter& set_feature;
                unsigned long offset;

                inline void operator() (
                    unsigned long feat_index
                )
                {
                    set_feature(offset+feat_index);
                }

                inline void operator() (
                    unsigned long feat_index,
                    double feat_value
                )
                {
                    set_feature(offset+feat_index, feat_value);
                }
            };

        public:

            template <typename EXP>
            bool reject_labeling (
                const sequence_type& ,
                const matrix_exp<EXP>& y,
                unsigned long 
            ) const
            {
                // Don't allow BIO label patterns that don't correspond to a sensical
                // segmentation. 
                if (y.size() > 1 && y(0) == INSIDE && y(1) == OUTSIDE)
                    return true;
                return false;
            }

            template <typename feature_setter, typename EXP>
            void get_features (
                feature_setter& set_feature,
                const sequence_type& x,
                const matrix_exp<EXP>& y,
                unsigned long position
            ) const
            {
                // Pull out an indicator feature for the type of transition between the
                // previous label and the current label.
                if (y.size() > 1)
                    set_feature(y(1)*num_labels() + y(0));

                unsigned long offset = num_labels()*num_labels();

                const int window_size = fe.window_size();

                const int base_dims = fe.num_features();
                for (int i = 0; i < window_size; ++i)
                {
                    const long pos = i-window_size/2 + static_cast<long>(position);
                    if (0 <= pos && pos < (long)x.size())
                    {
                        const unsigned long off1 = y(0)*base_dims;
                        dot_functor<feature_setter> fs1(set_feature, offset+off1);
                        fe.get_features(fs1, x, pos);
                        if (y.size() > 1)
                        {
                            const unsigned long off2 = num_labels()*base_dims + (y(0)*num_labels()+y(1))*base_dims;
                            dot_functor<feature_setter> fs2(set_feature, offset+off2);
                            fe.get_features(fs2, x, pos);
                        }
                    }

                    offset += num_labels()*(base_dims + num_labels()*base_dims);
                }

            }
        };

    } // end namespace impl_ss

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    unsigned long total_feature_vector_size (
        const feature_extractor& fe
    )
    {
        return 3*3 + 12*fe.num_features()*fe.window_size();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename feature_extractor
        >
    class sequence_segmenter
    {
    public:
        typedef typename feature_extractor::sequence_type sample_sequence_type;
        typedef std::vector<std::pair<unsigned long, unsigned long> > segmented_sequence_type;


        sequence_segmenter()
        {
            const feature_extractor& fe = labeler.get_feature_extractor().fe;
            DLIB_ASSERT(fe.window_size() >= 1 && fe.num_features() >= 1,
                "\t sequence_segmenter::sequence_segmenter()"
                << "\n\t An invalid feature extractor was supplied."
                << "\n\t fe.window_size():  " << fe.window_size() 
                << "\n\t fe.num_features(): " << fe.num_features() 
                << "\n\t this: " << this
            );
        }

        explicit sequence_segmenter(
            const matrix<double,0,1>& weights
        ) : 
            labeler(weights)
        {
            const feature_extractor& fe = labeler.get_feature_extractor().fe;
            // make sure requires clause is not broken
            DLIB_ASSERT(total_feature_vector_size(fe) == (unsigned long)weights.size(),
                "\t sequence_segmenter::sequence_segmenter(weights)"
                << "\n\t These sizes should match"
                << "\n\t total_feature_vector_size(fe):  " << total_feature_vector_size(fe) 
                << "\n\t weights.size(): " << weights.size() 
                << "\n\t this: " << this
                );
            DLIB_ASSERT(fe.window_size() >= 1 && fe.num_features() >= 1,
                "\t sequence_segmenter::sequence_segmenter()"
                << "\n\t An invalid feature extractor was supplied."
                << "\n\t fe.window_size():  " << fe.window_size() 
                << "\n\t fe.num_features(): " << fe.num_features() 
                << "\n\t this: " << this
            );
        }

        sequence_segmenter(
            const matrix<double,0,1>& weights,
            const feature_extractor& fe
        ) :
            labeler(weights, impl_ss::feature_extractor<feature_extractor>(fe))
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(total_feature_vector_size(fe) == (unsigned long)weights.size(),
                "\t sequence_segmenter::sequence_segmenter(weights,fe)"
                << "\n\t These sizes should match"
                << "\n\t total_feature_vector_size(fe):  " << total_feature_vector_size(fe) 
                << "\n\t weights.size(): " << weights.size() 
                << "\n\t this: " << this
                );
            DLIB_ASSERT(fe.window_size() >= 1 && fe.num_features() >= 1,
                "\t sequence_segmenter::sequence_segmenter()"
                << "\n\t An invalid feature extractor was supplied."
                << "\n\t fe.window_size():  " << fe.window_size() 
                << "\n\t fe.num_features(): " << fe.num_features() 
                << "\n\t this: " << this
            );
        }

        const feature_extractor& get_feature_extractor (
        ) const { return labeler.get_feature_extractor().fe; }

        const matrix<double,0,1>& get_weights (
        ) const { return labeler.get_weights(); }

        segmented_sequence_type operator() (
            const sample_sequence_type& x
        ) const
        {
            segmented_sequence_type y;
            segment_sequence(x,y);
            return y;
        }

        void segment_sequence (
            const sample_sequence_type& x,
            segmented_sequence_type& y
        ) const
        {
            y.clear();
            std::vector<unsigned long> labels;
            labeler.label_sequence(x, labels);

            // Convert from BIO tagging to the explicit segments representation.
            for (unsigned long i = 0; i < labels.size(); ++i)
            {
                if (labels[i] == impl_ss::BEGIN)
                {
                    const unsigned long begin = i;
                    ++i;
                    while (i < labels.size() && labels[i] == impl_ss::INSIDE)
                        ++i;

                    y.push_back(std::make_pair(begin, i));
                    --i;
                }
            }
        }

        friend void serialize(const sequence_segmenter& item, std::ostream& out)
        {
            int version = 1;
            serialize(version, out);
            serialize(item.labeler, out);
        }

        friend void deserialize(sequence_segmenter& item, std::istream& in)
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::sequence_segmenter.");
            deserialize(item.labeler, in);
        }

    private:
        sequence_labeler<impl_ss::feature_extractor<feature_extractor> > labeler;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SEQUENCE_SeGMENTER_H___


