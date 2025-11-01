// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_TRANSFORMER_H_
#define DLIB_DNN_TRANSFORMER_H_

#include "transformer_abstract.h"
#include "layers.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

    template <long d_k_>
    class scale_weights_ : public multiply_
    {
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    template <template <typename> class ACT, long reduction_factor, long d_model, typename SUBNET>
    using projection_head = ACT<fc<d_model / reduction_factor, rms_norm<SUBNET>>>;

    // ----------------------------------------------------------------------------------------

    template <long num_embeddings, long embedding_length, typename SUBNET>
    using token_embeddings = positional_encodings<
        embeddings<num_embeddings, embedding_length, SUBNET>>;

    // ----------------------------------------------------------------------------------------

    // CANONICAL TRANSFORMER ARCHITECTURE (separate projections)
    namespace canonical_transformer
    {

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using query = reshape_to<num_heads, seq_len, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using key = reshape_to<num_heads, seq_len, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using value = reshape_to<num_heads, seq_len, d_model / num_heads,
            linear_no_bias<d_model, SUBNET>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using multihead_attention = add_prev1<
            DO<linear_no_bias<d_model, reshape_to<1, seq_len, d_model,
            multm_prev2<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev3<
            rope<query<seq_len, d_model, num_heads, skip1<
            tag3<transpose<
            rope<key<seq_len, d_model, num_heads, skip1<
            tag2<value<seq_len, d_model, num_heads,
            rms_norm<tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using feed_forward =
            add_prev4<
            DO<linear<d_model, DO<ACT<linear<d_model * 4, rms_norm<tag4<SUBNET>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            act<feed_forward<ACT, DO, d_model,
            multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>>;

        template<long remaining_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<ACT, DO, seq_len, d_model, num_heads,
                typename transformer_stack_impl<remaining_layers - 1, ACT, DO, seq_len, d_model, num_heads, SUBNET>::type>;
        };

        template<template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        struct transformer_stack_impl<0, ACT, DO, seq_len, d_model, num_heads, SUBNET, void>
        {
            using type = tag5<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, seq_len, d_model, num_heads, SUBNET>::type;

    } // namespace std_transformer

    // FUSED TRANSFORMER ARCHITECTURE (separate projections)
    namespace fused_transformer
    {

        template <long num_heads, long d_model, typename SUBNET>
        using query = extract<0, num_heads, d_model / num_heads, 1, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using key = extract<d_model, num_heads, 1, d_model / num_heads, SUBNET>;

        template <long num_heads, long d_model, typename SUBNET>
        using value = extract<(d_model * 2), num_heads, d_model / num_heads, 1, SUBNET>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention = add_prev1<
            DO<extract<0, 1, 1, d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            query<num_heads, d_model, skip2<
            tag4<key<num_heads, d_model, skip2<
            tag3<value<num_heads, d_model,
            tag2<fc_no_bias<d_model * 3,
            rms_norm<tag1<SUBNET>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using feed_forward =
            add_prev5<extract<0, 1, 1, d_model,
            DO<fc<d_model, DO<ACT<fc<d_model * 4, rms_norm<tag5<SUBNET>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            feed_forward<ACT, DO, d_model,
            multihead_attention<ACT, DO, d_model, num_heads, SUBNET>>;

        template<long remaining_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<ACT, DO, seq_len, d_model, num_heads,
                typename transformer_stack_impl<remaining_layers - 1, ACT, DO, seq_len, d_model, num_heads, SUBNET>::type>;
        };

        template<template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        struct transformer_stack_impl<0, ACT, DO, seq_len, d_model, num_heads, SUBNET, void>
        {
            using type = tag6<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, seq_len, d_model, num_heads, SUBNET>::type;

    } // namespace fused_transformer

    // Default to fused transformer implementation
    using namespace fused_transformer;
}

#endif // DLIB_DNN_TRANSFORMER_H_