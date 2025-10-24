// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_TRANSFORMER_H_
#define DLIB_DNN_TRANSFORMER_H_

#include "layers.h"

/*!
    The transformer.h file contains specialized layers and building blocks designed
    specifically for transformer architectures and attention mechanisms.

    These components are separated into their own header for better organization and
    maintainability, as transformer architectures have distinct requirements from
    traditional convolutional or fully-connected networks.

    All transformer components are seamlessly integrated with the standard DNN API
    and can be combined with any other layers defined in this file.
!*/

namespace dlib
{

	// ----------------------------------------------------------------------------------------

    template <long d_k_>
    class scale_weights_ : public multiply_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This layer scales inputs by 1/sqrt(d_k), which is the standard scaling
                factor used in transformer attention mechanisms.

                This scaling prevents the dot products in attention from growing too large,
                which would push the softmax function into regions with small gradients.

                The scaling factor is: 1/sqrt(d_k) where d_k is the key/query dimension.

            TEMPLATE PARAMETERS
                - d_k: The dimension of keys/queries in the attention mechanism
        !*/
    public:
        explicit scale_weights_() : multiply_(1.0f / std::sqrt(static_cast<float>(d_k_))) {}
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

	// ----------------------------------------------------------------------------------------

    /*!
        These templates create the query, key, and value projections used in multi-head
        attention. They project the input of dimension d_model into num_heads parallel
        attention heads, each operating on d_model/num_heads dimensions.

        The output is reshaped to facilitate parallel computation across heads:
        Input shape:  (batch, 1, seq_len, d_model)
        Output shape: (batch, num_heads, seq_len, d_model/num_heads)
    !*/
    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using query = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using key = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    template <long seq_len, long d_model, long num_heads, typename SUBNET>
    using value = reshape_to<num_heads, seq_len, d_model / num_heads, linear_no_bias<d_model, SUBNET>>;

    // ----------------------------------------------------------------------------------------

    /*!
        WHAT THIS REPRESENTS
            This template implements a complete multi-head self-attention mechanism with
            causal masking, rotary positional embeddings (RoPE), and post-attention
            normalization.

            The attention mechanism computes:
                Attention(Q, K, V) = softmax((Q*K^T) / sqrt(d_k)) * V

            Where Q, K, V are the Query, Key, and Value projections respectively.

        ARCHITECTURE FLOW
            1. Input is split into Query, Key, and Value projections
            2. RoPE is applied to Query and Key for positional encoding
            3. Scaled dot-product attention: Q*K^T / sqrt(d_head)
            4. Causal masking (tril_mask) prevents attending to future positions
            5. Softmax normalization across the sequence dimension
            6. Attention weights multiply Values: softmax(scores)·V
            7. Reshape and project back to d_model dimension
            8. Residual connection with input
            9. RMS normalization

        TEMPLATE PARAMETERS
            - ACT: Activation function template (e.g., gelu, relu)
            - DO: Dropout policy template (e.g., dropout_10, multiply for inference)
            - seq_len: Maximum sequence length (context window size)
            - d_model: Model dimension (must be divisible by num_heads)
            - num_heads: Number of parallel attention heads
            - SUBNET: The input layer type

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, d_model)
            Output: (batch_size, 1, seq_len, d_model)

        NOTES
            - Uses causal masking (tril_mask) for autoregressive generation
            - RoPE is applied to both Query and Key for relative position encoding
            - The d_head per head is d_model / num_heads
            - Attention scores are scaled by 1/sqrt(d_head) for stability
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using multihead_attention =
        rms_norm<add_prev1<
        DO<linear_no_bias<d_model, reshape_to<1, seq_len, d_model,
        multm_prev2<softmaxm<tril_mask<
        scale_weights<d_model/num_heads,
        multm_prev3<
        rope<query<seq_len, d_model, num_heads, skip1<
        tag3<transpose<
        rope<key<seq_len, d_model, num_heads, skip1<
        tag2<value<seq_len, d_model, num_heads,
        tag1<SUBNET>>>>>>>>>>>>>>>>>>>>>;

    // ----------------------------------------------------------------------------------------

    /*!
        WHAT THIS REPRESENTS
            This template implements the position-wise feed-forward network used in
            transformer blocks. It consists of two linear transformations with an
            activation function in between.

            The FFN applies the same transformation independently to each position:
                FFN(x) = activation(x * W1 + b1) * W2 + b2
            where the hidden dimension is typically 4x the model dimension.

        ARCHITECTURE FLOW
            1. Linear projection: d_model -> d_model * 4 (expansion)
            2. Activation function (typically GELU or ReLU)
            3. Dropout (during training)
            4. Linear projection: d_model * 4 -> d_model (compression)
            5. Dropout (during training)
            6. Residual connection with input
            7. RMS normalization

        TEMPLATE PARAMETERS
            - ACT: Activation function template (e.g., gelu, relu, silu)
            - DO: Dropout policy template (e.g., dropout_10, multiply for inference)
            - d_model: Model dimension
            - SUBNET: The input layer type

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, d_model)
            Output: (batch_size, 1, seq_len, d_model)

        NOTES
            - Expansion ratio of 4x is standard in transformers
            - Each position is processed independently (no cross-position interaction)
            - Provides the bulk of the transformer's parameter count
            - The residual connection helps gradient flow during training
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using feed_forward =
        rms_norm<add_prev4<
        DO<linear<d_model, DO<ACT<linear<d_model * 4, tag4<SUBNET>>>>>>>>;

    /*!
        WHAT THIS REPRESENTS
            A complete transformer decoder block combining multi-head self-attention and
            feed-forward network with residual connections and RMS normalization.

        ARCHITECTURE FLOW
            Input -> MultiHeadAttention -> FFN -> Output

            Each sub-layer uses the pattern: RMSNorm(input + SubLayer(input))

        TEMPLATE PARAMETERS
            - ACT: Activation function template (e.g., gelu, relu)
            - DO: Dropout policy template (e.g., dropout_10, multiply for inference)
            - seq_len: Maximum sequence length (context window size)
            - d_model: Model dimension (must be divisible by num_heads)
            - num_heads: Number of parallel attention heads
            - SUBNET: The input layer type

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, d_model)
            Output: (batch_size, 1, seq_len, d_model)

        NOTES
            - Decoder-only architecture with causal masking
            - Uses RMS normalization for improved training stability
            - Cannot be used directly with repeat<> due to multiple template parameters
              (use transformer_stack<> instead for stacking multiple blocks)
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using transformer_block =
        act<feed_forward<ACT, DO, d_model,
        multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>>;

    // ----------------------------------------------------------------------------------------

    /*!
        WHAT THIS REPRESENTS
            Stacks multiple transformer blocks using compile-time recursion.

        TEMPLATE PARAMETERS
            - num_layers: Number of transformer blocks to stack (model depth)
            - ACT: Activation function template
            - DO: Dropout policy template
            - seq_len: Maximum sequence length
            - d_model: Model dimension
            - num_heads: Number of attention heads
            - SUBNET: The input layer type

        TYPICAL USAGE
            Create a 6-layer transformer:

            using my_model =
                loss_multiclass_log<fc<vocab_size,
                transformer_stack<6, gelu, dropout_10, 512, 256, 8,
                token_embeddings<dropout_10, vocab_size, 256,
                input<matrix<long>>>>>>;

        NOTES
            - Automatically adds tag5 at the base for proper compilation
            - Each layer has independent trainable parameters
            - Equivalent to manually nesting num_layers transformer_block definitions
    !*/
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

    // ----------------------------------------------------------------------------------------

    /*!
        WHAT THIS REPRESENTS
            This template creates the input embedding layer for transformer models, combining
            token embeddings with positional encodings to provide both semantic and positional
            information to the model.

            Token embeddings convert discrete token IDs into continuous vector representations,
            while positional encodings add information about the position of each token in
            the sequence. This combination allows the model to understand both what each token
            means and where it appears in the sequence.

        ARCHITECTURE FLOW
            1. Token embedding lookup: maps token IDs to dense vectors of size embedding_length
            2. Positional encoding: adds learnable or fixed position information
            3. Dropout: regularization applied to the combined embeddings (during training)
            4. RMS normalization: stabilizes the embedding magnitudes

        TEMPLATE PARAMETERS
            - DO: Dropout policy template (e.g., dropout_10, multiply for inference)
                  Controls regularization strength on the embeddings
            - num_embeddings: Size of the vocabulary (number of unique tokens)
            - embedding_length: Dimension of the embedding vectors (typically d_model)
            - SUBNET: The input layer type (typically input<matrix<long>>)

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, 1) - matrix of token IDs (long integers)
            Output: (batch_size, 1, seq_len, embedding_length) - continuous embedding vectors

        TYPICAL USAGE
            Used as the first layer in transformer models to convert input tokens:

            using transformer_model =
                fc<vocab_size,
                repeat<6, transformer_block, gelu, dropout_10, seq_len, d_model, num_heads,
                token_embeddings<dropout_10, vocab_size, d_model,
                input<matrix<long>>>>>;

        NOTES
            - The positional_encodings layer adds learnable position information
            - RMS normalization at the end helps with training stability
            - Dropout is applied to prevent overfitting on the vocabulary
            - The embedding_length should match d_model for standard transformers
            - Input tokens should be integers in the range [0, num_embeddings)
    !*/
    template <template <typename> class DO, long num_embeddings, long embedding_length, typename SUBNET>
    using token_embeddings = rms_norm<DO<positional_encodings<
        embeddings<num_embeddings, embedding_length, SUBNET>>>>;
}

#endif // DLIB_DNN_TRANSFORMER_H_