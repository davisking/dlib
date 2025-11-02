// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_TRANSFORMER_ABSTRACT_H_
#ifdef DLIB_DNN_TRANSFORMER_ABSTRACT_H_

#include "layers_abstract.h"

/*!
    The transformer.h file contains specialized layers and building blocks designed
    specifically for transformer architectures and attention mechanisms.

    Two architectural variants are provided:

    1. CANONICAL TRANSFORMER (namespace canonical_transformer):
       - Separate Q, K, V projections using linear_no_bias
       - Explicit reshape operations
       - More modular, easier to understand
       - Suitable for fine-grained control

    2. FUSED TRANSFORMER (namespace fused_transformer):
       - Combined QKV projection
       - Extraction-based separation
       - Optimized for performance and memory efficiency
!*/

namespace dlib
{

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
    };

    template <long d_k, typename SUBNET>
    using scale_weights = add_layer<scale_weights_<d_k>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    template <template <typename> class ACT, long reduction_factor, long d_model, typename SUBNET>
    using projection_head = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Reduces dimensionality before the final output layer to minimize parameters
            when vocab_size is large. Performs intermediate projection:
            d_model => d_model/reduction_factor using fully-connected layer.

        TEMPLATE PARAMETERS
            - ACT: activation function (gelu, relu, or silu recommended)
            - reduction_factor: compression ratio (3 recommended, typically 2-4)
            - d_model: input dimension from transformer stack

        TYPICAL USAGE
            using my_model =
                loss_multiclass_log<fc<vocab_size,
                projection_head<gelu, 3, d_model,
                transformer_stack<6, gelu, dropout_10, seq_len, d_model, num_heads,
                token_embeddings<vocab_size, d_model,
                input<matrix<int, 0, 1>>>>>>>;

        RECOMMENDATIONS
            - vocab_size > 10,000: Strongly recommended
            - vocab_size > 30,000: Almost mandatory
            - vocab_size < 5,000:  Optional (minimal gains)
    !*/

    // ----------------------------------------------------------------------------------------

    template <long num_embeddings, long embedding_length, typename SUBNET>
    using token_embeddings = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Converts discrete token IDs to continuous embedding vectors with positional
            encoding.

        ARCHITECTURE FLOW
            1. Token embedding lookup: maps token IDs to dense vectors
            2. Positional encoding: adds learnable position information

        TEMPLATE PARAMETERS
            - num_embeddings: vocabulary size (number of unique tokens)
            - embedding_length: embedding dimension (typically d_model)

        INPUT/OUTPUT SHAPES
            Input:  (batch_size, 1, seq_len, 1) - matrix of token IDs (long integers)
            Output: (batch_size, 1, seq_len, embedding_length) - embedding vectors

        TYPICAL USAGE
            using my_model =
                loss_multiclass_log<fc<vocab_size,
                transformer_stack<6, gelu, dropout_10, seq_len, d_model, num_heads,
                token_embeddings<vocab_size, d_model,
                input<matrix<int, 0, 1>>>>>>;

        NOTES
            - Input tokens must be integers in range [0, num_embeddings)
            - embedding_length should match d_model for transformer architectures
    !*/

    namespace canonical_transformer
    {
        /*!
            WHAT THIS REPRESENTS
                Standard transformer implementation with separate Q, K, V projections.

                This architecture uses three independent linear transformations followed
                by reshape operations to create the multi-head attention structure.

                Advantages:
                - Conceptually clearer and more modular
                - Easier to debug and understand
                - Each projection can be independently modified or analyzed

                Use cases:
                - When fine-grained control over each projection is needed
                - Prototyping new attention mechanisms
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Query projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Key projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <long seq_len, long d_model, long num_heads, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Creates Value projection for multi-head attention
                - Output shape: (batch, num_heads, seq_len, d_model/num_heads)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                This template implements a complete multi-head self-attention mechanism with
                causal masking, rotary positional embeddings (RoPE), and post-attention
                normalization.

                The attention mechanism computes:
                    Attention(Q, K, V) = softmax((Q*K^T) / sqrt(d_k)) * V

                Where Q, K, V are the Query, Key, and Value projections respectively.

            ARCHITECTURE FLOW
                1. RMS normalization
                2. Input is split into Query, Key, and Value projections
                3. RoPE is applied to Query and Key for positional encoding
                4. Scaled dot-product attention: Q*K^T / sqrt(d_head)
                5. Causal masking (tril_mask) prevents attending to future positions
                6. Softmax normalization across the sequence dimension
                7. Attention weights multiply Values: softmax(scores)*V
                8. Reshape and project back to d_model dimension
                9. Residual connection with input                

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., silu, gelu, relu)
                - DO: dropout policy template (e.g., dropout_10, multiply for inference)
                - seq_len: maximum sequence length (context window size)
                - d_model: model dimension (must be divisible by num_heads)
                - num_heads: number of parallel attention heads

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
            long d_model, typename SUBNET>
        using feed_forward = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                This template implements the position-wise feed-forward network used in
                transformer blocks. It consists of two linear transformations with an
                activation function in between.

                The FFN applies the same transformation independently to each position:
                    FFN(x) = activation(x * W1 + b1) * W2 + b2
                where the hidden dimension is typically 4x the model dimension.

            ARCHITECTURE FLOW
                1. RMS normalization
                2. Linear projection: d_model => d_model * 4 (expansion)
                3. Activation function
                4. Dropout (during training)
                5. Linear projection: d_model * 4 => d_model (compression)
                6. Dropout (during training)
                7. Residual connection with input                

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., silu, gelu, relu)
                - DO: dropout policy template (e.g., dropout_10, multiply for inference)
                - d_model: model dimension

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
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                A complete transformer decoder block combining multi-head self-attention and
                feed-forward network with residual connections and RMS normalization.

            ARCHITECTURE FLOW
                Input => MultiHeadAttention => FFN => Output
                Each sub-layer uses the pattern: RMSNorm(input + SubLayer(input))

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., silu, gelu, relu)
                - DO: dropout policy template (e.g., dropout_10, multiply for inference)
                - seq_len: maximum sequence length (context window size)
                - d_model: model dimension (must be divisible by num_heads)
                - num_heads: number of parallel attention heads

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Decoder-only architecture with causal masking
                - Uses RMS normalization for improved training stability
                - Cannot be used directly with repeat<> due to multiple template parameters
                  (use transformer_stack<> instead for stacking multiple blocks)
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Stacks multiple transformer blocks using compile-time recursion.

            TEMPLATE PARAMETERS
                - num_layers: number of transformer blocks to stack (model depth)
                - ACT: activation function template
                - DO: dropout policy template
                - seq_len: maximum sequence length
                - d_model: model dimension
                - num_heads: number of attention heads

            TYPICAL USAGE
                Create a 6-layer transformer:

                using my_model =
					loss_multiclass_log<fc<vocab_size, rms_norm<
                    transformer_stack<6, silu, dropout_10, 512, 256, 8,
                    token_embeddings<vocab_size, 256,
                    input<matrix<int, 0, 1>>>>>>>;

            NOTES
                - Each layer has independent trainable parameters
                - Equivalent to manually nesting num_layers transformer_block definitions
        !*/

    } // namespace std_transformer

    namespace fused_transformer
    {

        /*!
            WHAT THIS REPRESENTS
                Optimized transformer implementation with fused QKV projections,
                sometimes referred to as "kernel-fused" attention in the literature

                This architecture uses a single fc_no_bias layer to compute all Q, K, V
                projections simultaneously (dimension: d_model -> 3*d_model), then uses
                extract layers to separate them. This approach leverages Dlib's fc_ layer
                optimizations and reduces memory access patterns.

                Advantages:
                - Single matrix multiplication instead of three
                - Reduced memory bandwidth requirements
                - Better GPU utilization through larger operations

                Performance considerations:
                - Typically 10-30% faster than standard implementation
                - Lower memory footprint during forward/backward passes
                - Better cache utilization
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using query = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Query projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, d_model/num_heads, 1)
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using key = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Key projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, 1, d_model/num_heads)
        !*/

        template <long num_heads, long d_model, typename SUBNET>
        using value = some_template_expression;
        /*!
            requires
                - d_model % num_heads == 0
            ensures
                - Extracts Value projection from fused QKV output
                - Uses extract layer for efficient separation
                - Output shape: (batch, num_heads, d_model/num_heads, 1)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using multihead_attention = some_template_expression;
        /*!
            WHAT THIS OBJECT REPRESENTS
                Optimized multi-head self-attention using fused QKV projection.
                Functionally equivalent to canonical version but with better performance.

            ARCHITECTURE FLOW
                1. RMS normalization
                2. Single fused projection: d_model => 3*d_model for Q, K, V
                3. Extract Q, K, V from combined output
                4. Compute attention with causal masking
                5. Concatenate heads and project
                6. Residual connection and normalization

            TEMPLATE PARAMETERS
                - ACT: activation function (for compatibility)
                - DO: dropout policy
                - d_model: model dimension
                - num_heads: number of attention heads

            PERFORMANCE NOTES
                - Typically 10-20% faster than canonical version
                - Better memory access patterns
                - Reduced parameter initialization overhead
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using feed_forward = some_template_expression;
        /*!
            WHAT THIS OBJECT REPRESENTS
                Optimized feed-forward network using fc layers with dimension flattening.

            ARCHITECTURE FLOW
                Same as canonical version but uses fc instead of linear:
                1. RMS normalization
                2. fc expansion: d_model => d_model * 4
                3. Activation function
                4. Dropout
                5. fc compression: d_model * 4 => d_model
                6. Dropout
                7. Residual connection

            PERFORMANCE NOTES
                - Better utilization of optimized BLAS routines
                - Dimension flattening improves memory access
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = some_template_expression;
        /*!
            Same interface as canonical_transformer::transformer_block but with
            optimized implementation using fused operations.
        !*/

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = some_template_expression;
        /*!
            Same interface as canonical_transformer::transformer_stack but with
            optimized implementation using fused operations.
        !*/

    } // namespace fused_transformer

    // ----------------------------------------------------------------------------------------

    template<
        typename H_NET,
        typename L_NET,
        int N,
        int T
    >
        class hrm_
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - H_NET must be a valid dlib network type (complete network with input layer)
                - L_NET must be a valid dlib network type (complete network with input layer)
                - N > 0 (number of high-level cycles)
                - T > 0 (number of low-level steps per cycle)

            WHAT THIS OBJECT REPRESENTS
                This object implements a Hierarchical Reasoning Model (HRM) layer, a dual-
                recurrent architecture inspired by hierarchical and multi-timescale processing
                in cognitive systems.

                The model consists of two interdependent recurrent modules:
                    - High-level module (H_NET): executes N slow cycles for abstract planning
                      and global reasoning
                    - Low-level module (L_NET): executes T fast iterations per H-cycle for
                      detailed, rapid computations

                During forward propagation, the network performs N×T total recurrent steps
                with hierarchical convergence. For each of the N high-level cycles, the
                low-level module performs T iterations, converging locally before the
                high-level module updates.

                Mathematical formulation:
                    For each high-level cycle n ∈ [0, N):
                        For each low-level step t ∈ [0, T):
                            z_L^{n,t} = f_L(z_L^{prev} + z_H^n + x̃)
                        z_H^{n+1} = f_H(z_H^n + z_L^{n,T-1})
                    Output: z_H^N

                where:
                    - x̃ is the input with positional encodings
                    - z_H and z_L are the hidden states of the H and L modules
                    - f_H and f_L are the recurrent transformations (H_NET and L_NET)

                The backward pass uses a one-step gradient approximation, computing gradients
                only through the final update of each module. This provides O(1) memory
                complexity instead of O(N×T) required by full Backpropagation Through Time
                (BPTT), while maintaining training stability.

                Key features:
                    - Hierarchical processing with temporal separation of concerns
                    - Memory-efficient training (O(1) vs O(N×T) for BPTT)
                    - Biologically-plausible recurrent computation
                    - Suitable for complex reasoning tasks requiring iterative refinement

                References:
                    - Wang et al., "Hierarchical Reasoning Model", arXiv:2506.21734
                    - Bai et al., "Deep Equilibrium Models", NeurIPS 2019
        !*/

    public:

        hrm_();
        /*!
            ensures
                - #seq_len == 0
                - #hidden_dim == 0
                - Internal networks (h_net, l_net) are default-constructed
        !*/

        template <typename SUBNET>
        void setup(
            const SUBNET& sub
        );
        /*!
            ensures
                - Initializes the internal H and L networks based on input dimensions
                - Initializes hidden state vectors z_h_init and z_l_init with truncated
                  normal distribution (std=1, truncated at ±2)
                - Stores sequence length and hidden dimension from input
        !*/

        template <typename SUBNET>
        void forward(
            const SUBNET& sub,
            resizable_tensor& output
        );
        /*!
            ensures
                - Performs hierarchical recurrent computation:
                    * N high-level cycles, each with T low-level steps
                    * Total of N×T recurrent iterations
                    * All but the last iteration executed without gradient tracking
                    * Final iteration computes gradients for one-step approximation
                - #output contains the final high-level state z_H^{NT}
                - output has the same dimensions as sub.get_output()
        !*/

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor& params_grad
        );
        /*!
            ensures
                - Performs one-step gradient approximation:
                    * Backpropagates through final H-module update
                    * Backpropagates through final L-module update
                    * Accumulates gradients to input
                - Memory complexity: O(1) instead of O(N×T) for full BPTT
        !*/

        const h_net_type& get_h_net() const;
        h_net_type& get_h_net();
        const l_net_type& get_l_net() const;
        l_net_type& get_l_net();
        /*!
            ensures
                - Returns a reference to the high-level (H) or low-level (L) network
                - Allows inspection and manipulation of internal modules
        !*/

        const tensor& get_layer_params() const;
        tensor& get_layer_params();
        /*!
            ensures
                - Returns the parameters tensor
                - Note: hrm_ has no direct trainable parameters; all parameters
                  are contained within H_NET and L_NET
        !*/
    };

    template<typename H_NET, typename L_NET, int N, int T, typename SUBNET>
    using hrm = add_layer<hrm_<H_NET, L_NET, N, T>, SUBNET>;    

}

#endif // DLIB_DNN_TRANSFORMER_H_