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
        using std_ffn = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Standard position-wise feed-forward network used in transformer blocks.
                Implements a two-layer MLP with one intermediate activation and dropout
                regularization.

            ARCHITECTURE FLOW
                1. Linear expansion: d_model => 4*d_model
                2. Activation function (ACT)
                3. Linear projection: 4*d_model => d_model
                4. Dropout (DO) for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., gelu, silu, relu)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Expansion factor is fixed at 4x (standard transformer practice)
                - Single dropout applied after final projection
                - No normalization inside FFN (handled by transformer_block)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                SwiGLU (Swish-Gated Linear Unit) feed-forward network, an alternative to
                standard FFN with improved performance on language modeling tasks.

            REFERENCE
                Noam Shazeer, "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

            ARCHITECTURE FLOW
                1. Split into two branches from input:
                   - Gate branch: W1 projection => ACT activation
                   - Linear branch: V projection
                2. Element-wise multiplication of branches (Hadamard product)
                3. Final projection: W2 => d_model
                4. Dropout for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (typically silu for true SwiGLU)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)

            NOTES
                - Uses (8*d_model)/3 for hidden dimension (equivalent parameters to 4x expansion)
                - More expressive than standard FFN due to gating mechanism
                - Single dropout applied after final projection
                - ACT is typically silu (Swish) for standard SwiGLU
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
                projections simultaneously (dimension: d_model => 3*d_model), then uses
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
        using std_ffn = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Fused implementation of standard feed-forward network using fc layers with
                automatic dimension flattening for better BLAS/GEMM utilization.

            ARCHITECTURE FLOW
                1. fc layer: d_model => 4*d_model (with dimension flattening)
                2. Activation function (ACT)
                3. fc layer: 4*d_model => d_model
                4. Dropout (DO)
                5. extract operation to restore proper tensor dimensions

            TEMPLATE PARAMETERS
                - ACT: activation function template (e.g., gelu, silu, relu)
                - DO: dropout policy template (dropout_10 in training, multiply in inference)
                - d_model: model dimension (input and output size)

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)
        !*/

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = some_template_expression;
        /*!
            WHAT THIS REPRESENTS
                Fused implementation of SwiGLU using fc layers with automatic dimension
                flattening for better BLAS/GEMM utilization.

            REFERENCE
                Noam Shazeer, "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

            ARCHITECTURE FLOW
                1. fc projections with dimension flattening
                2. Split into gate and linear branches
                3. Element-wise multiplication
                4. Final fc projection with extraction
                5. Dropout for regularization

            TEMPLATE PARAMETERS
                - ACT: activation function template (typically silu)
                - DO: dropout policy template
                - d_model: model dimension

            INPUT/OUTPUT SHAPES
                Input:  (batch_size, 1, seq_len, d_model)
                Output: (batch_size, 1, seq_len, d_model)
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
                    For each high-level cycle n E [0, N):
                        For each low-level step t E [0, T):
                            z_L^{n,t} = f_L(z_L^{prev} + z_H^n + x0)
                        z_H^{n+1} = f_H(z_H^n + z_L^{n,T-1})
                    Output: z_H^N

                where:
                    - x0 is the input with positional encodings
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

    // ----------------------------------------------------------------------------------------

    // Tags and type definitions for Mixture of Experts (MoE)
    struct training_mode_tag {};
    struct inference_mode_tag {};

    template <long num_experts, template <typename> class DO, typename SUBNET>
    using gate = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            Gating network that learns to route inputs to experts in a Mixture of Experts model.
            Produces a probability distribution over experts using a learned hierarchical function
            with multiple fully-connected layers and dropout for regularization.

        TEMPLATE PARAMETERS
            - num_experts: number of experts to route between
            - DO: dropout policy template (e.g., dropout_10 for training, multiply for inference)

        OUTPUT
            Tensor with shape (batch_size, num_experts, 1, 1) containing softmax-normalized
            expert selection probabilities for each sample in the batch.
    !*/

    template
        typename EXPERT_NET,
        long top_k,
        typename MODE,
        template<typename> class TAG,
        typename SUBNET
>
class moe_
    {
        /*!
            REQUIREMENTS ON TEMPLATE PARAMETERS
                - EXPERT_NET must be a valid Dlib network type that can process tensors
                  through its forward() and back_propagate_error() methods
                - top_k >= 0 (use 0 for automatic selection of 20% of available experts)
                - MODE must be either training_mode_tag or inference_mode_tag
                - TAG must be a valid layer tag template (e.g., tag9, tag8, etc.)
                - SUBNET must be a valid Dlib network type
                - The gate network referenced by TAG must output a tensor with shape
                  (batch_size, num_experts, 1, 1)

            WHAT THIS OBJECT REPRESENTS
                This layer implements a Mixture of Experts (MoE) architecture with per-sample
                dynamic routing. Each input sample in a batch independently selects and routes
                through its own subset of specialized expert networks. This enables:

                - Conditional computation: only top-k experts are activated per sample,
                  reducing computational cost while increasing model capacity
                - Per-sample routing: each sample can route to different experts based on
                  its learned gating probabilities, enabling sample-specific specialization
                - Load balancing: Auxiliary loss encourages balanced expert utilization
                  across the batch to prevent expert collapse
                - Mode-specific behavior: training mode includes exploration noise and
                  usage tracking, while inference mode is deterministic

            ROUTING MECHANISM
                Unlike batch-wide routing that selects the same experts for all samples, this
                implementation performs independent routing for each sample:

                For each sample in the batch:
                1. Read that sample's gate probabilities (from gate network via TAG)
                2. Optionally add exploration noise (training mode only)
                3. Select top-k experts with highest probabilities for a specific sample
                4. Normalize the selected expert weights
                5. Route the selected sample through its selected experts
                6. Combine expert outputs using weighted averaging

                This per-sample routing allows different samples to utilize different experts,
                providing fine-grained specialization and better model capacity utilization.

            FORWARD PASS DETAILS
                The forward pass processes each sample independently:
                1. Extract gate probabilities for each sample from layer<TAG>(sub).get_output()
                2. For each sample:
                   a. Add exploration noise to probabilities (training mode only)
                   b. Select top-k experts with highest (possibly noisy) scores
                   c. Normalize selected expert weights to sum to 1
                   d. Route sample through selected experts
                   e. Accumulate weighted expert outputs
                3. Track expert usage statistics across batch for load balancing

            BACKWARD PASS DETAILS
                The backward pass mirrors the forward routing:
                1. For each sample:
                   a. Reconstruct which experts were selected (matching forward logic)
                   b. Create weighted gradient by scaling input gradient by expert weight
                   c. Add auxiliary load balancing gradient (training mode)
                   d. Backpropagate through activated experts only
                   e. Accumulate weighted expert gradients to input
                2. Update exponential moving average of expert usage

            LOAD BALANCING
                To prevent expert collapse (where few experts dominate), an auxiliary loss
                is computed as the coefficient of variation of expert usage:

                    CV = sqrt(variance(usage)) / mean(usage)

                This encourages experts to be utilized equally across the dataset. The auxiliary
                loss gradient is added to expert outputs during backpropagation and flows back
                to the gate network, incentivizing more balanced routing decisions.

            TEMPLATE PARAMETERS
                - EXPERT_NET: network architecture for each expert (e.g., feed-forward block)
                - top_k: number of experts to activate per sample (0 auto-select 20%)
                - MODE: compile-time mode tag (training_mode_tag or inference_mode_tag)
                - TAG: layer tag for accessing gate network output

            HYPERPARAMETERS
                The following hyperparameters control MoE behavior (set to sensible defaults):
                - balance_loss_weight (0.01): weight for auxiliary load balancing loss
                - noise_scale (0.2): standard deviation of exploration noise (training only)
                - usage_update_rate (0.05): EMA smoothing factor for usage statistics
        !*/

    public:
        explicit moe_();
        /*!
            ensures
                - #num_experts() == 0 (experts are created during setup() based on gate)
                - #num_active_experts() == top_k (or will be auto-selected if top_k == 0)
                - Initializes hyperparameters with default values:
                    * balance_loss_weight = 0.01
                    * noise_scale = 0.2 (for training exploration)
                    * usage_update_rate = 0.05 (for EMA tracking)
        !*/

        moe_(const moe_& other);
        /*!
            ensures
                - Performs deep copy of all expert networks and state
        !*/

        moe_& operator=(const moe_& other);
        /*!
            ensures
                - Performs deep copy assignment
        !*/

        template <typename SUBNET_TYPE>
        void setup(const SUBNET_TYPE& sub);
        /*!
            requires
                - SUBNET_TYPE implements the SUBNET interface
                - layer<TAG>(sub).get_output() returns a tensor with shape (N, E, 1, 1)
                  where N is batch size and E is the number of experts
            ensures
                - Initializes the MoE layer based on gate network output:
                    * Creates E expert network instances (E = gate output dimension k)
                    * #num_experts() == E
                    * If top_k == 0: #num_active_experts() == max(1, floor(E * 0.2))
                    * If top_k > 0: #num_active_experts() == min(top_k, E)
        !*/

        template <typename SUBNET_TYPE>
        void forward(const SUBNET_TYPE& sub, resizable_tensor& output);
        /*!
            requires
                - setup(sub) has been called at least once
                - sub.get_output() is a valid tensor that experts can process
                - layer<TAG>(sub).get_output() has shape (batch_size, num_experts(), 1, 1)
                  containing valid probability values
            ensures
                - Performs per-sample expert routing and computation:
                    * For each sample in the batch:
                      - Extracts that sample's gate probabilities
                      - Adds exploration noise if MODE == training_mode_tag && noise_scale > 0
                      - Selects top_n experts with highest (possibly noisy) scores
                      - Normalizes selected expert weights to sum to 1
                      - Routes sample through selected experts
                      - Combines expert outputs using weighted averaging
                - #output has same dimensions as sub.get_output():
                    * #output.num_samples() == sub.get_output().num_samples()
                    * #output.k() == sub.get_output().k()
                    * #output.nr() == sub.get_output().nr()
                    * #output.nc() == sub.get_output().nc()
                - If MODE == training_mode_tag && usage_update_rate > 0:
                    * Updates expert usage statistics using exponential moving average
                    * These statistics are used for load balancing computation
                - Only num_active_experts() experts are activated per sample (sparse computation)
        !*/

        template <typename SUBNET_TYPE>
        void backward(
            const tensor& gradient_input,
            SUBNET_TYPE& sub,
            tensor& params_grad
        );
        /*!
            requires
                - setup(sub) has been called
                - forward(sub, output) was previously called with the same sub
                - gradient_input has same dimensions as the forward() output
                - All tensors are properly allocated
            ensures
                - Backpropagates gradients through the MoE layer:
                    * For each sample:
                      - Reconstructs which experts were activated (matching forward)
                      - Creates weighted gradient by scaling input gradient by expert weight
                      - Adds auxiliary load balancing gradient if MODE == training_mode_tag
                      - Backpropagates through activated experts only
                      - Accumulates weighted expert gradients to sub.get_gradient_input()
                - If MODE == training_mode_tag:
                    * Computes auxiliary load balancing loss as coefficient of variation
                    * Adds auxiliary loss gradient to encourage balanced expert usage
                    * This gradient flows back through experts to the gate network
                - Only backpropagates through num_active_experts() that were activated
                - Gradients automatically flow to gate network via Dlib's computation graph
        !*/

        void clean();
        /*!
            ensures
                - Calls clean() on each expert network (if they implement this method)
                - Prepares the network for inference or serialization
                - This is typically called after training completes
        !*/

        EXPERT_NET& get_expert(size_t idx);
        const EXPERT_NET& get_expert(size_t idx) const;
        /*!
            requires
                - idx < num_experts()
            ensures
                - Returns reference to the expert network at index idx
                - Can be used for inspection, modification, or expert-specific initialization
                - Useful for analyzing individual expert specialization
        !*/

        long num_experts() const;
        /*!
            ensures
                - Returns the total number of expert networks in this MoE layer
        !*/

        long num_active_experts() const;
        /*!
            ensures
                - Returns the number of experts activated per sample (top-k)
        !*/

        bool is_training_mode() const;
        /*!
            ensures
                - Returns true if MODE == training_mode_tag
                - Returns false if MODE == inference_mode_tag
        !*/

        const std::vector<float>& get_expert_usage() const;
        /*!
            ensures
                - Returns the exponential moving average of expert usage statistics
        !*/

        friend void serialize(const moe_& item, std::ostream& out);
        friend void deserialize(moe_& item, std::istream& in);
        /*!
            ensures
                - Provides serialization support for the MoE layer.
        !*/

        friend std::ostream& operator<<(std::ostream& out, const moe_& item);
        friend void to_xml(const moe_& item, std::ostream& out);
        /*!
            ensures
                - Writes a human-readable summary to the output stream
        !*/
    };

    template<
        typename EXPERT_NET,
        long top_k,
        typename MODE,
        template<typename> class TAG,
        typename SUBNET
    >
    using moe = add_layer<moe_<EXPERT_NET, top_k, MODE, TAG, SUBNET>, SUBNET>;

    template<
        typename EXPERT_NET,
        long num_experts,
        long top_k,
        typename MODE,
        template <typename> class DO,
        typename SUBNET
    >
    using moe_ffn = some_template_expression;
    /*!
        WHAT THIS OBJECT REPRESENTS
            A drop-in replacement for transformer feed-forward layers using MoE architecture.
            Combines gating, expert routing, and skip connections in a single template.

        TEMPLATE PARAMETERS
            - EXPERT_NET: expert network architecture (typically feed-forward network)
            - num_experts: total number of experts in the mixture
            - top_k: number of experts to activate per forward pass
            - MODE: training_mode_tag or inference_mode_tag
            - DO: dropout policy template
    !*/
}

#endif // DLIB_DNN_TRANSFORMER_H_