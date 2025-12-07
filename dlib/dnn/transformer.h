// Copyright (C) 2025  Cydral Technology (cydraltechnology@gmail.com)
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
    
    template <long num_embeddings, long embedding_length, typename SUBNET>
    using positional_embeddings = positional_encodings<
        embeddings<num_embeddings, embedding_length, SUBNET>>;

    // ----------------------------------------------------------------------------------------

    // CANONICAL TRANSFORMER ARCHITECTURE
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
        using multihead_attention =
            DO<linear_no_bias<d_model, reshape_to<1, seq_len, d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            rope<query<seq_len, d_model, num_heads, skip1<
            tag4<transpose<
            rope<key<seq_len, d_model, num_heads, skip2<
            tag3<value<seq_len, d_model, num_heads,
            tag2<SUBNET>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using std_ffn = DO<linear<d_model, ACT<linear<d_model * 4, SUBNET>>>>;

        // Standard SwiGLU FFN implementation
        // Reference: Noam Shazeer's "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
        template <template <typename> class DO, long d_model, typename SUBNET>
        using swiglu = DO<linear<d_model, mult_prev7<linear<(d_model * 2) / 7, skip6<
            tag7<silu<linear<(d_model * 2) / 7, tag6<SUBNET>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = 
            add_prev5<std_ffn<ACT, DO, d_model, rms_norm<tag5<
            add_prev1<multihead_attention<ACT, DO, seq_len, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>>>>;

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
            using type = tag10<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, seq_len, d_model, num_heads, SUBNET>::type;

    } // namespace std_transformer

    // FUSED TRANSFORMER ARCHITECTURE
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
        using multihead_attention =
            DO<extract<0, 1, 1, d_model, fc_no_bias<d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            query<num_heads, d_model, skip1<
            tag4<key<num_heads, d_model, skip2<
            tag3<value<num_heads, d_model,
            tag2<fc_no_bias<d_model * 3,
            SUBNET>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using std_ffn = extract<0, 1, 1, d_model,
            DO<fc<d_model, ACT<fc<d_model * 4, SUBNET>>>>>;

        template <template <typename> class DO, long d_model, typename SUBNET>
        using swiglu = extract<0, 1, 1, d_model,
            DO<fc<d_model, mult_prev7<fc<(d_model * 2) / 7, skip6<
            tag7<silu<fc<(d_model * 2) / 7, tag6<SUBNET>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_block = 
            add_prev5<std_ffn<ACT, DO, d_model, rms_norm<tag5<
            add_prev1<multihead_attention<ACT, DO, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>>>>;

        template<long remaining_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET, typename enabled = void>
        struct transformer_stack_impl
        {
            using type = transformer_block<ACT, DO, d_model, num_heads,
                typename transformer_stack_impl<remaining_layers - 1, ACT, DO, d_model, num_heads, SUBNET>::type>;
        };

        template<template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        struct transformer_stack_impl<0, ACT, DO, d_model, num_heads, SUBNET, void>
        {
            using type = tag10<SUBNET>;
        };

        template<long num_layers, template <typename> class ACT, template <typename> class DO,
            long d_model, long num_heads, typename SUBNET>
        using transformer_stack = typename transformer_stack_impl<num_layers, ACT, DO, d_model, num_heads, SUBNET>::type;

    } // namespace fused_transformer

    // Default to canonical transformer implementation
    using namespace canonical_transformer;

	// ----------------------------------------------------------------------------------------

    // HIERARCHICAL REASONING MODEL (HRM)
    template<
        typename H_NET,
        typename L_NET,
        int N,
        int T
    >
        class hrm_
    {
        static_assert(N > 0, "N (high-level cycles) must be positive");
        static_assert(T > 0, "T (low-level timesteps per cycle) must be positive");

    public:
        using h_net_type = H_NET;
        using l_net_type = L_NET;

        explicit hrm_() :
            seq_len(0),
            hidden_dim(0),
            learning_rate_multiplier(1.0)
        {
        }

        hrm_(const hrm_& other) :
            h_net(other.h_net),
            l_net(other.l_net),
            z_h_init(other.z_h_init),
            z_l_init(other.z_l_init),
            seq_len(other.seq_len),
            hidden_dim(other.hidden_dim),
            learning_rate_multiplier(other.learning_rate_multiplier)
        {
        }

        hrm_& operator=(const hrm_& other)
        {
            if (this != &other) {
                h_net = other.h_net;
                l_net = other.l_net;
                z_h_init = other.z_h_init;
                z_l_init = other.z_l_init;
                seq_len = other.seq_len;
                hidden_dim = other.hidden_dim;
                learning_rate_multiplier = other.learning_rate_multiplier;
            }
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            const tensor& input = sub.get_output();

            // Store dimensions for initialization
            seq_len = input.nr();
            hidden_dim = input.nc();

            // Initialize hidden states with truncated normal (std=1, trunc=2)
            init_hidden_states();
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& x = sub.get_output();
            const long batch_size = x.num_samples();
            const long k = x.k();

            // Allocate working tensors with proper batch size
            z_h_current.copy_size(x);
            z_l_current.copy_size(x);

            // Broadcast initial states to all samples and positions
            // Initialize each (sample, k, row, col) with the same initial vector
            auto* z_h_ptr = z_h_current.host();
            auto* z_l_ptr = z_l_current.host();
            const auto* h_init_ptr = z_h_init.host();
            const auto* l_init_ptr = z_l_init.host();

            for (long n = 0; n < batch_size; ++n) {
                for (long kk = 0; kk < k; ++kk) {
                    for (long r = 0; r < seq_len; ++r) {
                        for (long c = 0; c < hidden_dim; ++c) {
                            const long idx = ((n * k + kk) * seq_len + r) * hidden_dim + c;
                            z_h_ptr[idx] = h_init_ptr[c];
                            z_l_ptr[idx] = l_init_ptr[c];
                        }
                    }
                }
            }

            // Main HRM recurrent loop (N×T iterations, all but last without gradients)
            for (int n = 0; n < N; ++n)
            {
                for (int t = 0; t < T; ++t)
                {
                    // Skip last iteration (computed with gradients after loop)
                    if (n == N - 1 && t == T - 1) continue;

                    // L-Module: z_L' = f_L(z_L + z_H + x)
                    l_input.copy_size(x);
                    tt::copy_tensor(false, l_input, 0, z_l_current, 0, z_l_current.k());
                    tt::add(1.0f, l_input, 1.0f, z_h_current);
                    tt::add(1.0f, l_input, 1.0f, x);

                    l_net.forward(l_input);
                    const tensor& l_out = l_net.get_output();
                    tt::copy_tensor(false, z_l_current, 0, l_out, 0, l_out.k());
                }

                // Skip last H-Module update (computed with gradients after loop)
                if (n == N - 1) continue;

                // H-Module: z_H' = f_H(z_H + z_L)
                h_input.copy_size(x);
                tt::copy_tensor(false, h_input, 0, z_h_current, 0, z_h_current.k());
                tt::add(1.0f, h_input, 1.0f, z_l_current);

                h_net.forward(h_input);
                const tensor& h_out = h_net.get_output();
                tt::copy_tensor(false, z_h_current, 0, h_out, 0, h_out.k());
            }

            // Final L-Module update
            last_l_input.copy_size(x);
            tt::copy_tensor(false, last_l_input, 0, z_l_current, 0, z_l_current.k());
            tt::add(1.0f, last_l_input, 1.0f, z_h_current);
            tt::add(1.0f, last_l_input, 1.0f, x);

            l_net.forward(last_l_input);
            const tensor& l_final = l_net.get_output();

            // Final H-Module update
            last_h_input.copy_size(x);
            tt::copy_tensor(false, last_h_input, 0, z_h_current, 0, z_h_current.k());
            tt::add(1.0f, last_h_input, 1.0f, l_final);

            h_net.forward(last_h_input);
            const tensor& h_final = h_net.get_output();

            // Output is final high-level state z_H^{NT}
            output.copy_size(h_final);
            tt::copy_tensor(false, output, 0, h_final, 0, h_final.k());
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            // Backprop through final H-Module update
            h_net.back_propagate_error(last_h_input, gradient_input);
            const tensor& grad_h = h_net.get_gradient_input();

            // Backprop through final L-Module update
            // Gradient from H-Module flows to z_L (and z_H_prev which we ignore)
            l_net.back_propagate_error(last_l_input, grad_h);
            const tensor& grad_l = l_net.get_gradient_input();

            // Propagate gradient to input x (and z_L_prev, z_H_prev which we ignore)
            tensor& prev_grad = sub.get_gradient_input();
            tt::add(1.0f, prev_grad, 1.0f, grad_l);
        }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;            
            set_all_learning_rate_multipliers(h_net, val);
            set_all_learning_rate_multipliers(l_net, val);
        }
        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }


        // Cleans up the internal state of H and L networks
        void clean()
        {
            clean_subnet(h_net);
            clean_subnet(l_net);
        }

        // Returns the H/L module network
        const h_net_type& get_h_net() const { return h_net; }
        const l_net_type& get_l_net() const { return l_net; }
        h_net_type& get_h_net() { return h_net; }
        l_net_type& get_l_net() { return l_net; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const hrm_& item, std::ostream& out)
        {
            serialize("hrm_", out);
            serialize(item.h_net, out);
            serialize(item.l_net, out);
            serialize(item.z_h_init, out);
            serialize(item.z_l_init, out);
            serialize(item.seq_len, out);
            serialize(item.hidden_dim, out);
            serialize(item.learning_rate_multiplier, out);
        }

        friend void deserialize(hrm_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "hrm_")
                throw serialization_error("Unexpected version '" + version + "' while deserializing hrm_");

            deserialize(item.h_net, in);
            deserialize(item.l_net, in);
            deserialize(item.z_h_init, in);
            deserialize(item.z_l_init, in);
            deserialize(item.seq_len, in);
            deserialize(item.hidden_dim, in);
            deserialize(item.learning_rate_multiplier, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const hrm_& item)
        {
            out << "hrm\t ("
                << "N=" << N
                << ", T=" << T
                << ")";
            out << " learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const hrm_& item, std::ostream& out)
        {
            out << "<hrm"
                << " N='" << N << "'"
                << " T='" << T << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier << "'"
                << ">\n";
            out << "  <h_module>\n";
            to_xml(item.h_net, out);
            out << "  </h_module>\n";
            out << "  <l_module>\n";
            to_xml(item.l_net, out);
            out << "  </l_module>\n";
            out << "</hrm>\n";
        }

    private:
        void init_hidden_states()
        {
            // Initialize single vector for H and L (will be broadcast to full tensor)
            // Shape: (1, 1, 1, hidden_dim) - single vector per dimension
            z_h_init.set_size(1, 1, 1, hidden_dim);
            z_l_init.set_size(1, 1, 1, hidden_dim);

            dlib::rand rnd(std::time(0));

            auto* h_ptr = z_h_init.host();
            auto* l_ptr = z_l_init.host();

            // Truncated normal initialization (std=1, trunc=2)
            for (long c = 0; c < hidden_dim; ++c) {
                float h_val, l_val;
                do {
                    h_val = rnd.get_random_gaussian();
                } while (std::abs(h_val) > 2.0f);

                do {
                    l_val = rnd.get_random_gaussian();
                } while (std::abs(l_val) > 2.0f);

                h_ptr[c] = h_val;
                l_ptr[c] = l_val;
            }
        }

        template<typename NET>
        auto clean_subnet(NET& net) -> decltype(net.clean(), void())
        {
            net.clean();
        }
        template<typename NET>
        void clean_subnet(...) {}

        // Internal recurrent modules
        h_net_type h_net;
        l_net_type l_net;

        // Initial hidden states (persistent, updated after each forward)
        resizable_tensor z_h_init;
        resizable_tensor z_l_init;

        // Dimensions and learning rate
        long seq_len;
        long hidden_dim;
        double learning_rate_multiplier;

        // Temporary computation tensors
        resizable_tensor z_h_current;
        resizable_tensor z_l_current;
        resizable_tensor h_input;
        resizable_tensor l_input;

        // Saved for one-step gradient backward
        resizable_tensor last_h_input;
        resizable_tensor last_l_input;

        resizable_tensor params; // No direct trainable parameters
    };

    template<typename H_NET, typename L_NET, int N, int T, typename SUBNET>
    using hrm = add_layer<hrm_<H_NET, L_NET, N, T>, SUBNET>;

    // ----------------------------------------------------------------------------------------

    // Gate network: produces raw logits for expert selection
    template <long num_experts, template <typename> class DO, typename SUBNET>
    using gate = fc<num_experts, DO<leaky_relu<fc<num_experts * 8, SUBNET>>>>;

    struct training_mode_tag {};
    struct inference_mode_tag {};

    template<
        typename EXPERT_NET,                    // Expert network architecture
        long top_e,                             // Number of experts to activate (0 = auto: 20%)
        typename MODE,                          // Tag-based mode selection (training/inference)
        template<typename> class TAG,           // Tag for gate input location
        typename SUBNET                         // Input subnet type
    >
    class moe_
    {
    public:
        /*!
            Mixture of Experts layer with sample-wise expert routing.

            Key features:
            - Each sample independently selects top-k experts via gating network
            - Gate produces logits, optional noise added before softmax (training only)
            - Forward/backward consistency via cached expert selections
            - Tracks expert usage statistics for monitoring

            Hyperparameters:
            - noise_scale: Gaussian noise std applied to gate logits (exploration)
            - usage_update_rate: EMA smoothing for usage statistics
        !*/
        explicit moe_() :
            n_experts(0),
            noise_scale(0.1f),
            top_k(top_e),
            usage_update_rate(0.05f),
            load_balance_weight(0.01f),
            learning_rate_multiplier(1.0),
            cached_batch_size_(0)
        {
        }

        moe_(const moe_& other) :
            n_experts(other.n_experts),
            noise_scale(other.noise_scale),
            top_k(other.top_k),
            usage_update_rate(other.usage_update_rate),
            load_balance_weight(other.load_balance_weight),
            learning_rate_multiplier(other.learning_rate_multiplier),
            expert_usage(other.expert_usage),
            cached_batch_size_(0)
        {
            // Deep copy of expert networks
            experts.reserve(other.experts.size());
            for (const auto& expert : other.experts)
                experts.push_back(expert);
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                n_experts = other.n_experts;
                noise_scale = other.noise_scale;
                top_k = other.top_k;
                usage_update_rate = other.usage_update_rate;
                load_balance_weight = other.load_balance_weight;
                learning_rate_multiplier = other.learning_rate_multiplier;
                expert_usage = other.expert_usage;
                cached_batch_size_ = 0;

                // Deep copy of expert networks
                experts.clear();
                experts.reserve(other.experts.size());
                for (const auto& expert : other.experts)
                    experts.push_back(expert);
            }
            return *this;
        }

        /*!
            SETUP
                Initializes expert networks based on gate output dimensions.
                - Number of experts automatically determined from gate output channels
                - If top_e == 0 (auto mode), activates 20% of experts (minimum 1)
        !*/
        template <typename SUBNET_TYPE>
        void setup(const SUBNET_TYPE& sub) {
            const tensor& gate_output = layer<TAG>(sub).get_output();
            long new_n_experts = gate_output.k();

            // Initialize experts if needed
            if (new_n_experts != n_experts) {
                n_experts = new_n_experts;
                expert_usage.resize(n_experts, 0.0f);

                // Create expert network instances
                experts.clear();
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i)
                    experts.emplace_back(EXPERT_NET{});

                // Determine top-k activation count
                if (top_e == 0) {
                    // Auto mode: activate 20% of experts (minimum 1)
                    top_k = std::max(1L, static_cast<long>(std::floor(n_experts * 0.2f)));
                }
                else {
                    top_k = std::min(top_e, n_experts);
                }
            }
        }

        /*!
            FORWARD PASS
                Sample-wise expert routing with optional exploration noise.

                Process per sample:
                1. Retrieve gate logits for this sample
                2. Add Gaussian noise to logits (training only, if noise_scale > 0)
                3. Apply softmax to obtain expert probabilities
                4. Select top-k experts with highest probabilities
                5. Renormalize top-k weights to sum to 1
                6. Route sample through selected experts with weighted combination
                7. Cache expert indices and weights for backward pass

                The cache ensures forward/backward consistency: backward uses the
                exact same experts and weights, even with stochastic noise.
        !*/
        template <typename SUBNET_TYPE>
        void forward(const SUBNET_TYPE& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const tensor& gate_logits = layer<TAG>(sub).get_output();

            DLIB_CASSERT(gate_logits.k() == n_experts &&
                gate_logits.nr() == 1 && gate_logits.nc() == 1,
                "\nExpected gate output shape [batch_size, " << n_experts << ", 1, 1]"
                << "\nReceived shape [" << gate_logits.num_samples() << ", "
                << gate_logits.k() << ", " << gate_logits.nr() << ", "
                << gate_logits.nc() << "]");

            const long num_samples = gate_logits.num_samples();
            const long k = expert_input.k();
            const long nr = expert_input.nr();
            const long nc = expert_input.nc();
            const long sample_size = k * nr * nc;
            const float* logits_data = gate_logits.host();

            // Initialize output tensor
            output.copy_size(expert_input);
            output = 0;

            // Prepare forward pass cache for backward consistency
            if (std::is_same<MODE, training_mode_tag>::value) {
                cached_batch_size_ = num_samples;
                selected_expert_indices_.resize(num_samples);
                selected_expert_weights_.resize(num_samples);
                cached_gate_probs_.resize(num_samples);
            }

            // Track expert usage for monitoring
            std::vector<float> batch_expert_usage(n_experts, 0.0f);
            std::vector<float> routing_fraction(n_experts, 0.0f);
            std::vector<float> gate_prob_sum(n_experts, 0.0f);

            alias_tensor sample_alias(1, k, nr, nc);

            // Process each sample independently with its own expert routing
            for (long n = 0; n < num_samples; ++n) {
                const float* sample_logits = logits_data + n * n_experts;

                // Apply optional Gaussian noise to logits before softmax
                std::vector<float> noisy_logits(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    noisy_logits[e] = sample_logits[e];

                    if (std::is_same<MODE, training_mode_tag>::value && noise_scale > 0) {
                        static thread_local dlib::rand rnd(std::time(0));
                        noisy_logits[e] += noise_scale * rnd.get_random_gaussian();
                    }
                }

                // Softmax: numerically stable implementation
                float max_logit = *std::max_element(noisy_logits.begin(), noisy_logits.end());

                std::vector<float> exp_logits(n_experts);
                float sum_exp = 0.0f;
                for (long e = 0; e < n_experts; ++e) {
                    exp_logits[e] = std::exp(noisy_logits[e] - max_logit);
                    sum_exp += exp_logits[e];
                }

                std::vector<float> probs(n_experts);
                for (long e = 0; e < n_experts; ++e) {
                    probs[e] = exp_logits[e] / sum_exp;
                    gate_prob_sum[e] += probs[e];
                }
                if (std::is_same<MODE, training_mode_tag>::value) {
                    cached_gate_probs_[n] = probs;
                }

                // Select top-k experts by probability
                std::vector<std::pair<float, size_t>> expert_scores;
                expert_scores.reserve(n_experts);
                for (long e = 0; e < n_experts; ++e)
                    expert_scores.emplace_back(probs[e], e);

                std::partial_sort(expert_scores.begin(),
                    expert_scores.begin() + top_k,
                    expert_scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

                // Renormalize top-k weights to sum to 1
                float sum_weights = 0.0f;
                for (long i = 0; i < top_k; ++i)
                    sum_weights += expert_scores[i].first;

                // Handle degenerate case (should be extremely rare with softmax)
                if (sum_weights < 1e-8f) {
                    sum_weights = top_k;
                    for (long i = 0; i < top_k; ++i)
                        expert_scores[i].first = 1.0f;
                }

                for (long i = 0; i < top_k; ++i)
                    expert_scores[i].first /= sum_weights;

                // Cache selection for backward pass
                if (std::is_same<MODE, training_mode_tag>::value) {
                    selected_expert_indices_[n].resize(top_k);
                    selected_expert_weights_[n].resize(top_k);

                    for (long i = 0; i < top_k; ++i) {
                        selected_expert_indices_[n][i] = expert_scores[i].second;
                        selected_expert_weights_[n][i] = expert_scores[i].first;
                        routing_fraction[expert_scores[i].second] += 1.0f;
                    }
                }

                // Zero-copy views into input and output tensors
                const long sample_offset = n * sample_size;
                auto sample_input = sample_alias(expert_input, sample_offset);
                auto sample_output = sample_alias(output, sample_offset);

                // Route through selected experts and accumulate weighted outputs
                for (long i = 0; i < top_k; ++i) {
                    const size_t expert_idx = expert_scores[i].second;
                    const float weight = expert_scores[i].first;

                    experts[expert_idx].forward(sample_input);
                    const auto& expert_out = experts[expert_idx].get_output();

                    tt::add(1, sample_output, weight, expert_out);

                    batch_expert_usage[expert_idx] += weight;
                }
            }

            // Update exponential moving average of expert usage (for monitoring)
            if (std::is_same<MODE, training_mode_tag>::value) {
                for (long e = 0; e < n_experts; ++e) {
                    routing_fraction[e] /= num_samples;
                    gate_prob_sum[e] /= num_samples;
                }

                load_balance_loss_ = 0.0f;
                for (long e = 0; e < n_experts; ++e) {
                    load_balance_loss_ += routing_fraction[e] * gate_prob_sum[e];
                }
                load_balance_loss_ *= n_experts * load_balance_weight;

                cached_routing_fraction_ = routing_fraction;
                cached_gate_prob_avg_ = gate_prob_sum;

                if (usage_update_rate > 0) {
                    for (long e = 0; e < n_experts; ++e) {
                        float avg_usage = batch_expert_usage[e] / num_samples;
                        expert_usage[e] = (1.0f - usage_update_rate) * expert_usage[e] +
                            usage_update_rate * avg_usage;
                    }
                }
            }
        }

        /*!
            BACKWARD PASS
                Backpropagates gradients through cached expert selections.

                Process per sample:
                1. Retrieve cached expert indices and weights from forward pass
                2. For each selected expert:
                   a. Scale incoming gradient by expert's weight
                   b. Backpropagate through expert network
                   c. Accumulate expert's input gradient

                Note: Gradients automatically flow back to gate network through
                Dlib's computational graph without explicit implementation here.
        !*/
        template <typename SUBNET_TYPE>
        void backward(const tensor& gradient_input, SUBNET_TYPE& sub, tensor& params_grad)
        {
            tensor& expert_input_grad = sub.get_gradient_input();
            expert_input_grad = 0;

            const tensor& expert_input = sub.get_output();
            const long num_samples = cached_batch_size_;
            const long k = gradient_input.k();
            const long nr = gradient_input.nr();
            const long nc = gradient_input.nc();
            const long sample_size = k * nr * nc;

            DLIB_CASSERT(num_samples == (long)selected_expert_indices_.size(),
                "Forward pass cache missing or invalid in backward pass");

            alias_tensor sample_alias(1, k, nr, nc);

            for (long n = 0; n < num_samples; ++n) {
                const long sample_offset = n * sample_size;

                auto sample_grad = sample_alias(gradient_input, sample_offset);
                auto sample_input = sample_alias(expert_input, sample_offset);
                auto sample_input_grad = sample_alias(expert_input_grad, sample_offset);

                // Use cached expert routing from forward pass
                const auto& expert_indices = selected_expert_indices_[n];
                const auto& expert_weights = selected_expert_weights_[n];

                for (size_t i = 0; i < expert_indices.size(); ++i) {
                    const size_t expert_idx = expert_indices[i];
                    const float weight = expert_weights[i];

                    // Scale gradient by expert weight
                    resizable_tensor weighted_grad;
                    weighted_grad.copy_size(sample_grad);

                    const float* src_data = gradient_input.host() + sample_offset;
                    float* dst_data = weighted_grad.host();
                    std::transform(src_data, src_data + sample_size, dst_data,
                        [weight](float v) { return v * weight; });

                    // Backpropagate through expert
                    experts[expert_idx].back_propagate_error(sample_input, weighted_grad);
                    const auto& expert_grad = experts[expert_idx].get_gradient_input();

                    // Accumulate gradient
                    tt::add(1, sample_input_grad, 1, expert_grad);
                }
            }

            if (std::is_same<MODE, training_mode_tag>::value && load_balance_weight > 0
                && learning_rate_multiplier > 0) {
                tensor& gate_grad = layer<TAG>(sub).get_gradient_input();
                float* gate_grad_data = gate_grad.host();

                // Compute gradient of load balancing loss w.r.t. gate logits
                // Loss: L_aux = alpha * N * sum_e (f_e * P_e)
                // where f_e = routing fraction, P_e = gate probability average
                //
                // Gradient through softmax: dL/dz_j = P_j * (w_j - sum_e (w_e * P_e))
                // where w_e = df_e + dP_e = routing_fraction[e] + gate_prob_avg[e]

                for (long n = 0; n < num_samples; ++n) {
                    const auto& gate_probs = cached_gate_probs_[n];

                    // First pass: compute weighted sum for softmax normalization term
                    float sum_weighted_probs = 0.0f;
                    for (long e = 0; e < n_experts; ++e) {
                        float w_e = (cached_routing_fraction_[e] + cached_gate_prob_avg_[e]) *
                            n_experts * load_balance_weight / num_samples;
                        sum_weighted_probs += w_e * gate_probs[e];
                    }

                    // Second pass: apply complete softmax gradient formula
                    for (long e = 0; e < n_experts; ++e) {
                        float w_e = (cached_routing_fraction_[e] + cached_gate_prob_avg_[e]) *
                            n_experts * load_balance_weight / num_samples;

                        // Gradient component: P_j * (w_j - sum_e (w_e * P_e))
                        gate_grad_data[n * n_experts + e] += gate_probs[e] * (w_e - sum_weighted_probs);
                    }
                }
            }
        }

        void clean()
        {
            for (auto& expert : experts)
                clean_subnet(expert);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        void set_learning_rate_multiplier(double val)
        {
            learning_rate_multiplier = val;
            for (auto& expert : experts)
                set_all_learning_rate_multipliers(expert, val);
        }
        double get_learning_rate_multiplier() const { return learning_rate_multiplier; }

        // Direct access to expert networks (for inspection/debugging)
        EXPERT_NET& get_expert(size_t idx) {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        const EXPERT_NET& get_expert(size_t idx) const {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        // Accessors
        long num_experts() const { return n_experts; }
        long num_active_experts() const { return top_k; }
        bool is_training_mode() const { return std::is_same<MODE, training_mode_tag>::value; }
        const std::vector<float>& get_expert_usage() const { return expert_usage; }
        float get_load_balance_loss() const { return load_balance_loss_; }

        friend void serialize(const moe_& item, std::ostream& out)
        {
            serialize("moe_", out);
            serialize(item.n_experts, out);
            serialize(item.top_k, out);
            serialize(item.noise_scale, out);
            serialize(item.usage_update_rate, out);
            serialize(item.load_balance_weight, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.experts, out);
            serialize(item.expert_usage, out);
        }

        friend void deserialize(moe_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "moe_")
                throw serialization_error("Incorrect version '" + version + "' found while deserializing moe_.");

            deserialize(item.n_experts, in);
            deserialize(item.top_k, in);
            deserialize(item.noise_scale, in);
            deserialize(item.usage_update_rate, in);
            deserialize(item.load_balance_weight, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.experts, in);
            deserialize(item.expert_usage, in);

            item.cached_batch_size_ = 0;
        }

        friend std::ostream& operator<<(std::ostream& out, const moe_& item)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "moe\t ("
                << "experts=" << item.n_experts
                << ", top_k=" << item.top_k
                << ", mode=" << (is_training ? "train" : "infer")
                << ", noise=" << item.noise_scale
                << ", lb=" << item.load_balance_weight
                << ")";
            out << " learning_rate_mult=" << item.learning_rate_multiplier;
            return out;
        }

        friend void to_xml(const moe_& item, std::ostream& out)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "<moe"
                << " num_experts='" << item.n_experts << "'"
                << " top_k='" << item.top_k << "'"
                << " noise_scale='" << item.noise_scale << "'"
                << " usage_update_rate='" << item.usage_update_rate << "'"
                << " load_balance_weight='" << item.load_balance_weight << "'"
                << " learning_rate_mult='" << item.learning_rate_multiplier << "'"
                << " mode='" << (is_training ? "training" : "inference") << "'"
                << ">\n";
            for (size_t i = 0; i < item.experts.size(); ++i)
            {
                out << "<expert index='" << i << "'>\n";
                to_xml(item.experts[i], out);
                out << "</expert>\n";
            }
            out << "<expert_usage>";
            for (size_t i = 0; i < item.expert_usage.size(); ++i)
            {
                if (i > 0) out << " ";
                out << item.expert_usage[i];
            }
            out << "</expert_usage>\n";
            out << "</moe>\n";
        }

    private:
        template<typename NET>
        auto clean_subnet(NET& net) -> decltype(net.clean(), void())
        {
            net.clean();
        }

        template<typename NET>
        void clean_subnet(...)
        {
            // No-op if network doesn't have clean() method
        }

        // Configuration
        long n_experts;                     // Number of expert networks
        float noise_scale;                  // Gaussian noise std for exploration
        long top_k;                         // Number of experts to activate per sample
        float usage_update_rate;            // EMA smoothing rate for usage tracking
        float load_balance_weight;          // Auxiliary loss coefficient for expert load balancing
        double learning_rate_multiplier;

        // Expert networks
        std::vector<EXPERT_NET> experts;
        std::vector<float> expert_usage;     // Usage statistics (for monitoring)

        // Forward/backward cache (training mode only)
        std::vector<std::vector<size_t>> selected_expert_indices_;  // [sample][top_k]
        std::vector<std::vector<float>> selected_expert_weights_;   // [sample][top_k]
        std::vector<std::vector<float>> cached_gate_probs_;
        std::vector<float> cached_routing_fraction_;
        std::vector<float> cached_gate_prob_avg_;
        long cached_batch_size_;
        float load_balance_loss_;

		resizable_tensor params; // Unused
    };

    template<
        typename EXPERT_NET,
        long top_e,
        typename MODE,
        template<typename> class TAG,
        typename SUBNET
    >
    using moe = add_layer<moe_<EXPERT_NET, top_e, MODE, TAG, SUBNET>, SUBNET>;

    // This is a drop-in replacement for standard transformer feed-forward layers
    template<
        typename EXPERT_NET,
        long num_experts,
        long top_e,
        typename MODE,
        template <typename> class DO,
        typename SUBNET
    >
    using moe_ffn = add_prev8<moe<EXPERT_NET, top_e, MODE, tag9, rms_norm<skip8<
        tag9<gate<num_experts, DO, tag8<SUBNET>>>>>>>;
}

#endif // DLIB_DNN_TRANSFORMER_H_