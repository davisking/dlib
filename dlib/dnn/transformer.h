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
        using std_ffn = DO<linear<d_model, ACT<linear<d_model * 4, SUBNET>>>>;

        // Standard SwiGLU FFN implementation
        // Reference: Noam Shazeer's "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = DO<linear<d_model, multm_prev6<linear<(d_model * 8) / 3, skip5<
            tag6<ACT<linear<(d_model * 8) / 3, tag5<SUBNET>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block = act<
            add_prev4<std_ffn<ACT, DO, d_model, rms_norm<tag4<
            multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>>>>>;

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
            DO<extract<0, 1, 1, d_model, fc_no_bias<d_model,
            multm_prev3<softmaxm<tril_mask<
            scale_weights<d_model / num_heads,
            multm_prev4<
            query<num_heads, d_model, skip2<
            tag4<key<num_heads, d_model, skip2<
            tag3<value<num_heads, d_model,
            tag2<fc_no_bias<d_model * 3,
            rms_norm<tag1<SUBNET>>>>>>>>>>>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using std_ffn = extract<0, 1, 1, d_model,
            DO<fc<d_model, ACT<fc<d_model * 4, SUBNET>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long d_model, typename SUBNET>
        using swiglu = DO<extract<0, 1, 1, d_model,
            fc<d_model, mult_prev7<fc<(d_model * 8) / 3, skip6<
            tag7<silu<fc<(d_model * 8) / 3, tag6<SUBNET>>>>>>>>>>;

        template <template <typename> class ACT, template <typename> class DO,
            long seq_len, long d_model, long num_heads, typename SUBNET>
        using transformer_block =
            add_prev5<std_ffn<ACT, DO, d_model, rms_norm<tag5<
            multihead_attention<ACT, DO, d_model, num_heads, SUBNET>>>>>;

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
            hidden_dim(0)
        {
        }

        hrm_(const hrm_& other) :
            h_net(other.h_net),
            l_net(other.l_net),
            z_h_init(other.z_h_init),
            z_l_init(other.z_l_init),
            seq_len(other.seq_len),
            hidden_dim(other.hidden_dim)
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

            // Initialize internal networks with proper tensor shape
            initialize_networks(input);

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
        }

        friend std::ostream& operator<<(std::ostream& out, const hrm_& item)
        {
            out << "hrm (N=" << N << ", T=" << T << ")";
            return out;
        }

        friend void to_xml(const hrm_& item, std::ostream& out)
        {
            out << "<hrm N='" << N << "' T='" << T << "'>\n";
            out << "  <h_module>\n";
            to_xml(item.h_net, out);
            out << "  </h_module>\n";
            out << "  <l_module>\n";
            to_xml(item.l_net, out);
            out << "  </l_module>\n";
            out << "</hrm>\n";
        }

    private:
        void initialize_networks(const tensor& input_shape)
        {
            // Create dummy input data for network initialization
            const long nr = input_shape.nr();
            const long nc = input_shape.nc();

            matrix<float> dummy_data(nr, nc);
            dlib::rand rng(std::time(0));
            for (long r = 0; r < nr; ++r)
            {
                for (long c = 0; c < nc; ++c)
                {
                    dummy_data(r, c) = static_cast<float>(rng.get_random_gaussian() * 0.1);
                }
            }
            resizable_tensor init_tensor(1, 1, nr, nc);
            std::vector<matrix<float>> x(1, dummy_data);

            // Initialize both networks using to_tensor
            h_net.to_tensor(&x[0], &x[0] + 1, init_tensor);
            l_net.to_tensor(&x[0], &x[0] + 1, init_tensor);
        }

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

        // Dimensions
        long seq_len;
        long hidden_dim;

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

    template <long num_experts, template <typename> class DO, typename SUBNET>
    using gate = softmax<fc<num_experts, avg_pool_everything<
        DO<leaky_relu<fc<16,
        DO<leaky_relu<fc<32,
        DO<fc<16, SUBNET>>>>>>>>>>>;

    struct training_mode_tag {};
    struct inference_mode_tag {};

    template<
        typename EXPERT_NET,                    // Expert network architecture
        long top_k,                             // Number of experts to activate (0 = auto)
        typename MODE,                          // Tag-based mode selection
        template<typename> class TAG,           // Tag for gate input
        typename SUBNET                         // Input subnet type
    >
    class moe_
    {
    public:
        /*!
            Initializes hyperparameters with sensible defaults:
            - balance_loss_weight: controls strength of load balancing
            - noise_scale: exploration noise during training
            - usage_update_rate: exponential moving average rate for usage stats
        !*/
        explicit moe_() :
            n_experts(0),
            balance_loss_weight(0.01f),
            noise_scale(0.2f),
            top_n(top_k),
            usage_update_rate(0.05f)
        {
        }

        moe_(const moe_& other) :
            n_experts(other.n_experts),
            balance_loss_weight(other.balance_loss_weight),
            noise_scale(other.noise_scale),
            top_n(other.top_n),
            usage_update_rate(other.usage_update_rate),
            expert_usage(other.expert_usage),
            indices(other.indices)
        {
            // Deep copy of expert networks
            experts.reserve(other.experts.size());
            for (const auto& expert : other.experts) {
                experts.push_back(expert);
            }
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                n_experts = other.n_experts;
                balance_loss_weight = other.balance_loss_weight;
                noise_scale = other.noise_scale;
                top_n = other.top_n;
                usage_update_rate = other.usage_update_rate;
                expert_usage = other.expert_usage;
                indices = other.indices;

                // Deep copy of expert networks
                experts.clear();
                experts.reserve(other.experts.size());
                for (const auto& expert : other.experts) {
                    experts.push_back(expert);
                }
            }
            return *this;
        }

        /*!
            Sets up the expert networks based on gate output dimensions.
            The number of experts is determined by gate_input.k().
        !*/
        template <typename SUBNET_TYPE>
        void setup(const SUBNET_TYPE& sub) {
            // Get gate output to determine number of experts
            const tensor& gate_input = layer<TAG>(sub).get_output();
            long new_n_experts = gate_input.k();

            // Initialize experts if needed
            if (new_n_experts != n_experts) {
                n_experts = new_n_experts;
                expert_weights.resize(n_experts, 0.0f);
                expert_usage.resize(n_experts, 0.0f);
                indices.resize(n_experts);

                // Create expert network instances
                experts.clear();
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i)
                    experts.emplace_back(EXPERT_NET{});

                // Set top-k if auto mode (top_k == 0)
                if (top_k == 0) {
                    // Use 20% of experts by default
                    top_n = std::max(1L, static_cast<long>(std::floor(n_experts * 0.2f)));
                }
                else {
                    top_n = std::min(top_k, n_experts);
                }

                // Initialize expert networks with dummy input
                initialize_experts(sub.get_output());
            }
        }

        /*!
            Process:
            1. Read gate probabilities for expert selection
            2. Compute aggregated expert weights across batch
            3. Add exploration noise during training
            4. Select top-k experts based on weights
            5. Route inputs through selected experts
            6. Combine expert outputs using normalized weights
        !*/
        template <typename SUBNET_TYPE>
        void forward(const SUBNET_TYPE& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const tensor& gate_input = layer<TAG>(sub).get_output();

            // Validate gate output dimensions
            DLIB_CASSERT(gate_input.k() == n_experts &&
                gate_input.nr() == 1 && gate_input.nc() == 1,
                "\nExpected gate output shape [batch_size, " << n_experts << ", 1, 1]"
                << "\nReceived shape [" << gate_input.num_samples() << ", "
                << gate_input.k() << ", " << gate_input.nr() << ", "
                << gate_input.nc() << "]");

            const long num_samples = gate_input.num_samples();
            const float* gate_probs = gate_input.host();

            // Initialize output tensor
            output.copy_size(expert_input);
            output = 0;

            // Aggregate gate probabilities across batch to compute expert importance
            std::fill(expert_weights.begin(), expert_weights.end(), 0.0f);
            for (long n = 0; n < num_samples; ++n) {
                for (long e = 0; e < n_experts; ++e) {
                    expert_weights[e] += gate_probs[n * n_experts + e];
                }
            }

            // Normalize weights and add exploration noise during training
            if (std::is_same<MODE, training_mode_tag>::value) {
                static dlib::rand rnd(std::time(0));
                for (auto& w : expert_weights) {
                    // Average over batch + exploration noise
                    w = w / num_samples + noise_scale * rnd.get_random_float();
                }
            }
            else {
                // Pure averaging during inference
                for (auto& w : expert_weights) {
                    w /= num_samples;
                }
            }

            // Select top-k experts based on aggregated weights
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_n, indices.end(),
                [&](size_t a, size_t b) {
                    return expert_weights[a] > expert_weights[b];
                });

            // Normalize weights of selected experts
            float sum_top_weights = 0.0f;
            for (size_t i = 0; i < top_n; ++i)
                sum_top_weights += expert_weights[indices[i]];

            // Forward through selected experts and combine outputs
            for (size_t i = 0; i < top_n; ++i) {
                const size_t eidx = indices[i];
                expert_weights[eidx] /= sum_top_weights;

                // Process input through expert
                experts[eidx].forward(expert_input);
                auto& expert_out = experts[eidx].get_output();

                // Add weighted expert output to final result
                tt::add(1, output, expert_weights[eidx], expert_out);

                // Track expert usage for load balancing
                expert_usage[eidx] += expert_weights[eidx];
            }
        }

        /*!
            Process:
            1. Compute auxiliary load balancing loss
            2. Backpropagate through activated experts only
            3. Scale gradients by expert weights
            4. Update expert usage statistics
        !*/
        template <typename SUBNET_TYPE>
        void backward(const tensor& gradient_input, SUBNET_TYPE& sub, tensor& params_grad)
        {
            tensor& expert_input_grad = sub.get_gradient_input();

            // Compute auxiliary loss for load balancing
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            float aux_loss = compute_auxiliary_loss();

            // Backpropagate through each activated expert
            for (size_t i = 0; i < top_n; ++i) {
                const size_t eidx = indices[i];

                // Prepare gradient for this expert
                resizable_tensor adjusted_gradient = gradient_input;

                // Add auxiliary loss gradient if needed
                if (aux_loss > 0)
                    tt::add(1, adjusted_gradient, aux_loss, experts[eidx].get_output());

                // Backpropagate through expert network
                experts[eidx].back_propagate_error(sub.get_output(), adjusted_gradient);
                auto& expert_grad = experts[eidx].get_gradient_input();

                // Accumulate weighted gradients to input
                tt::add(1, expert_input_grad, expert_weights[eidx], expert_grad);
            }

            // Update exponential moving average of expert usage
            if (is_training && usage_update_rate > 0 && usage_update_rate <= 1.0f) {
                for (size_t i = 0; i < top_n; ++i) {
                    const size_t eidx = indices[i];
                    expert_usage[eidx] = (1.0f - usage_update_rate) * expert_usage[eidx] +
                        usage_update_rate * expert_weights[eidx];
                }
            }
        }

        // Cleans up the internal expert networks
        void clean()
        {
            for (auto& expert : experts)
                clean_subnet(expert);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        // Get reference to expert network
        EXPERT_NET& get_expert(size_t idx) {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }        
        const EXPERT_NET& get_expert(size_t idx) const {
            DLIB_CASSERT(idx < experts.size(), "Expert index out of bounds");
            return experts[idx];
        }

        long num_experts() const { return n_experts; }
        bool is_training_mode() const { return std::is_same<MODE, training_mode_tag>::value; }
        
        friend void serialize(const moe_& item, std::ostream& out)
        {
            serialize("moe_", out);
            serialize(item.n_experts, out);
            serialize(item.top_n, out);
            serialize(item.balance_loss_weight, out);
            serialize(item.noise_scale, out);
            serialize(item.usage_update_rate, out);
            serialize(item.experts, out);
            serialize(item.expert_usage, out);
        }

        friend void deserialize(moe_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "moe_")
                throw serialization_error("Incorrect version found while deserializing moe_.");

            deserialize(item.n_experts, in);
            deserialize(item.top_n, in);
            deserialize(item.balance_loss_weight, in);
            deserialize(item.noise_scale, in);
            deserialize(item.usage_update_rate, in);

            item.expert_weights.resize(item.n_experts, 0.0f);
            item.indices.resize(item.n_experts);

            deserialize(item.experts, in);
            deserialize(item.expert_usage, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const moe_& item)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "moe"
                << " (experts=" << item.n_experts
                << ", top_k=" << item.top_n
                << ", mode=" << (is_training ? "train" : "infer") << ")";
            return out;
        }

        friend void to_xml(const moe_& item, std::ostream& out)
        {
            const bool is_training = std::is_same<MODE, training_mode_tag>::value;
            out << "<moe>\n";
            out << "  <num_experts>" << item.n_experts << "</num_experts>\n";
            out << "  <top_k>" << item.top_n << "</top_k>\n";
            out << "  <training>" << is_training << "</training>\n";
            out << "</moe>\n";
        }

    private:
        /*!
            Ensures all expert networks are properly set up with correct dimensions
            before actual training begins.
        !*/
        void initialize_experts(const tensor& expert_input) {
            const long nr = expert_input.nr();
            const long nc = expert_input.nc();

            // Create dummy input matching dimensions
            matrix<float> input_data(nr, nc);
            input_data = 0.0f;

            resizable_tensor input_tensor(1, 1, nr, nc);
            std::vector<matrix<float>> x(1, input_data);

            // Forward through each expert to initialize
            for (size_t i = 0; i < experts.size(); ++i)
                experts[i].to_tensor(&x[0], &x[0] + 1, input_tensor);
        }

        /*!
            Encourages balanced expert utilization by penalizing high variance
            in expert usage. Returns normalized standard deviation of usage.
        !*/
        float compute_auxiliary_loss() const {
            if (n_experts < 2) return 0.0f;

            // Compute mean usage
            float mean_usage = std::accumulate(expert_usage.begin(),
                expert_usage.end(), 0.0f) / n_experts;
            if (mean_usage < 1e-8f) return 0.0f;

            // Compute variance
            float var = 0.0f;
            for (float usage : expert_usage) {
                float diff = usage - mean_usage;
                var += diff * diff;
            }
            var /= n_experts;

            // Normalized standard deviation as load imbalance measure
            float stddev = std::sqrt(var);
            float normalized_stddev = stddev / (mean_usage + 1e-6f);

            return balance_loss_weight * normalized_stddev;
        }

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

        // Model parameters
        long n_experts;                          // Number of experts (detected from gate)
        long top_n;                              // Number of experts to activate
        float balance_loss_weight;               // Weight for auxiliary balancing loss
        float noise_scale;                       // Exploration noise magnitude (training)
        float usage_update_rate;                 // EMA rate for usage statistics

        // Expert networks and state
        std::vector<EXPERT_NET> experts;         // Expert network instances
        std::vector<float> expert_weights;       // Current expert selection weights
        std::vector<float> expert_usage;         // Exponential moving average of usage
        std::vector<size_t> indices;             // Helper for top-k selection
        resizable_tensor params;                 // Layer parameters (if any)
    };

    template<
        typename EXPERT_NET,
        long top_k,
        typename MODE,
        template<typename> class TAG,
        typename SUBNET
    >
    using moe = add_layer<moe_<EXPERT_NET, top_k, MODE, TAG, SUBNET>, SUBNET>;

    // This is a drop-in replacement for standard transformer feed-forward layers
    template<
        typename EXPERT_NET,
        long num_experts,
        long top_k,
        typename MODE,
        template <typename> class DO,
        typename SUBNET
    >
    using moe_ffn = add_prev8<moe<EXPERT_NET, top_k, MODE, tag9, skip8<
        tag9<gate<num_experts, DO, rms_norm<tag8<SUBNET>>>>>>>;
}

#endif // DLIB_DNN_TRANSFORMER_H_