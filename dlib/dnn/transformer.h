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
}

#endif // DLIB_DNN_TRANSFORMER_H_