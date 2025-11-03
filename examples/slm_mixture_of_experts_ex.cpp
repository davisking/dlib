/*!
    @file slm_advanced_train_ex.cpp
    @brief Transformer-based text training/generation

    This program implements a complete training and generation pipeline for a
    Transformer-based text compression system.
    The model features:

    1. Rotary Positional Embeddings (RoPE) for enhanced positional encoding
    2. Multi-head self-attention with efficient memory handling
    3. Mixture-of-Experts architecture for specialized processing
    4. BPE tokenization with custom vocabulary
    5. Full training/generation/verification workflow

    Key capabilities demonstrated:
    - Perfect memorization and reproduction of training text
    - Efficient autoregressive generation
    - Byte-level verification of reconstructed text

    References:
    [1] Vaswani et al., "Attention Is All You Need" (Transformer architecture)
        arXiv:1706.03762
    [2] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        arXiv:2104.09864
    [3] Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
        Mixture-of-Experts Layer" (MoE architecture) arXiv:1701.06538

    Usage modes:
    --train         Train model on dataset
    --generate      Generate text from trained model
    --verify        Compare generated output with original
    --tokenize-only Only perform tokenization step

    Configuration:
    - Adjust template parameters in transformer_config for model architecture
    - Modify training parameters in main() for optimization
    - Set sequence length and memory limits according to available hardware
!*/
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <csignal>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/tokenizer/bpe_tokenizer.h>
#include <dlib/serialize.h>

using namespace std;
using namespace dlib;

namespace dlib
{
    template <long num_experts, template <typename> class DO, typename SUBNET>
    using gate = softmax<fc<num_experts, avg_pool_everything<
        DO<leaky_relu<fc<32,
        DO<leaky_relu<fc<64,
        DO<fc<32, SUBNET>>>>>>>>>>>;

    namespace layer_test {
        template <template <typename> class DO>
        struct is_multiply { static constexpr bool value = false; };

        template <>
        struct is_multiply<multiply> { static constexpr bool value = true; };
    }

    /*!
        @class moe_
        @brief Implements a Mixture-of-Experts (MoE) layer with dynamic routing

        This layer implements a sparse mixture-of-experts architecture where:
        - Inputs are dynamically routed to top-k experts
        - Experts are simple feed-forward networks
        - Routing is learned through a gating network
        - Auxiliary loss encourages balanced expert utilization

        The implementation supports:
        - Training/inference modes with different behaviors
        - Configurable number of experts and top-k selection
        - Noise injection for exploration during training
        - Usage tracking and balancing mechanism

        Template parameters:
            @param d_model      Model dimension size
            @param ACT          Activation function type for experts
            @param DO           Dropout policy for experts
            @param TAG          Tag type for gating network input
    !*/
    template <long d_model, template<typename> class ACT,
        template<typename> class DO, template<typename> class TAG>
    class moe_
    {
    public:
        //using expert_net_type = DO<linear<d_model, ACT<linear<d_model * 4, input<matrix<float>>>>>>;
        using expert_net_type = linear<d_model, ACT<linear<d_model * 4, input<matrix<float>>>>>;

        explicit moe_() :
            n_experts(0),
            balance_loss_weight(0.01f),
            noise_scale(0.2f),
            training_phase(!layer_test::is_multiply<DO>::value),
            top_n(1),
            usage_update_rate(0.05f)
        {
        }

        moe_(const moe_& other) :
            n_experts(other.n_experts),
            balance_loss_weight(other.balance_loss_weight),
            noise_scale(other.noise_scale),
            training_phase(other.training_phase),
            top_n(other.top_n),
            usage_update_rate(other.usage_update_rate),
            experts(other.experts),
            expert_weights(other.expert_weights),
            expert_usage(other.expert_usage),
            indices(other.indices)
        {
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            const tensor& gate_input = layer<TAG>(sub).get_output();
            long new_n_experts = gate_input.k();
            if (new_n_experts != n_experts) {
                n_experts = new_n_experts;
                expert_weights.resize(n_experts, 0.0f);
                expert_usage.resize(n_experts, 0.0f);
                indices.resize(n_experts);

                experts.clear();
                experts.reserve(n_experts);
                for (long i = 0; i < n_experts; ++i)
                    experts.emplace_back(expert_net_type{});
                top_n = std::max(1L, static_cast<long>(std::floor(n_experts * 0.2f)));

                initialize_experts(sub.get_output());
            }
        }

        moe_& operator=(const moe_& other)
        {
            if (this != &other) {
                n_experts = other.n_experts;
                balance_loss_weight = other.balance_loss_weight;
                noise_scale = other.noise_scale;
                training_phase = other.training_phase;
                top_n = other.top_n;
                usage_update_rate = other.usage_update_rate;
                experts = other.experts;
                expert_weights = other.expert_weights;
                expert_usage = other.expert_usage;
                indices = other.indices;
            }
            return *this;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& expert_input = sub.get_output();
            const tensor& gate_input = layer<TAG>(sub).get_output();

            DLIB_CASSERT(gate_input.k() == n_experts &&
                gate_input.nr() == 1 && gate_input.nc() == 1,
                "\nExpected gate output shape [batch_size, " << n_experts << ", 1, 1]"
                << "\nReceived shape [" << gate_input.num_samples() << ", "
                << gate_input.k() << ", " << gate_input.nr() << ", "
                << gate_input.nc() << "]");

            const long num_samples = gate_input.num_samples();
            const float* gate_probs = gate_input.host();
            output.copy_size(expert_input);
            output = 0;

            std::fill(expert_weights.begin(), expert_weights.end(), 0.0f);
            for (long n = 0; n < num_samples; ++n) {
                for (long e = 0; e < n_experts; ++e) {
                    expert_weights[e] += gate_probs[n * n_experts + e];
                }
            }
            if (training_phase) {
                static dlib::rand rnd(std::time(0));
                for (auto& w : expert_weights)
                    w = w / num_samples + noise_scale * rnd.get_random_float();
            }
            else {
                for (auto& w : expert_weights) w /= num_samples;
            }

            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + top_n, indices.end(),
                [&](size_t a, size_t b) {
                return expert_weights[a] > expert_weights[b];
            });

            float sum_top_weights = 0.0f;
            for (size_t i = 0; i < top_n; ++i)
                sum_top_weights += expert_weights[indices[i]];

            for (size_t i = 0; i < top_n; ++i) {
                const size_t eidx = indices[i];
                expert_weights[eidx] /= sum_top_weights;

                experts[eidx].forward(expert_input);
                auto& expert_out = experts[eidx].get_output();

                tt::add(1, output, expert_weights[eidx], expert_out);
                expert_usage[eidx] += expert_weights[eidx];
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            tensor& expert_input_grad = sub.get_gradient_input();

            float aux_loss = compute_auxiliary_loss();

            for (size_t i = 0; i < top_n; ++i) {
                const size_t eidx = indices[i];
                //cout << "backward - selected expert: " << eidx << endl;
                resizable_tensor adjusted_gradient = gradient_input;
                if (aux_loss > 0)
                    tt::add(1, adjusted_gradient, aux_loss, experts[eidx].get_output());

                experts[eidx].back_propagate_error(sub.get_output(), adjusted_gradient);
                auto& expert_grad = experts[eidx].get_gradient_input();

                tt::add(1, expert_input_grad, expert_weights[eidx], expert_grad);
                experts[eidx].clean();
            }

            if (usage_update_rate > 0 && usage_update_rate <= 1.0f) {
                for (size_t i = 0; i < top_n; ++i) {
                    const size_t eidx = indices[i];
                    expert_usage[eidx] = (1.0f - usage_update_rate) * expert_usage[eidx] +
                        usage_update_rate * expert_weights[eidx];
                }
            }
        }

        void set_training_phase(bool t) { training_phase = t; }
        bool is_training_phase() const { return training_phase; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const moe_& item, std::ostream& out)
        {
            serialize("moe_", out);
            serialize(item.n_experts, out);
            serialize(item.top_n, out);
            serialize(item.balance_loss_weight, out);
            serialize(item.noise_scale, out);
            serialize(item.usage_update_rate, out);
            serialize(item.experts, out);
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
            item.expert_usage.resize(item.n_experts, 0.0f);
            item.indices.resize(item.n_experts);
            item.experts.reserve(item.n_experts);

            deserialize(item.experts, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const moe_& item)
        {
            out << "moe (num_experts=" << item.n_experts << ")";
            return out;
        }

        friend void to_xml(const moe_& item, std::ostream& out)
        {
            out << "<moe>\n";
            out << "<num_experts>" << item.n_experts << "</num_experts>\n";
            out << "</moe>\n";
        }

    private:
        void initialize_experts(const tensor& expert_input) {
            const long nr = expert_input.nr(), nc = expert_input.nc();
            matrix<float> input_data(nr, nc);
            input_data = 0.0f;

            resizable_tensor input_tensor(1, 1, nr, nc);
            std::vector<matrix<float>> x(1, input_data);

            for (size_t i = 0; i < experts.size(); ++i)
                experts[i].to_tensor(&x[0], &x[0] + 1, input_tensor);
        }

        float compute_auxiliary_loss() const {
            if (n_experts < 2) return 0.0f;
            float mean_usage = std::accumulate(expert_usage.begin(), expert_usage.end(), 0.0f) / n_experts;
            if (mean_usage < 1e-8f) return 0.0f;

            float var = 0.0f;
            for (float usage : expert_usage) {
                float diff = usage - mean_usage;
                var += diff * diff;
            }
            var /= n_experts;
            float stddev = std::sqrt(var);

            float normalized_stddev = stddev / (mean_usage + 1e-6f);
            return balance_loss_weight * normalized_stddev;
        }

        long n_experts;
        size_t top_n;
        float balance_loss_weight, noise_scale, usage_update_rate;
        bool training_phase;

        std::vector<expert_net_type> experts;
        std::vector<float> expert_weights, expert_usage;
        std::vector<size_t> indices;
        resizable_tensor params;
    };

    template <long d_model, template<typename> class ACT,
        template<typename> class DO, template<typename> class TAG, typename SUBNET>
    using moe = add_layer<moe_<d_model, ACT, DO, TAG>, SUBNET>;

    // Complete MoE feed-forward layer
    template <long d_model, template <typename> class ACT, template <typename> class DO,
        long num_experts, typename SUBNET>
    using moe_feed_forward = rms_norm<add_prev5<
        moe<d_model, ACT, DO, tag6, skip5<
        tag6<gate<num_experts, DO, tag5<SUBNET>>>>>>>;


    template <template <typename> class DO, long num_experts, typename SUBNET>
    using moe_router = softmax<fc<num_experts, avg_pool_everything<
        DO<leaky_relu<fc<16, DO<leaky_relu<fc<32,
        DO<fc<16, SUBNET>>>>>>>>>>>;

    // Single expert network
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using expert = DO<linear<d_model, DO<ACT<linear<d_model * 4, SUBNET>>>>>;

    // Combines expert outputs using router probabilities
    // Performs weighted sum of experts with residual connection
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using weighted_sum_of_experts = add_prev<itag3,
        mult_prev<itag1, extract<0, 1, 1, 1, skip6<         // Expert 1
        itag1<expert<ACT, DO, d_model, iskip<
        itag3<mult_prev<itag2, extract<1, 1, 1, 1, skip6<   // Expert 2
        itag2<expert<ACT, DO, d_model,
        itag0<SUBNET>>>>>>>>>>>>>>;

    // Complete MoE feed-forward layer
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, typename SUBNET>
    using moe_feed_forward =
        rms_norm<add_prev5<
        weighted_sum_of_experts<ACT, DO, d_model, skip5<
        tag6<moe_router<DO, 2,
        tag5<SUBNET>>>>>>>;

    /*!
        This defines a standard transformer encoder block with self-attention
        followed by a feed-forward network, each with residual connections.

        Template parameters:
            - ACT: activation function type
            - DO: dropout layer type for regularization
            - seq_len: sequence length (number of tokens/patches)
            - d_model: model dimension
            - num_heads: number of attention heads
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename SUBNET>
    using trans_moe_block =
        moe_feed_forward<ACT, DO, d_model,
        canonical_transformer::multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;

    // Classification Head   
    template <long num_logits, typename SUBNET>
    using classification_head = loss_multiclass_log<fc<num_logits, SUBNET>>;

    /**
     * @brief Transformer Model Configuration Template
     *
     * Provides a flexible and type-safe configuration mechanism for Transformer models
     * with compile-time parameter validation and network generation.
     *
     * Template parameters:
     * @param vocab_size Vocabulary size for token embedding
     * @param num_layers Number of Transformer layers
     * @param num_heads Number of attention heads
     * @param embedding_dim Dimension of token embeddings
     * @param max_seq_len Maximum sequence length
     * @param activation_func Activation function type
     * @param dropout_policy Dropout regularization policy
     */
    template <
        long vocab_size = 15000,                                // Default vocabulary size
        long num_layers = 6,                                    // Default number of layers
        long num_heads = 8,                                     // Default number of attention heads
        long embedding_dim = 512,                               // Default embedding dimension
        long max_seq_len = 300,                                 // Default maximum sequence length
        template <typename> class activation_func = gelu,       // Default activation function
        template <typename> class dropout_policy = dropout_10   // Default dropout policy
    >
    struct transformer_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long MAX_SEQ_LEN = max_seq_len;

        /**
         * @brief Compile-time validation of model configuration
         *
         * Performs static assertions to ensure valid model parameters
         */
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        // Network component definitions
        template <typename SUBNET>
        using t_transformer_block =
            trans_moe_block<activation_func, dropout_policy, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS, SUBNET>;

        template <typename SUBNET>
        using i_transformer_block =
            trans_moe_block<activation_func, multiply, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS, SUBNET>;

        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            projection_head<activation_func, 2, EMBEDDING_DIM,
            repeat<NUM_LAYERS, t_transformer_block,
            token_embeddings<dropout_policy, VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>,
            classification_head<VOCAB_SIZE,
            projection_head<activation_func, 2, EMBEDDING_DIM,
            repeat<NUM_LAYERS, i_transformer_block,
            token_embeddings<multiply, VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Transformer model configuration:\n"
                    << "- vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- layers: " << NUM_LAYERS << "\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- sequence length: " << MAX_SEQ_LEN;
                return ss.str();
            }
        };
    };
}

// Define a cross-platform signal handling system
namespace {
    std::atomic<bool> g_terminate_flag(false);

#ifdef _WIN32
    // Windows-specific handler
    BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
        if (ctrl_type == CTRL_C_EVENT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
            return TRUE;
        }
        return FALSE;
    }
#else
    // Unix/Linux/macOS handler
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
        }
    }
#endif

    // Setup the interrupt handler based on platform
    void setup_interrupt_handler() {
#ifdef _WIN32
        if (!SetConsoleCtrlHandler(console_ctrl_handler, TRUE)) {
            cerr << "ERROR: Could not set control handler" << endl;
        }
#else
        struct sigaction sa {};
        sigemptyset(&sa.sa_mask);
        sa.sa_handler = signal_handler;
        sigaction(SIGINT, &sa, NULL);
#endif
    }
}

// Utility function to get file size
size_t get_file_size(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return 0;
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.close();
    return file_size;
}

// Function to generate tokens filename based on input file and size
std::string generate_tokens_filename(const std::string& input_file, size_t max_bytes) {
    // Extract base name from input file
    std::string base_name = input_file;
    size_t pos = base_name.find_last_of("/\\");
    if (pos != std::string::npos) base_name = base_name.substr(pos + 1);

    // Create filename with size information
    std::string size_info = (max_bytes > 0) ? "partial" : "full";
    return base_name + "." + size_info + ".tokens.bin";
}

// Function to save tokens to binary file
bool save_tokens_to_file(const std::vector<int>& tokens, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // Write number of tokens
    uint64_t num_tokens = tokens.size();
    file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));

    // Write tokens
    for (int token : tokens) {
        uint32_t t = static_cast<uint32_t>(token);
        file.write(reinterpret_cast<const char*>(&t), sizeof(t));
    }
    file.flush();
    file.close();

    return true;
}

// Function to load tokens from binary file
bool load_tokens_from_file(std::vector<int>& tokens, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // Read number of tokens
    uint64_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));

    // Read tokens
    tokens.resize(num_tokens);
    for (uint64_t i = 0; i < num_tokens; ++i) {
        uint32_t t;
        file.read(reinterpret_cast<char*>(&t), sizeof(t));
        tokens[i] = static_cast<int>(t);
    }
    file.close();

    return true;
}

// Function to read the data file (entire or portion)
std::string read_data(const std::string& filepath, size_t max_bytes = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open data file: " + filepath);
    }
    size_t file_size = get_file_size(filepath);

    // If max_bytes is specified and valid, limit the reading
    size_t bytes_to_read = (max_bytes > 0 && max_bytes < file_size) ? max_bytes : file_size;

    std::string content(bytes_to_read, ' ');
    file.read(&content[0], bytes_to_read);

    return content;
}

// Function to verify byte-for-byte matching with detailed error reporting
bool verify_match(const std::string& original, const std::string& generated) {
    if (original.size() != generated.size()) {
        cout << "Size mismatch: original=" << original.size()
            << " bytes, generated=" << generated.size() << " bytes\n";
        return false;
    }

    // Helper function to determine if a character is printable
    auto is_printable = [](unsigned char c) { return c >= 32 && c < 127; };

    // Helper function to format a byte as string (either character or hex)
    auto format_byte = [&is_printable](unsigned char c) -> std::string {
        if (is_printable(c)) {
            return std::string(1, c);
        }
        else {
            std::stringstream ss;
            ss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
            return ss.str();
        }
        };

    // Helper function to display context around a position
    auto show_context = [&](size_t pos, size_t context_size) {
        size_t start = (pos >= context_size) ? pos - context_size : 0;
        size_t end = std::min(original.size(), pos + context_size + 1);

        std::string orig_context, gen_context;
        std::string orig_highlight, gen_highlight;

        for (size_t i = start; i < end; ++i) {
            unsigned char orig_c = static_cast<unsigned char>(original[i]);
            unsigned char gen_c = static_cast<unsigned char>(generated[i]);

            orig_context += format_byte(orig_c);
            gen_context += format_byte(gen_c);

            if (i == pos) {
                orig_highlight = format_byte(orig_c);
                gen_highlight = format_byte(gen_c);
            }
        }

        cout << "Context at position " << pos << ":\n";
        cout << "Original (" << (int)original[pos] << " = '" << orig_highlight
            << "'): " << orig_context << "\n";
        cout << "Generated (" << (int)generated[pos] << " = '" << gen_highlight
            << "'): " << gen_context << "\n";
        };

    size_t mismatch_count = 0;
    const size_t max_detailed_mismatches = 10;  // Maximum number of detailed errors to display
    const size_t context_size = 10;             // Number of characters to show before/after error

    // Track error patterns
    std::map<std::pair<char, char>, int> error_patterns;

    // Analyze consecutive error regions
    size_t current_region_start = 0;
    size_t current_region_length = 0;
    std::vector<std::pair<size_t, size_t>> error_regions; // (start, length)

    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != generated[i]) {
            // Track error pattern
            error_patterns[{original[i], generated[i]}]++;

            // Increment mismatch count
            mismatch_count++;

            // Handle error regions
            if (current_region_length == 0) {
                current_region_start = i;
                current_region_length = 1;
            }
            else if (i == current_region_start + current_region_length) {
                current_region_length++;
            }
            else {
                // Save previous region and start new one
                error_regions.push_back({ current_region_start, current_region_length });
                current_region_start = i;
                current_region_length = 1;
            }

            // Show detailed information for first few mismatches
            if (mismatch_count <= max_detailed_mismatches) {
                cout << "\n----- Mismatch #" << mismatch_count << " -----\n";
                show_context(i, context_size);
            }
        }
    }

    // Add the last region if exists
    if (current_region_length > 0) {
        error_regions.push_back({ current_region_start, current_region_length });
    }

    if (mismatch_count > 0) {
        cout << "\n===== Error summary =====\n";
        cout << "Total mismatches: " << mismatch_count << " bytes ("
            << (mismatch_count * 100.0 / original.size()) << "%)\n";

        // Report on error regions
        cout << "\nFound " << error_regions.size() << " error regions:\n";
        for (size_t i = 0; i < error_regions.size() && i < 20; ++i) {
            cout << "  Region #" << (i + 1) << ": Position " << error_regions[i].first
                << ", Length " << error_regions[i].second << "\n";
        }
        if (error_regions.size() > 20)
            cout << "  ... and " << (error_regions.size() - 20) << " more regions\n";

        // Report on most common error patterns
        cout << "\nMost common error patterns (original -> generated):\n";
        std::vector<std::pair<std::pair<char, char>, int>> patterns(
            error_patterns.begin(), error_patterns.end());
        std::sort(patterns.begin(), patterns.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < patterns.size() && i < 10; ++i) {
            char orig = patterns[i].first.first;
            char gen = patterns[i].first.second;
            int count = patterns[i].second;

            cout << "  '" << format_byte(static_cast<unsigned char>(orig)) << "' ("
                << static_cast<int>(static_cast<unsigned char>(orig)) << ") -> '"
                << format_byte(static_cast<unsigned char>(gen)) << "' ("
                << static_cast<int>(static_cast<unsigned char>(gen)) << "): "
                << count << " occurrences\n";
        }

        return false;
    }

    cout << "Files match perfectly. All " << original.size() << " bytes are identical.\n";
    return true;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        setup_interrupt_handler();

        command_line_parser parser;
        parser.add_option("train", "Train a transformer model");
        parser.add_option("generate", "Generate data from a previously trained model");
        parser.add_option("verify", "Verify generated output against original data");
        parser.add_option("tokenize-only", "Only tokenize the input file and save tokens");
        parser.add_option("data", "Path to the data file (default: data.txt)", 1);
        parser.add_option("max-tokens", "Maximum number of tokens to load in memory", 1);
        parser.add_option("max-bytes", "Maximum number of bytes to process from data", 1);
        parser.add_option("percent", "Percentage of data to process (0-100)", 1);
        parser.add_option("learning-rate", "Set the learning rate (default: 3e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 64)", 1);
        parser.add_option("patience", "Iterations without progress before early stopping (default: 15000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 10)", 1);
        parser.add_option("alpha", "Set the weight decay for Adam (default: 0.004)", 1);
        parser.add_option("beta1", "Set Adam's first moment coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set Adam's second moment coefficient (default: 0.999)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_slm_data_model.dat)", 1);
        parser.add_option("output-file", "Path for output (default: data_generated.txt)", 1);
        parser.add_option("tokenizer", "Path to pre-trained tokenizer (default: data_tokenizer.vocab)", 1);
        parser.add_option("tokens-file", "Path to pre-tokenized tokens file (optional)", 1);
        parser.add_option("force-tokenize", "Force tokenization even if tokens file exists");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("train") && !parser.option("generate") &&
            !parser.option("verify") && !parser.option("tokenize-only"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 3e-4);
        const size_t batch_size = get_option(parser, "batch-size", 64);
        const long patience = get_option(parser, "patience", 15000);
        const size_t max_epochs = get_option(parser, "max-epochs", 10);
        const double alpha = get_option(parser, "alpha", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.999);
        const std::string model_file = get_option(parser, "model-file", "data_model.dat");
        const std::string output_file = get_option(parser, "output-file", "data_generated.txt");
        const std::string data_path = get_option(parser, "data", "data.txt");
        const long max_seq_len = 50;
        const long num_layers = 4;
        const long num_heads = 6;
        const long embedding_dim = 228;
        const std::string tokenizer_path = get_option(parser, "tokenizer", "data_tokenizer.vocab");
        // Default number of prompt tokens = input sequence length
        const bool force_tokenize = parser.option("force-tokenize");
        const long num_tokens = 1500;

        // Calculate max bytes to process
        size_t max_bytes = 0, max_tokens = 0;
        if (parser.option("max-tokens"))
            max_tokens = std::stoul(parser.option("max-tokens").argument());
        if (parser.option("max-bytes")) {
            max_bytes = std::stoul(parser.option("max-bytes").argument());
        }
        else if (parser.option("percent")) {
            double percent = std::stod(parser.option("percent").argument());
            size_t file_size = get_file_size(data_path);
            if (file_size > 0) {
                max_bytes = static_cast<size_t>(file_size * percent / 100.0);
                cout << "Processing " << percent << "% of data = " << max_bytes << " bytes\n";
            }
            else {
                cerr << "Warning: Cannot determine file size for percentage calculation\n";
            }
        }

        // Tokenizer BPE
        bpe_tokenizer tokenizer;

        // Load pre-trained tokenizer
        if (file_exists(tokenizer_path)) {
            cout << "Loading pre-trained tokenizer from: " << tokenizer_path << endl;
            deserialize(tokenizer_path) >> tokenizer;
            cout << "Tokenizer loaded successfully with vocabulary size: " << tokenizer.get_vocab_size() << endl;
        }
        else {
            cout << "Pre-trained tokenizer not found at: " << tokenizer_path << endl;
            cout << "Will train a new tokenizer if in training mode." << endl;
        }

        // Determine tokens filename
        std::string tokens_file = parser.option("tokens-file") ?
            parser.option("tokens-file").argument() :
            generate_tokens_filename(data_path, max_bytes);

        using my_transformer = transformer_config<
            num_tokens,     // vocab_size
            num_layers,     // number of layers
            num_heads,      // number of attention heads
            embedding_dim,  // embedding dimension
            max_seq_len     // maximum sequence length
        >;

        // For GPU usage (if available)
        std::vector<int> gpus{ 0 };

        // Variables to store tokens (used in multiple modes)
        std::vector<int> full_tokens;
        bool tokens_loaded = false;

        // ----------------------------------------------------------------------------------------
        // Tokenize-only mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("tokenize-only")) {
            cout << "=== TOKENIZE-ONLY MODE ===\n";

            // Read the data file (or portion)
            cout << "Reading data file from: " << data_path;
            if (max_bytes > 0) cout << " (limited to " << max_bytes << " bytes)";
            cout << endl;

            std::string data_text = read_data(data_path, max_bytes);
            cout << "Read " << data_text.size() << " bytes\n";

            // Train a new tokenizer if needed
            if (!file_exists(tokenizer_path)) {
                cout << "Training new BPE tokenizer with vocabulary size " << num_tokens << "...\n";
                tokenizer.train(data_text, num_tokens, 1e6, true);
                serialize(tokenizer_path) << tokenizer;
                cout << "Tokenizer saved to " << tokenizer_path << endl;
            }

            // Tokenize the full text
            cout << "Tokenizing input text...\n";
            auto start_time = std::chrono::high_resolution_clock::now();
            int text_start_id = tokenizer.get_special_token_id("<text>"),
                text_end_id = tokenizer.get_special_token_id("</text>");
            if (text_start_id < 0 || text_end_id < 0)
                cout << "Warning: Special tokens not found in tokenizer vocabulary.\n";
            full_tokens.clear();
            full_tokens.push_back(text_start_id);
            auto encoded_tokens = tokenizer.encode(data_text);
            full_tokens.insert(full_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
            full_tokens.push_back(text_end_id);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto tokenize_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

            cout << "Tokenization completed in " << tokenize_time << " seconds.\n";
            cout << "Number of tokens: " << full_tokens.size() << endl;

            // Save tokens
            cout << "Saving tokens to file: " << tokens_file << endl;
            if (save_tokens_to_file(full_tokens, tokens_file)) {
                cout << "Tokens successfully saved.\n";
            }
            else {
                cerr << "Failed to save tokens.\n";
            }

            return 0;
        }

        // ----------------------------------------------------------------------------------------
        // Training mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";

            // Check if we should load pre-tokenized tokens
            if (!force_tokenize && file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens from file...\n";
                if (load_tokens_from_file(full_tokens, tokens_file)) {
                    cout << "Loaded " << full_tokens.size() << " tokens from file.\n";
                    if (max_tokens > 0 && max_tokens < full_tokens.size()) {
                        full_tokens.resize(max_tokens);
                        cout << "But limited to " << full_tokens.size() << " tokens for training.\n";
                    }
                    tokens_loaded = true;
                }
                else {
                    cerr << "Failed to load tokens from file. Will tokenize again.\n";
                }
            }

            if (!tokens_loaded) {
                // 1) Read the data file (or portion)
                cout << "Reading data file from: " << data_path;
                if (max_bytes > 0) cout << " (limited to " << max_bytes << " bytes)";
                cout << endl;

                std::string data_text = read_data(data_path, max_bytes);
                cout << "Read " << data_text.size() << " bytes\n";

                // Train a new tokenizer if needed
                if (!file_exists(tokenizer_path)) {
                    cout << "Training new BPE tokenizer with vocabulary size " << num_tokens << "...\n";
                    tokenizer.train(data_text, num_tokens, 1e6, true);
                    serialize(tokenizer_path) << tokenizer;
                    cout << "Tokenizer saved to " << tokenizer_path << endl;
                }

                // Tokenize the full text
                cout << "Tokenizing input text...\n";
                int text_start_id = tokenizer.get_special_token_id("<text>"),
                    text_end_id = tokenizer.get_special_token_id("</text>");
                if (text_start_id < 0 || text_end_id < 0)
                    cout << "Warning: Special tokens not found in tokenizer vocabulary.\n";
                auto start_time = std::chrono::high_resolution_clock::now();
                full_tokens.clear();
                full_tokens.push_back(text_start_id);
                auto encoded_tokens = tokenizer.encode(data_text);
                full_tokens.insert(full_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
                full_tokens.push_back(text_end_id);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto tokenize_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

                cout << "Tokenization completed in " << tokenize_time << " seconds.\n";
                cout << "Number of tokens: " << full_tokens.size() << endl;

                // Save tokens for future use
                cout << "Saving tokens to file: " << tokens_file << endl;
                if (save_tokens_to_file(full_tokens, tokens_file)) {
                    cout << "Tokens successfully saved for future use.\n";
                }
                else {
                    cerr << "Warning: Failed to save tokens for future use.\n";
                }
            }

            // Prepare training sequences (sliding window)
            cout << "Preparing training sequences...\n";
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            // Calculate the maximum number of sequences we can create
            size_t num_sequences = full_tokens.size() - max_seq_len;
            if (num_sequences <= 0) {
                cerr << "Error: Not enough tokens to create training sequences. Need at least "
                    << (max_seq_len + 1) << " tokens.\n";
                return 1;
            }

            cout << "Creating training samples...\n";

            // For very large datasets, using a stride can reduce training time 
            // without significantly affecting model quality
            size_t stride = 1;  // Default: use every possible sequence
            const size_t max_samples = 10e6;  // Optional: limit total samples to prevent memory issues

            // If dataset is very large, use adaptive stride
            if (num_sequences > max_samples && max_samples > 0) {
                stride = num_sequences / max_samples + 1;
                cout << "Dataset is large. Using stride of " << stride
                    << " to limit samples to approximately " << max_samples << "\n";
            }

            // Reserve memory for better performance
            samples.reserve(num_sequences / stride + 1);
            labels.reserve(num_sequences / stride + 1);

            // Create training samples with stride
            for (size_t start = 0; start < num_sequences; start += stride) {
                matrix<int, 0, 1> seq(max_seq_len, 1);
                for (long t = 0; t < max_seq_len; ++t) {
                    seq(t, 0) = full_tokens[start + t];
                }
                samples.push_back(seq);
                labels.push_back(full_tokens[start + max_seq_len]);

                if (samples.size() % 10000 == 0) {
                    cout << "Created " << samples.size() << " training samples ("
                        << (start * 100 / num_sequences) << "%)...\r";
                }
            }
            full_tokens.clear();
            cout << "Created " << samples.size() << " training samples (100%)...\n";

            // Build and train the network
            using net_type = my_transformer::network_type<true>;
            net_type net;
            cout << my_transformer::model_info::describe() << endl;
            if (file_exists(model_file)) deserialize(model_file) >> net;

            // Create trainer
            dnn_trainer<net_type, adam> trainer(net, adam(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            // For perfect memorization, we allow more epochs without improvement
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_max_num_epochs(max_epochs); // More epochs for perfect memorization
            trainer.set_synchronization_file("data_trainer.sync", std::chrono::minutes(10));
            trainer.be_quiet();

            // Custom training loop - trainer.train(samples, labels)
            cout << "Starting training...\n";
            size_t epoch = 0, samples_seen = 0, batches_seen = 0;
            double total_loss = 0;
            auto start_time = std::chrono::steady_clock::now();

            // Shuffle indices for epoch
            std::vector<size_t> indices(samples.size());
            std::iota(indices.begin(), indices.end(), 0);

            while (epoch < max_epochs && trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && !g_terminate_flag.load())
            {
                // Shuffle for new epoch
                std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

                // Process mini-batches
                for (size_t i = 0; i < samples.size() && !g_terminate_flag.load(); i += batch_size)
                {
                    // Get current mini-batch
                    std::vector<matrix<int, 0, 1>> batch_samples;
                    std::vector<unsigned long> batch_labels;

                    batch_samples.reserve(batch_size);
                    batch_labels.reserve(batch_size);

                    for (size_t j = 0; j < batch_size; ++j) {
                        size_t pos = (i + j) >= indices.size() ? j : (i + j);
                        batch_samples.push_back(samples[indices[pos]]);
                        batch_labels.push_back(labels[indices[pos]]);
                    }

                    // Train on this batch
                    trainer.train_one_step(batch_samples, batch_labels);
                    double loss = trainer.get_average_loss();

                    // Update stats
                    total_loss += loss;
                    samples_seen += batch_size;
                    batches_seen++;

                    // Progress reporting
                    if (batches_seen % 100 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        double avg_loss = total_loss / batches_seen;
                        double samples_per_sec = samples_seen / (elapsed > 0 ? elapsed : 1);

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " \t batch: " << batches_seen
                            << " \t samples: " << samples_seen
                            << " \t loss: " << avg_loss
                            << " \t speed: " << samples_per_sec << " samples/sec\n";
                        cout.flush();
                    }
                }
                epoch++;

                // Evaluate progress at end of epoch
                cout << ">>> completed epoch " << epoch << " - average loss: " << (total_loss / batches_seen) << endl;
            }

            // Save model
            net.clean();
            serialize(model_file) << net << tokenizer;
            cout << "Model saved to " << model_file << "\n";
            std::remove("data_trainer.sync");
            std::remove("data_trainer.sync_");

            // Evaluate on training set
            {
                if (!g_terminate_flag.load()) {
                    cout << "Evaluating model accuracy...\n";
                    using net_infer = my_transformer::network_type<false>;
                    net_infer g_infer = net;
                    auto predicted = g_infer(samples);
                    size_t correct = 0;
                    for (size_t i = 0; i < labels.size(); ++i)
                        if (predicted[i] == labels[i]) correct++;
                    double accuracy = (double)correct / labels.size();
                    cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";

                    // We need perfect accuracy to reconstruct data
                    if (accuracy < 0.999) {
                        cout << "WARNING: Model accuracy is less than 99.90%. The model may not "
                            << "perfectly reconstruct the input text.\n";
                    }
                }
            }
        }

        // ----------------------------------------------------------------------------------------
        // Generation mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("generate"))
        {
            cout << "=== GENERATION MODE ===\n";

            // Load the model
            using net_infer = my_transformer::network_type<false>;
            net_infer net;
            if (file_exists(model_file)) {
                deserialize(model_file) >> net >> tokenizer;
                cout << "Loaded model from " << model_file << "\n";
                cout << "Number of model parameters: " << count_parameters(net) << endl;
            }
            else {
                cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }

            // Check that tokenizer is loaded
            if (tokenizer.get_vocab_size() == 0) {
                cerr << "Error: Tokenizer not loaded. Please provide a valid tokenizer file.\n";
                return 0;
            }

            // Read beginning of data file for prompt
            std::vector<int> prompt_tokens;

            // Check if we have pre-tokenized tokens
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens for prompt...\n";

                // We only need max_seq_len tokens, so we can load
                // just the necessary part of the file
                std::ifstream file(tokens_file, std::ios::binary);
                if (!file) {
                    cerr << "Failed to open tokens file: " << tokens_file << endl;
                }
                else {
                    // Read total number of tokens
                    uint64_t num_tokens;
                    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));

                    // Read only the first max_seq_len tokens
                    size_t tokens_to_read = std::min(static_cast<size_t>(max_seq_len), static_cast<size_t>(num_tokens));
                    prompt_tokens.resize(tokens_to_read);

                    for (size_t i = 0; i < tokens_to_read; ++i) {
                        uint32_t t;
                        file.read(reinterpret_cast<char*>(&t), sizeof(t));
                        prompt_tokens[i] = static_cast<int>(t);
                    }

                    cout << "Loaded " << prompt_tokens.size() << " tokens for prompt from file.\n";
                }
            }

            // If we couldn't load tokens, tokenize the prompt text
            if (prompt_tokens.empty()) {
                cout << "Reading initial prompt from data...\n";
                std::string data_prompt;

                if (file_exists(data_path)) {
                    // Read a portion large enough to cover the first tokens
                    std::ifstream file(data_path, std::ios::binary);
                    // Buffer intentionally large to ensure we have enough text for tokens
                    char buffer[max_seq_len * 10];
                    file.read(buffer, sizeof(buffer));
                    size_t bytes_read = file.gcount();
                    data_prompt = std::string(buffer, bytes_read);
                }
                else {
                    cerr << "Error: Cannot find original data file for initial prompt.\n";
                    return 0;
                }

                // Tokenize the prompt
                cout << "Tokenizing prompt...\n";
                int text_start_id = tokenizer.get_special_token_id("<text>");
                prompt_tokens.clear();
                prompt_tokens.push_back(text_start_id);
                auto encoded_tokens = tokenizer.encode(data_prompt);
                prompt_tokens.insert(prompt_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
            }

            // Limit to requested number of tokens (exact, no padding)
            if (prompt_tokens.size() > (size_t)max_seq_len) {
                prompt_tokens.resize(max_seq_len);
            }
            else if (prompt_tokens.size() < (size_t)max_seq_len) {
                cerr << "Warning: Not enough tokens in prompt. Got " << prompt_tokens.size()
                    << ", needed " << max_seq_len << ". Consider using a larger input file.\n";
                return 0;
            }
            cout << "Using " << prompt_tokens.size() << " tokens for initial prompt\n";

            // Put prompt in input sequence
            inference_context llm_context(max_seq_len, 4, tokenizer.get_special_token_id("<pad>"));
            llm_context.add_tokens(prompt_tokens);
            auto input_seq = llm_context.get_input_window();

            // Determine text size to generate
            size_t target_size = (max_bytes > 0) ? max_bytes : get_file_size(data_path);
            cout << "Will generate approximately " << target_size << " bytes\n";

            // Open output file
            std::ofstream outfile(output_file, std::ios::binary);
            if (!outfile) {
                cerr << "Error: Cannot open output file: " << output_file << "\n";
                return 0;
            }

            // Write initial text (corresponding to prompt tokens)
            std::string initial_text = tokenizer.decode(prompt_tokens, false);
            outfile.write(initial_text.c_str(), initial_text.size());

            // Generate the rest of the text autoregressively
            cout << "Starting autoregressive generation...\n";

            // Buffer for accumulation before writing
            std::vector<int> token_buffer;
            const size_t buffer_size = 100;

            // Save start time to measure execution time
            auto start_time = std::chrono::high_resolution_clock::now();
            size_t total_bytes = initial_text.size();
            size_t token_count = prompt_tokens.size();

            // Generate until target size is reached
            int start_of_text = tokenizer.get_special_token_id("<text>"),
                end_of_text = tokenizer.get_special_token_id("</text>"), next_token = 0;
            while (total_bytes < target_size && next_token != start_of_text && next_token != end_of_text
                && !g_terminate_flag.load()) {
                // Predict next token
                std::vector<matrix<int, 0, 1>> in_tokens = { input_seq, input_seq };
                auto out_token = net(in_tokens);
                next_token = static_cast<int>(out_token[0]);
                token_buffer.push_back(next_token);
                token_count++;

                // Shift the input window
                llm_context.add_token(next_token);
                input_seq = llm_context.get_input_window();

                // If buffer is full, write to file
                if (token_buffer.size() >= buffer_size) {
                    std::string chunk = tokenizer.decode(token_buffer, false);
                    outfile.write(chunk.c_str(), chunk.size());
                    total_bytes += chunk.size();
                    token_buffer.clear();

                    // Display progress
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        current_time - start_time).count();
                    double tokens_per_second = (token_count - input_seq.size()) / (elapsed > 0 ? elapsed : 1);

                    cout << "Generated " << (token_count - input_seq.size()) << " tokens, "
                        << total_bytes << " bytes ("
                        << (total_bytes * 100.0 / target_size) << "%) - "
                        << tokens_per_second << " tokens/sec - "
                        << "Est. completion: "
                        << (int)((target_size - total_bytes) / (tokens_per_second * (chunk.size() / (double)buffer_size)))
                        << " seconds\r";
                }
                if (max_tokens > 0 && token_count >= max_tokens) break;
            }

            // Flush remaining buffer
            if (!token_buffer.empty()) {
                std::string chunk = tokenizer.decode(token_buffer, false);
                outfile.write(chunk.c_str(), chunk.size());
                total_bytes += chunk.size();
            }
            outfile.flush();
            outfile.close();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();

            cout << "Generation complete in " << total_time << " seconds!\n";
            cout << "Generated " << (token_count - input_seq.size()) << " tokens after prompt, "
                << total_bytes << " bytes total\n";
            cout << "Output saved to " << output_file << "\n";
        }

        // ----------------------------------------------------------------------------------------
        // Verification mode - Compare original and generated file
        // ----------------------------------------------------------------------------------------
        if (parser.option("verify"))
        {
            cout << "=== VERIFICATION MODE ===\n";

            if (!file_exists(data_path)) {
                cerr << "Error: Original data file not found at " << data_path << "\n";
                return 0;
            }

            if (!file_exists(output_file)) {
                cerr << "Error: Generated file not found at " << output_file << "\n";
                return 0;
            }

            // Read generated file
            cout << "Reading generated file...\n";
            std::string generated = read_data(output_file);

            // Read the same portion of original file
            cout << "Reading original file (same size as generated)...\n";
            std::string original = read_data(data_path, generated.size());

            cout << "Verifying byte-for-byte match...\n";
            bool match = verify_match(original, generated);

            if (match)
                cout << "SUCCESS: The generated file matches the original text perfectly!\n";
            else
                cout << "FAILED: The generated file does not match the original text.\n";
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}