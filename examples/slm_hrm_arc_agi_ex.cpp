/*!
    @file slm_hrm_arc_agi.cpp
    @brief Transformer-based training for ARC-AGI reasoning tasks

    This program implements a complete training and evaluation pipeline for
    solving ARC-AGI (Abstraction and Reasoning Corpus) tasks using a
    Transformer-based architecture.

    Key capabilities:
    - Learning visual transformation patterns from demonstrations
    - Generating output grids token-by-token autoregressively
    - Handling non-square grids through implicit dimension encoding

    Usage modes:
    --train       Train model on ARC-AGI training set
    --eval        Evaluate model on test pairs

    References:
    [1] Chollet, "On the Measure of Intelligence" (ARC-AGI)
        arXiv:1911.01547
!*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <csignal>

#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/serialize.h>
#include <dlib/dnn.h>

using namespace std;
using namespace dlib;

namespace dlib
{
    class display_
    {
    public:
        explicit display_(const std::string& label = "") :
            layer_label(label)
        {
        }

        display_(const display_& other) :
            layer_label(other.layer_label)
        {
        }

        display_& operator=(const display_& other)
        {
            if (this != &other) {
                layer_label = other.layer_label;
            }
            return *this;
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
            // No setup needed
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            const tensor& input = sub.get_output();

            // Display tensor dimensions
            std::cout << "[DISPLAY";
            if (!layer_label.empty())
                std::cout << " " << layer_label;
            std::cout << "] "
                << "num_samples=" << input.num_samples() << ", "
                << "k=" << input.k() << ", "
                << "nr=" << input.nr() << ", "
                << "nc=" << input.nc()
                << std::endl;

            // Copy input to output (transparent pass-through)
            output.copy_size(input);
            tt::copy_tensor(false, output, 0, input, 0, input.k());
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            // Simply propagate gradients unchanged
            tensor& prev_grad = sub.get_gradient_input();
            tt::copy_tensor(true, prev_grad, 0, gradient_input, 0, gradient_input.k());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const display_& item, std::ostream& out)
        {
            serialize("display_", out);
            serialize(item.layer_label, out);
        }

        friend void deserialize(display_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "display_")
                throw serialization_error("Unexpected version '" + version +
                    "' while deserializing display_");
            deserialize(item.layer_label, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const display_& item)
        {
            out << "display";
            if (!item.layer_label.empty())
                out << " (" << item.layer_label << ")";
            return out;
        }

        friend void to_xml(const display_& item, std::ostream& out)
        {
            out << "<display";
            if (!item.layer_label.empty())
                out << " label='" << item.layer_label << "'";
            out << "/>\n";
        }

        inline dpoint map_input_to_output(const dpoint& p) const { return p; }
        inline dpoint map_output_to_input(const dpoint& p) const { return p; }

    private:
        std::string layer_label;
        resizable_tensor params; // No trainable parameters
    };

    template <typename SUBNET>
    using display = add_layer<display_, SUBNET>;

    /*!
        Provides a flexible configuration for HRM-based transformer models with dual
        recurrent modules (H and L) for hierarchical reasoning.

        Template parameters:
        @param vocab_size Vocabulary size for token embedding
        @param num_h_layers Number of transformer layers in H module (high-level)
        @param num_l_layers Number of transformer layers in L module (low-level)
        @param num_heads Number of attention heads
        @param embedding_dim Dimension of token embeddings (must be divisible by num_heads)
        @param window_len Maximum sequence length (context window)
        @param hrm_N Number of high-level cycles
        @param hrm_T Number of low-level steps per high-level cycle
        @param activation_func Activation function type
        @param dropout_policy Dropout regularization policy
    !*/
    template<
        long vocab_size = ARC_VOCAB_SIZE_TOTAL,                    // 17 tokens for ARC-AGI
        long num_h_layers = 4,                                     // H module depth
        long num_l_layers = 4,                                     // L module depth
        long num_heads = 8,                                        // Attention heads
        long embedding_dim = 512,                                  // Embedding dimension
        long window_len = 768,                                     // Context window
        long hrm_N = 2,                                            // High-level cycles
        long hrm_T = 2,                                            // Low-level steps
        template <typename> class activation_func = gelu,          // Activation
        template <typename> class dropout_policy = dropout_10      // Dropout policy
    >
    struct hrm_config {
        // Core model parameters
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_H_LAYERS = num_h_layers;
        static constexpr long NUM_L_LAYERS = num_l_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long WINDOW_LEN = window_len;
        static constexpr long HRM_N = hrm_N;
        static constexpr long HRM_T = hrm_T;

        // Compile-time validation of model configuration
        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_H_LAYERS > 0, "Number of H layers must be positive");
            static_assert(NUM_L_LAYERS > 0, "Number of L layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM % NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
            static_assert(WINDOW_LEN > 0, "Window length must be positive");
            static_assert(HRM_N > 0, "HRM N cycles must be positive");
            static_assert(HRM_T > 0, "HRM T steps must be positive");
        };

        // Basic 3x3 convolution with padding=1, stride=2 for exact 2x downsampling
        template <typename SUBNET>
        using con3d = add_layer<con_<1, 3, 3, 2, 2, 1, 1>, SUBNET>;

        // Recursive implementation of signal compressor
        template <int depth, template<typename> class NORM, typename SUBNET>
        struct signal_compressor_impl
        {
            using type = relu<NORM<
                con3d<typename signal_compressor_impl<depth - 1, NORM, SUBNET>::type>>>;
        };

        // Base case: depth=0 returns input unchanged
        template <template<typename> class NORM, typename SUBNET>
        struct signal_compressor_impl<0, NORM, SUBNET>
        {
            using type = SUBNET;
        };

        // Signal compressor for training mode (with batch normalization)
        template <int depth, typename SUBNET>
        using signal_compressor_t = typename signal_compressor_impl<depth, bn_con, SUBNET>::type;

        // Signal compressor for inference mode (with affine transform)
        template <int depth, typename SUBNET>
        using signal_compressor_i = typename signal_compressor_impl<depth, affine, SUBNET>::type;

        // Network component definitions for training (with dropout)
        using t_h_net_type = transformer_stack<NUM_H_LAYERS, activation_func, dropout_policy,
            WINDOW_LEN, EMBEDDING_DIM, NUM_HEADS,
            input<matrix<float>>>;
        using t_l_net_type = transformer_stack<NUM_L_LAYERS, activation_func, dropout_policy,
            WINDOW_LEN, EMBEDDING_DIM, NUM_HEADS,
            input<matrix<float>>>;

        // Network component definitions for inference (without dropout)
        using i_h_net_type = transformer_stack<NUM_H_LAYERS, activation_func, multiply,
            WINDOW_LEN, EMBEDDING_DIM, NUM_HEADS,
            input<matrix<float>>>;
        using i_l_net_type = transformer_stack<NUM_L_LAYERS, activation_func, multiply,
            WINDOW_LEN, EMBEDDING_DIM, NUM_HEADS,
            input<matrix<float>>>;

        // Complete network type selector
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            loss_multiclass_log<fc<VOCAB_SIZE, rms_norm<
            tag10<hrm<t_h_net_type, t_l_net_type, HRM_N, HRM_T,
            token_embeddings<VOCAB_SIZE, EMBEDDING_DIM,
            input<matrix<long, 0, 1>>>>>>>>,
            loss_multiclass_log<fc<VOCAB_SIZE, rms_norm<
            tag10<hrm<i_h_net_type, i_l_net_type, HRM_N, HRM_T,
            token_embeddings<VOCAB_SIZE, EMBEDDING_DIM,
            input<matrix<long, 0, 1>>>>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "HRM network configuration:\n"
                    << "- Vocabulary: " << VOCAB_SIZE << " tokens\n"
                    << "- H module: " << NUM_H_LAYERS << " transformer layers\n"
                    << "- L module: " << NUM_L_LAYERS << " transformer layers\n"
                    << "- Attention heads: " << NUM_HEADS << "\n"
                    << "- Embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- Context window: " << WINDOW_LEN << " tokens\n"
                    << "- HRM cycles: N=" << HRM_N << " (high-level), T=" << HRM_T << " (low-level)\n"
                    << "- Total reasoning steps: " << (HRM_N * HRM_T) << " iterations";
                return ss.str();
            }
        };
    };

    /*!
        ensures
            - Returns detailed parameter count for HRM-based networks
            - Provides breakdown by component (H, L, embeddings, output)
    !*/
    struct hrm_param_info
    {
        size_t h_net_params;
        size_t l_net_params;
        size_t other_params;
        size_t total_params;

        void print() const
        {
            std::cout << "Parameter breakdown:\n"
                << "  H-module: " << h_net_params << "\n"
                << "  L-module: " << l_net_params << "\n"
                << "  Other layers: " << other_params << "\n"
                << "  Total: " << total_params << " parameters" << "\n\n";
        }
    };

    template <typename net_type>
    hrm_param_info get_hrm_param_info(const net_type& net)
    {
        hrm_param_info info;

        const auto& hrm_layer = layer<tag10>(net).subnet().layer_details();

        info.h_net_params = count_parameters(hrm_layer.get_h_net());
        info.l_net_params = count_parameters(hrm_layer.get_l_net());

        info.other_params = count_parameters(net);
        info.total_params = info.h_net_params + info.l_net_params + info.other_params;

        return info;
    }
}

// Cross-platform signal handling
namespace {
    std::atomic<bool> g_terminate_flag(false);

#ifdef _WIN32
    BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
        if (ctrl_type == CTRL_C_EVENT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up..." << endl;
            return TRUE;
        }
        return FALSE;
    }
#else
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up..." << endl;
        }
    }
#endif

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

// Utility function to validate token sequence before detokenization
bool validate_token_sequence(const std::vector<long>& tokens, bool verbose = false)
{
    if (tokens.empty()) return false;

    std::vector<long> current_row_lengths;
    long current_row_length = 0;

    for (long token : tokens) {
        if (token == TOKEN_ROW_END) {
            if (current_row_length > 0) {
                current_row_lengths.push_back(current_row_length);
                current_row_length = 0;
            }
        }
        else if (token >= COLOR_0 && token <= COLOR_9) {
            current_row_length++;
        }
        else if (token == TOKEN_END_OF_OUTPUT || token == TOKEN_SEP_IO || token == TOKEN_SEP_PAIR) {
            break;
        }
    }

    // Check if all rows have the same length
    if (current_row_lengths.empty()) {
        if (verbose) cout << "      Validation: No complete rows found\n";
        return false;
    }

    long expected_length = current_row_lengths[0];
    for (size_t i = 1; i < current_row_lengths.size(); ++i) {
        if (current_row_lengths[i] != expected_length) {
            if (verbose) {
                cout << "      Validation: Inconsistent row lengths detected\n";
                cout << "        Row 0 has " << expected_length << " columns\n";
                cout << "        Row " << i << " has " << current_row_lengths[i] << " columns\n";
            }
            return false;
        }
    }

    return true;
}

/*!
    WHAT THIS OBJECT REPRESENTS
        Tracks the generation state of an ARC-AGI output grid during autoregressive
        token generation. Monitors row consistency and detects invalid patterns early.

        ARC-AGI constraints:
        - Maximum grid size: 30×30
        - All rows must have the same length
        - Valid tokens: 0-9 (colors), TOKEN_ROW_END
!*/
struct generation_state
{
    std::vector<long> row_lengths;      // Length of each completed row
    long current_row_length = 0;        // Length of current incomplete row
    bool is_valid = true;               // Whether generation is valid so far
    bool is_complete = false;           // Whether a complete grid has been generated

    void add_token(long token)
    {
        if (token == TOKEN_ROW_END)
        {
            if (current_row_length > 0)
            {
                // Check consistency with previous rows
                if (!row_lengths.empty() && row_lengths[0] != current_row_length)
                    is_valid = false;  // Inconsistent row lengths

                row_lengths.push_back(current_row_length);
                current_row_length = 0;

                // Stop if too many rows (ARC-AGI max is 30)
                if (row_lengths.size() > 30) is_valid = false;
            }
        }
        else if (token >= COLOR_0 && token <= COLOR_9)
        {
            current_row_length++;

            // Stop if row is too long (ARC-AGI max is 30)
            if (current_row_length > 30) is_valid = false;
        }
    }

    bool should_stop() const
    {
        // Stop if generation has become invalid
        if (!is_valid) return true;

        // Continue if we have a valid grid in progress
        // We could add more sophisticated stopping conditions here
        // (e.g., if we detect the expected output size)

        return false;
    }

    size_t num_rows() const { return row_lengths.size(); }

    long grid_width() const
    {
        return row_lengths.empty() ? 0 : row_lengths[0];
    }
};

struct generation_result
{
    arc_grid_t grid;
    long context_size;
    long window_size;
    bool context_fits;
    long tokens_truncated;

    generation_result(long ctx_size, long win_size)
        : context_size(ctx_size),
        window_size(win_size),
        context_fits(ctx_size <= win_size),
        tokens_truncated(std::max(0L, ctx_size - win_size))
    {
    }
};

template <typename TASK_TYPE, typename PAIR_TYPE>
long compute_context_size(const TASK_TYPE& task, const PAIR_TYPE& test_pair)
{
    auto input_context = arc_agi_manager::tokenize_input_context(task, test_pair);
    return input_context.size();
}

/*!
    ensures
        - Generates the output grid for a given ARC-AGI test pair
        - Uses autoregressive token generation with smart early stopping
        - Stops generation when:
          * TOKEN_END_OF_OUTPUT is generated
          * Invalid pattern detected (inconsistent row lengths)
          * Maximum grid size exceeded (30×30)
          * Maximum token limit reached (1024 tokens)
        - Throws std::runtime_error if generation produces invalid grid
        - Returns the predicted output grid if successful
!*/
template <typename NET_TYPE, long WINDOW_LEN>
generation_result generate_output_for_test_pair_with_info(
    NET_TYPE& net,
    const arc_task& task,
    const arc_task_pair& test_pair,
    bool verbose = false)
{
    constexpr long MAX_OUTPUT_TOKENS = 1024;
    constexpr long MAX_ROWS = 30;

    // Tokenize input context
    auto input_context = arc_agi_manager::tokenize_input_context(task, test_pair);

    // Create result with context info
    generation_result result(input_context.size(), WINDOW_LEN);

    if (verbose) {
        cout << "  Input context: " << input_context.size() << " tokens\n";
        cout << "  Window size: " << WINDOW_LEN << " tokens\n";
        if (result.context_fits) {
            cout << "  Context fits: YES (margin: " << (WINDOW_LEN - input_context.size()) << " tokens)\n";
        }
        else {
            cout << "  Context fits: NO (truncated: " << result.tokens_truncated << " tokens)\n";
        }
    }

    // Initialize context window
    std::vector<long> context_window(WINDOW_LEN);
    long start_pos = std::max(0L, input_context.size() - WINDOW_LEN);

    for (long i = 0; i < WINDOW_LEN; ++i) {
        long idx = start_pos + i;
        context_window[i] = (idx < input_context.size()) ?
            input_context(idx) : TOKEN_PADDING;
    }

    // Generate tokens
    std::vector<long> generated_tokens;
    generation_state state;
    long generated_count = 0;

    while (generated_count < MAX_OUTPUT_TOKENS)
    {
        arc_token_sequence_t input_seq(WINDOW_LEN);
        for (long i = 0; i < WINDOW_LEN; ++i) {
            input_seq(i) = context_window[i];
        }

        const long next_token = net(input_seq);

        if (next_token == TOKEN_END_OF_OUTPUT) {
            if (verbose) {
                cout << "  Stopping: TOKEN_END_OF_OUTPUT generated\n";
            }
            break;
        }

        generated_tokens.push_back(next_token);
        generated_count++;

        state.add_token(next_token);

        if (state.should_stop())
        {
            if (verbose) {
                cout << "  Early stopping: invalid generation detected\n";
                cout << "    Rows generated: " << state.num_rows() << "\n";
            }

            if (!state.is_valid && state.num_rows() < 2)
            {
                throw std::runtime_error("Generation failed: invalid grid structure");
            }
            break;
        }

        if (state.num_rows() >= MAX_ROWS)
        {
            if (verbose) {
                cout << "  Stopping: maximum rows reached\n";
            }
            break;
        }

        for (long i = 0; i < WINDOW_LEN - 1; ++i) {
            context_window[i] = context_window[i + 1];
        }
        context_window[WINDOW_LEN - 1] = next_token;
    }

    if (verbose) {
        cout << "  Generated " << generated_count << " tokens\n";
        cout << "  Detected " << state.num_rows() << " complete rows\n";
    }

    if (!validate_token_sequence(generated_tokens, verbose)) {
        throw std::runtime_error("Invalid token sequence");
    }

    arc_token_sequence_t output_seq(generated_tokens.size());
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        output_seq(i) = generated_tokens[i];
    }

    result.grid = arc_agi_manager::detokenize_to_grid(output_seq, 0);
    return result;
}

/*!
    ensures
        - Returns true if the sample should be kept for training
        - Keeps sample if:
          1. No TOKEN_PADDING in sequence (full context visible), OR
          2. Has TOKEN_PADDING AND last token is TOKEN_GEN_START
             (context was left-padded to align TOKEN_GEN_START at end)
!*/
inline bool should_keep_sample(const arc_token_sequence_t& input_window,
    long target_token)
{
    const long window_len = input_window.size();
    bool has_padding = false;

    // Check for padding in the window
    for (long i = 0; i < window_len; ++i)
    {
        if (input_window(i) == TOKEN_PADDING)
        {
            has_padding = true;
            break;
        }
    }

    // No padding - keep the sample (full context is visible)
    if (!has_padding) return true;

    // Has padding - only keep if TOKEN_GEN_START is the last token
    if (input_window(window_len - 1) == TOKEN_GEN_START) return true;

    // Has padding but TOKEN_GEN_START is not at the end - discard
    return false;
}

/*!
    ensures
        - Filters training samples to keep only high-quality ones
        - Removes samples with padding, incorrect TOKEN_GEN_START position, etc.
        - Returns the number of samples removed
!*/
inline size_t filter_training_samples(
    std::vector<arc_token_sequence_t>& X,
    std::vector<unsigned long>& Y)
{
    DLIB_CASSERT(X.size() == Y.size(), "X and Y must have same size");

    size_t original_size = X.size();
    std::vector<arc_token_sequence_t> filtered_X;
    std::vector<unsigned long> filtered_Y;

    filtered_X.reserve(X.size());
    filtered_Y.reserve(Y.size());

    for (size_t i = 0; i < X.size(); ++i)
    {
        if (should_keep_sample(X[i], Y[i]))
        {
            filtered_X.push_back(X[i]);
            filtered_Y.push_back(Y[i]);
        }
    }
    X = std::move(filtered_X);
    Y = std::move(filtered_Y);

    return (original_size - X.size());
}

int main(int argc, char** argv)
{
    try
    {
        setup_interrupt_handler();

        command_line_parser parser;
        parser.add_option("train", "Train transformer model on ARC-AGI tasks");
        parser.add_option("eval", "Evaluate model on test pairs with generation");
        parser.add_option("training-path", "Path to training JSON files", 1);
        parser.add_option("eval-path", "Path to evaluation JSON files", 1);
        parser.add_option("model-file", "Path for model file", 1);
        parser.add_option("learning-rate", "Learning rate (default: 1e-4)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 4)", 1);
        parser.add_option("max-epochs", "Maximum training epochs (default: 10000)", 1);
        parser.add_option("patience", "Early stopping patience (default: 5000)", 1);
        parser.add_option("task-id", "Specific task ID to evaluate/generate", 1);
        parser.add_option("verbose", "Show detailed output during generation");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train") &&
            !parser.option("eval"))
        {
            parser.print_options();
            cout << "\nExample usage:\n"
                << "  Training:   " << argv[0] << " --train --training-path data/training --eval-path data/evaluation\n"
                << "  Evaluation: " << argv[0] << " --eval --eval-path data/evaluation\n"
                << "  Single task: " << argv[0] << " --eval --task-id 007bbfb7 --verbose\n";
            return 0;
        }

        // Configuration
        const std::string training_path = get_option(parser, "training-path", "data/training");
        const std::string eval_path = get_option(parser, "eval-path", "data/evaluation");
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_arc_agi_model.dat");
        const double learning_rate = get_option(parser, "learning-rate", 1e-4);
        const size_t batch_size = get_option(parser, "batch-size", 4);
        const size_t max_epochs = get_option(parser, "max-epochs", 10000);
        const long patience = get_option(parser, "patience", 5000);
        
        // Window length: 128 for quick testing, 512-1024 for better performance, 4096 for maximum context
        constexpr long WINDOW_LEN = 128;

        // Model configuration
        using arc_net_config = hrm_config<
            ARC_VOCAB_SIZE_TOTAL,   // vocab_size
            4,                      // num_h_layers
            4,                      // num_l_layers
            6,                      // num_heads
            228,                    // embedding_dim
            WINDOW_LEN,             // window_len
            2,                      // hrm_N (high-level cycles)
            2                       // hrm_T (low-level steps)
        >;
        cout << arc_net_config::model_info::describe() << "\n\n";
        using train_net_type = arc_net_config::network_type<true>;
        using infer_net_type = arc_net_config::network_type<false>;

        // Load ARC-AGI data
        arc_agi_manager data_mgr;
        data_mgr.load_data(training_path, eval_path);

        // ----------------------------------------------------------------------------------------
        // Training mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";

            if (data_mgr.num_training_tasks() == 0) {
                cerr << "Error: No training tasks loaded\n";
                return 1;
            }

            // Prepare training data from all tasks
            cout << "Preparing training data...\n";
            std::vector<arc_token_sequence_t> all_X;
            std::vector<unsigned long> all_Y;

            for (size_t task_idx = 0; task_idx < data_mgr.num_training_tasks(); ++task_idx)
            {
                const auto& task = data_mgr.get_training_task(task_idx);

                std::vector<arc_token_sequence_t> task_X;
                std::vector<long> task_Y;

                arc_agi_manager::prepare_training_data_batch(task, WINDOW_LEN, task_X, task_Y);

                all_X.insert(all_X.end(), task_X.begin(), task_X.end());

                // Convert long to unsigned long for dlib classification
                for (auto y : task_Y) {
                    all_Y.push_back(static_cast<unsigned long>(y));
                }

                if ((task_idx + 1) % 10 == 0) {
                    cout << "Processed " << (task_idx + 1) << "/"
                        << data_mgr.num_training_tasks() << " tasks...\r" << flush;
                }
            }
            size_t removed = filter_training_samples(all_X, all_Y);
            cout << "\nTotal training samples: " << all_X.size() << endl;

            // Build network
            train_net_type net;

            if (file_exists(model_file)) {
                cout << "Loading existing model from " << model_file << endl;
                deserialize(model_file) >> net;
            }
            cout << net << endl << endl; // Show the model architecture

            // Setup trainer
            std::vector<int> gpus{ 0 };
            dnn_trainer<train_net_type, adam> trainer(net, adam(0.1, 0.9, 0.95), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.be_quiet();

            // Training loop
            cout << "Starting training...\n";
            size_t epoch = 0;
            auto start_time = std::chrono::steady_clock::now();

            size_t batches_count = 0;
            while (trainer.get_learning_rate() >= 1e-6 && epoch < max_epochs && !g_terminate_flag.load())
            {
                // Shuffle indices
                std::vector<size_t> indices(all_X.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), std::default_random_engine());

                // Train epoch
                size_t batches_seen = 0;
                for (size_t i = 0; i < all_X.size() && !g_terminate_flag.load(); i += batch_size)
                {
                    std::vector<arc_token_sequence_t> batch_X;
                    std::vector<unsigned long> batch_Y;

                    batch_X.reserve(batch_size);
                    batch_Y.reserve(batch_size);

                    for (size_t j = 0; j < batch_size && (i + j) < all_X.size(); ++j) {
                        size_t idx = indices[i + j];
                        batch_X.push_back(all_X[idx]);
                        batch_Y.push_back(all_Y[idx]);
                    }                    

                    trainer.train_one_step(batch_X, batch_Y);
                    batches_seen++;

                    // Progress reporting
                    if (batches_count++ % 50 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " (batches: " << batches_seen << ")"
                            << " \t loss: " << trainer.get_average_loss()
                            << " \t patience: " << trainer.get_steps_without_progress()
                            << " \t time: " << elapsed << "s\n";
                        cout.flush();
                    }
                }

                epoch++;
            }

            // Save model
            net.clean();
            serialize(model_file) << net;
            cout << "Model saved to " << model_file << "\n";
        }

        // ----------------------------------------------------------------------------------------
        // Evaluation mode (with generation)
        // ----------------------------------------------------------------------------------------
        if (parser.option("eval"))
        {
            cout << "=== EVALUATION MODE ===\n";

            // Load model
            infer_net_type net;

            if (!file_exists(model_file)) {
                cerr << "Error: Model file not found: " << model_file << "\n";
                return 1;
            }

            deserialize(model_file) >> net;
            cout << "Model loaded.\n";
            auto info_params = get_hrm_param_info<infer_net_type>(net);
            info_params.print();

            const bool verbose = parser.option("verbose");
            const bool single_task = parser.option("task-id");

            // Statistics
            size_t total_tasks = 0;
            size_t tasks_with_correct_dims = 0;
            size_t tasks_fully_correct = 0;
            size_t generation_failures = 0;
            double total_pixel_accuracy = 0.0;

            // Get task list to evaluate
            std::vector<const arc_task*> tasks_to_eval;

            if (single_task) {
                std::string task_id = parser.option("task-id").argument();
                cout << "Evaluating single task: " << task_id << "\n\n";

                try {
                    tasks_to_eval.push_back(&data_mgr.get_evaluation_task_by_id(task_id));
                }
                catch (...) {
                    try {
                        tasks_to_eval.push_back(&data_mgr.get_training_task_by_id(task_id));
                    }
                    catch (...) {
                        cerr << "Error: Task not found: " << task_id << "\n";
                        return 1;
                    }
                }
            }
            else {
                // Evaluate all evaluation tasks
                for (size_t i = 0; i < data_mgr.num_evaluation_tasks(); ++i) {
                    tasks_to_eval.push_back(&data_mgr.get_evaluation_task(i));
                }
            }

            // Evaluate each task
            for (const arc_task* task_ptr : tasks_to_eval)
            {
                const arc_task& task = *task_ptr;

                if (task.test_pairs.empty()) {
                    if (verbose) cout << "Task " << task.task_id << ": No test pairs\n";
                    continue;
                }

                cout << "Task " << task.task_id << " (" << task.train_pairs.size()
                    << " train, " << task.test_pairs.size() << " test):\n";

                // Evaluate each test pair
                for (size_t pair_idx = 0; pair_idx < task.test_pairs.size(); ++pair_idx)
                {
                    const auto& test_pair = task.test_pairs[pair_idx];

                    if (verbose) {
                        cout << "  Test pair " << (pair_idx + 1) << "/" << task.test_pairs.size() << ":\n";
                    }

                    // Calculate context size BEFORE attempting generation
                    long context_size = compute_context_size(task, test_pair);
                    long tokens_truncated = std::max(0L, context_size - WINDOW_LEN);
                    bool context_fits = (context_size <= WINDOW_LEN);

                    // Display context information first
                    cout << "    Context: " << context_size << " tokens";
                    if (context_fits) {
                        cout << " (fits in window: " << WINDOW_LEN << ")\n";
                    }
                    else {
                        cout << " (TRUNCATED: -" << tokens_truncated
                            << " tokens, window: " << WINDOW_LEN << ")\n";
                    }

                    // Now attempt generation
                    generation_result gen_result(context_size, WINDOW_LEN);
                    bool generation_failed = false;

                    try {
                        gen_result = generate_output_for_test_pair_with_info<infer_net_type, WINDOW_LEN>(
                            net, task, test_pair, verbose);
                    }
                    catch (const std::exception& e) {
                        if (verbose) {
                            cout << "    Generation error: " << e.what() << "\n";
                        }
                        generation_failed = true;
                        generation_failures++;
                    }

                    if (generation_failed) {
                        cout << "    Generation: FAILED\n";
                        cout << "    Dimensions: KO\n";
                        cout << "    Pixel accuracy: 0.0% (0/0)\n";
                        cout << "    Fully correct: NO\n";
                        total_tasks++;
                        continue;
                    }

                    const arc_grid_t& generated = gen_result.grid;

                    // Rest of the evaluation code...
                    bool dims_correct = (generated.nr() == test_pair.output_rows &&
                        generated.nc() == test_pair.output_cols);

                    if (verbose) {
                        cout << "    Expected: " << test_pair.output_rows << "x" << test_pair.output_cols << "\n";
                        cout << "    Generated: " << generated.nr() << "x" << generated.nc() << "\n";
                    }

                    long correct_pixels = 0;
                    long total_pixels = 0;

                    if (dims_correct) {
                        total_pixels = generated.nr() * generated.nc();
                        for (long r = 0; r < generated.nr(); ++r) {
                            for (long c = 0; c < generated.nc(); ++c) {
                                if (generated(r, c) == test_pair.output(r, c)) {
                                    correct_pixels++;
                                }
                            }
                        }
                    }

                    bool fully_correct = (dims_correct && correct_pixels == total_pixels);
                    double pixel_acc = (total_pixels > 0) ?
                        (100.0 * correct_pixels / total_pixels) : 0.0;

                    total_tasks++;
                    if (dims_correct) tasks_with_correct_dims++;
                    if (fully_correct) tasks_fully_correct++;
                    if (total_pixels > 0) total_pixel_accuracy += pixel_acc;

                    cout << "    Dimensions: " << (dims_correct ? "OK" : "KO") << "\n";
                    cout << "    Pixel accuracy: " << std::fixed << std::setprecision(1)
                        << pixel_acc << "% (" << correct_pixels << "/" << total_pixels << ")\n";
                    cout << "    Fully correct: " << (fully_correct ? "YES" : "NO") << "\n";

                    if (verbose || !fully_correct) {
                        cout << "    Generated grid:\n";
                        for (long r = 0; r < std::min(10L, generated.nr()); ++r) {
                            cout << "      ";
                            for (long c = 0; c < std::min(15L, generated.nc()); ++c) {
                                cout << static_cast<int>(generated(r, c)) << " ";
                            }
                            if (generated.nc() > 15) cout << "...";
                            cout << "\n";
                        }
                        if (generated.nr() > 10) cout << "      ...\n";
                    }
                }
                cout << "\n";
                if (g_terminate_flag.load()) break;
            }

            // Final statistics
            cout << "=== EVALUATION SUMMARY ===\n";
            cout << "Tasks evaluated: " << total_tasks << "\n";
            if (generation_failures > 0) {
                cout << "Generation failures: " << generation_failures << "/" << total_tasks << "\n";
            }
            cout << "Correct dimensions: " << tasks_with_correct_dims << "/" << total_tasks
                << " (" << (100.0 * tasks_with_correct_dims / total_tasks) << "%)\n";
            cout << "Fully correct outputs: " << tasks_fully_correct << "/" << total_tasks
                << " (" << (100.0 * tasks_fully_correct / total_tasks) << "%)\n";
            cout << "Average pixel accuracy: "
                << (total_pixel_accuracy / total_tasks) << "%\n";
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}