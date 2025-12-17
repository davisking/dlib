/*!
    @file slm_mixture_of_experts_ex.cpp
    @brief Transformer with Mixture-of-Experts language model training and generation

    This program demonstrates how to build a transformer-based language model enhanced
    with Mixture-of-Experts (MoE) layers using Dlib's advanced deep learning capabilities.
    The example shows how to integrate the new moe_ffn layer that replaces
    standard feed-forward networks with dynamic expert routing for improved model capacity
    and specialization.

    Key features:
    - Mixture-of-Experts architecture with dynamic expert selection
    - Sparse activation pattern (only top-n experts active per input)
    - Automatic load balancing across experts through auxiliary loss
    - Multi-head self-attention with causal masking for autoregressive generation
    - BPE tokenization for efficient vocabulary management
    - Complete training and generation pipeline using datasets

    Usage modes:
    --train      Train model on internal datasets with MoE layers
    --generate   Generate text from trained MoE-enhanced model

    Performance considerations:
    - Sparse activation reduces inference compute (only top-n experts active)
    - Training is more challenging than standard transformers
    - Requires larger datasets for effective expert specialization
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

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/tokenizer/bpe_tokenizer.h>

// Include internal dataset
#include "slm_data.h"

using namespace std;
using namespace dlib;

namespace dlib
{
    // Expert network architecture for MoE layer
    template <template <typename> class DO, long d_model>
    using expert_net_type = swiglu<DO, d_model, input_tensor>;

    /*!
        Complete transformer block with MoE-based feed-forward layer.
        Architecture:
        1. Multi-head self-attention (from canonical_transformer)
        2. MoE feed-forward layer with multiple expert networks

        This replaces the standard transformer feed-forward layer with a
        mixture-of-experts that can specialize different experts for different
        types of patterns in the input.
    !*/
    template <template <typename> class ACT, template <typename> class DO,
        long seq_len, long d_model, long num_heads, typename MODE, typename SUBNET>
    using trans_moe_block =
        moe_ffn<expert_net_type<DO, d_model>, 4, 0, MODE, DO,
        add_prev1<multihead_attention<ACT, DO, seq_len, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>;

    /*!
        Classification head for next-token prediction.
    !*/
    template <long num_logits, typename SUBNET>
    using classification_head = loss_cross_entropy_per_logit<linear<num_logits, rms_norm<SUBNET>>>;

    // Core model parameters
    template<
        long vocab_size = 15000,
        long num_layers = 6,
        long num_heads = 8,
        long embedding_dim = 512,
        long max_seq_len = 300,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
        struct transformer_config {        
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;
        static constexpr long MAX_SEQ_LEN = max_seq_len;

        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        // Network component definitions for training (with dropout)
        template <typename SUBNET>
        using t_transformer_block =
            trans_moe_block<activation_func, dropout_policy, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS,
            training_mode_tag, SUBNET>;

        // Network component definitions for inference (using multiply)
        template <typename SUBNET>
        using i_transformer_block =
            trans_moe_block<activation_func, multiply, MAX_SEQ_LEN, EMBEDDING_DIM, NUM_HEADS,
            inference_mode_tag, SUBNET>;

        // Complete network type selector based on training/inference mode
        template<bool is_training>
        using network_type = std::conditional_t<is_training,
            classification_head<VOCAB_SIZE,
            repeat<NUM_LAYERS, t_transformer_block,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head<VOCAB_SIZE,
            repeat<NUM_LAYERS, i_transformer_block,
            embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>;

        struct model_info {
            static std::string describe() {
                std::stringstream ss;
                ss << "Transformer-MoE model configuration:\n"
                    << "- Vocabulary size: " << VOCAB_SIZE << "\n"
                    << "- Layers: " << NUM_LAYERS << "\n"
                    << "- Attention heads: " << NUM_HEADS << "\n"
                    << "- Embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- Sequence length: " << MAX_SEQ_LEN << "\n"
                    << "- Architecture: Transformer with MoE feed-forward layers\n"
                    << "- Experts per layer: 4 (auto top-n selection)";
                return ss.str();
            }
        };
    };
}

// Signal handling for clean termination
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

    void setup_interrupt_handler() {
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    }
#else
    // Unix/Linux-specific handler
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up and closing the program..." << endl;
        }
    }

    void setup_interrupt_handler() {
        std::signal(SIGINT, signal_handler);
    }
#endif
}

// Utility functions

bool load_tokens_from_file(std::vector<int>& tokens, const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    uint64_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    if (!file.good()) return false;

    tokens.clear();
    tokens.reserve(num_tokens);

    for (uint64_t i = 0; i < num_tokens; ++i) {
        uint32_t t;
        file.read(reinterpret_cast<char*>(&t), sizeof(t));
        if (!file.good()) return false;
        tokens.push_back(static_cast<int>(t));
    }

    return true;
}

// ----------------------------------------------------------------------------------------

// Structure to hold MoE parameter information with breakdown
// by component and computation mode
struct moe_param_info
{
    size_t expert_params;           // Parameters per single expert
    size_t other_params;            // Non-MoE layers (embeddings, attention, etc.)
    size_t total_training_params;   // Total parameters during training
    size_t total_inference_params;  // Active parameters during inference
    long num_experts;               // Number of experts per MoE layer
    long num_moe_layers;            // Number of MoE layers in the network
    long top_n;                     // Number of active experts during inference
    float efficiency_ratio;         // Ratio of inference/training params
    std::vector<float> expert_usage; // Usage statistics per expert

    void print() const
    {
        std::cout << "=== MoE network parameter analysis ===\n"
            << "Architecture:\n"
            << "  MoE layers: " << num_moe_layers << "\n"
            << "  Experts per layer: " << num_experts << "\n"
            << "  Active experts (top-n): " << top_n << "\n\n"
            << "Parameter breakdown per MoE layer:\n"
            << "  Single expert: " << expert_params << " params\n"
            << "Total network parameters:\n"
            << "  Other layers (attn, embed, etc.): " << other_params << " params\n"
            << "  Training (all experts): " << total_training_params << " params\n"
            << "  Inference (top-n experts): " << total_inference_params << " params\n\n"
            << "Efficiency:\n"
            << "  Inference uses " << (efficiency_ratio * 100.0f) << "% of training params\n"
            << "  Savings: " << ((1.0f - efficiency_ratio) * 100.0f) << "% fewer active params\n\n";

        // Display expert usage statistics
        if (!expert_usage.empty()) {
            std::cout << "Expert usage statistics (EMA):\n";

            // Calculate statistics
            float total_usage = 0.0f;
            float min_usage = expert_usage[0];
            float max_usage = expert_usage[0];

            for (float u : expert_usage) {
                total_usage += u;
                min_usage = std::min(min_usage, u);
                max_usage = std::max(max_usage, u);
            }

            float mean_usage = total_usage / num_experts;
            float ideal_usage = 1.0f / num_experts;

            // Calculate variance for coefficient of variation
            float variance = 0.0f;
            for (float u : expert_usage) {
                float diff = u - mean_usage;
                variance += diff * diff;
            }
            variance /= num_experts;
            float std_dev = std::sqrt(variance);
            float cv = (mean_usage > 1e-8f) ? (std_dev / mean_usage) : 0.0f;

            std::cout << "  Mean usage: " << std::fixed << std::setprecision(4)
                << mean_usage << " (ideal: " << ideal_usage << ")\n";
            std::cout << "  Range: [" << min_usage << ", " << max_usage << "]\n";
            std::cout << "  Std dev: " << std_dev << "\n";
            std::cout << "  Coefficient of variation: " << cv << "\n";

            // Balance quality assessment
            std::cout << "  Balance quality: ";
            if (cv < 0.3f)
                std::cout << "excellent (CV < 0.3)\n";
            else if (cv < 0.5f)
                std::cout << "good (CV < 0.5)\n";
            else if (cv < 0.8f)
                std::cout << "fair (CV < 0.8)\n";
            else
                std::cout << "poor (CV >= 0.8) - possible expert collapse\n";

            std::cout << "\n  Per-expert usage:\n";
            for (long e = 0; e < num_experts; ++e) {
                std::cout << "    expert " << e << ": "
                    << std::fixed << std::setprecision(4) << expert_usage[e];

                // Visual bar indicator
                int bar_length = static_cast<int>(expert_usage[e] * 50.0f / max_usage);
                std::cout << " [";
                for (int i = 0; i < bar_length; ++i)
                    std::cout << "=";
                for (int i = bar_length; i < 20; ++i)
                    std::cout << " ";
                std::cout << "]";

                // Flag over/under utilized experts
                float usage_ratio = expert_usage[e] / ideal_usage;
                if (usage_ratio < 0.5f)
                    std::cout << " (underutilized)";
                else if (usage_ratio > 2.0f)
                    std::cout << " (overutilized)";

                std::cout << "\n";
            }
        }
        else {
            std::cout << "Expert usage statistics: Not available (inference mode or no training yet)\n";
        }

        std::cout << "\n";
    }
};

template <typename net_type>
moe_param_info get_moe_param_info(const net_type& net, long num_layers)
{
    moe_param_info info;

    // Access first MoE layer
    const auto& moe_layer = layer<4>(net).layer_details();

    // Get MoE configuration
    info.num_experts = moe_layer.num_experts();
    info.num_moe_layers = num_layers;

    // Count parameters in one expert network
    if (info.num_experts > 0) {
        info.expert_params = count_parameters(moe_layer.get_expert(0));

        // Determine top_k (either fixed or auto-calculated as 20% of experts)
        info.top_n = std::max(1L, static_cast<long>(std::floor(info.num_experts * 0.2f)));
    }
    else {
        info.expert_params = 0;
        info.top_n = 0;
    }

    // Count other parameters (embeddings, attention layers, output layer)
    info.other_params = count_parameters(net);

    // Calculate total parameters for training (all experts in all MoE layers)
    size_t moe_training_params = info.num_moe_layers *
        (info.num_experts * info.expert_params);
    info.total_training_params = info.other_params + moe_training_params;

    // Calculate active parameters during inference (only top-n experts)
    size_t moe_inference_params = info.num_moe_layers *
        (info.top_n * info.expert_params);
    info.total_inference_params = info.other_params + moe_inference_params;

    // Calculate efficiency ratio
    if (info.total_training_params > 0) {
        info.efficiency_ratio = static_cast<float>(info.total_inference_params) /
            static_cast<float>(info.total_training_params);
    }
    else {
        info.efficiency_ratio = 1.0f;
    }

    // Retrieve expert usage statistics
    info.expert_usage = moe_layer.get_expert_usage();

    return info;
}

// Reads entire file content into a string.
std::string read_file_content(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        cerr << "Warning: Cannot open file: " << filepath << "\n";
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Replaces all occurrences of double newlines ("\n\n") with "@@" delimiter.
std::string normalize_paragraph_delimiters(const std::string& text)
{
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        // Check for double (or more) newlines
        if (i + 1 < text.size() && text[i] == '\n' && text[i + 1] == '\n') {
            result += "@@";
            i += 2;

            // Skip any additional consecutive newlines
            while (i < text.size() && text[i] == '\n') ++i;
        }
        else {
            result += text[i];
            ++i;
        }
    }

    return result;
}

// Recursively collects all text files from a directory using Dlib's directory class.
void collect_text_files_recursive(
    const directory& dir,
    std::vector<std::string>& text_files,
    size_t max_files = 0
)
{
    // Process files in current directory
    for (const auto& file : dir.get_files()) {
        if (max_files > 0 && text_files.size() >= max_files) return;

        // Check if it's a text file using file type detection
        file_content_type content_type;
        if (detect_file_type(file.full_name(), content_type)) {
            text_files.push_back(file.full_name());
            cout << "  Found text file: " << file.name() << "\n";
        }
    }

    // Recursively process subdirectories
    for (const auto& subdir : dir.get_dirs()) {
        if (max_files > 0 && text_files.size() >= max_files) {
            return;
        }
        collect_text_files_recursive(subdir, text_files, max_files);
    }
}

// Loads external text data from a file or directory
std::string load_external_data(
    const std::string& path,
    bool normalize_delimiters = true
)
{
    std::string combined_text;

    try {
        // Try as directory first
        directory dir(path);

        cout << "Scanning directory recursively: " << path << "\n";

        std::vector<std::string> text_files;
        collect_text_files_recursive(dir, text_files);

        cout << "Found " << text_files.size() << " text file(s)\n";

        if (text_files.empty()) {
            cerr << "Warning: No text files found in directory\n";
            return "";
        }

        // Sort files for consistent ordering
        std::sort(text_files.begin(), text_files.end());

        // Concatenate all files with delimiter
        size_t total_bytes = 0;
        for (const auto& filepath : text_files) {
            std::string content = read_file_content(filepath);
            if (!content.empty()) {
                combined_text += content;

                // Ensure content ends with delimiter for next file
                if (!combined_text.empty() &&
                    combined_text.size() >= 2 &&
                    combined_text.substr(combined_text.size() - 2) != "@@") {
                    combined_text += "@@";
                }

                total_bytes += content.size();
            }
        }

        cout << "Total loaded: " << total_bytes << " bytes from "
            << text_files.size() << " file(s)\n";
    }
    catch (const directory::dir_not_found&) {
        // Not a directory, try as single file
        cout << "Loading single text file: " << path << "\n";

        // Verify it's a text file
        file_content_type content_type;
        if (!detect_file_type(path, content_type)) {
            cerr << "Error: File does not appear to be text: " << path << "\n";
            cerr << "Only plain text files are supported for training.\n";
            return "";
        }

        combined_text = read_file_content(path);

        if (combined_text.empty()) {
            cerr << "Warning: File is empty or could not be read\n";
            return "";
        }

        cout << "Loaded " << combined_text.size() << " bytes from file\n";
    }
    catch (const std::exception& e) {
        cerr << "Error loading external data: " << e.what() << "\n";
        return "";
    }

    // Normalize paragraph delimiters if requested
    if (normalize_delimiters && !combined_text.empty())
        combined_text = normalize_paragraph_delimiters(combined_text);

    return combined_text;
}

// Parses text with @@ delimiters into individual segments.
std::vector<std::string> parse_delimited_segments(const std::string& text)
{
    std::vector<std::string> segments;
    std::string delimiter = "@@";

    size_t start = 0;
    size_t end = text.find(delimiter);

    while (end != std::string::npos) {
        std::string segment = text.substr(start, end - start);

        // Trim whitespace
        size_t first = segment.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = segment.find_last_not_of(" \t\n\r");
            segment = segment.substr(first, last - first + 1);

            // Add non-empty segments
            if (!segment.empty()) {
                segments.push_back(segment);
            }
        }

        start = end + delimiter.length();
        end = text.find(delimiter, start);
    }

    // Handle last segment
    if (start < text.size()) {
        std::string segment = text.substr(start);
        size_t first = segment.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = segment.find_last_not_of(" \t\n\r");
            segment = segment.substr(first, last - first + 1);
            if (!segment.empty()) {
                segments.push_back(segment);
            }
        }
    }

    return segments;
}

int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        setup_interrupt_handler();

        command_line_parser parser;
        parser.add_option("train", "Train a transformer model on internal datasets");
        parser.add_option("generate", "Generate text from a previously trained model");
        parser.add_option("learning-rate", "Set the learning rate (default: 3e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 96)", 1);
        parser.add_option("patience", "Iterations without progress before early stopping (default: 25000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 500)", 1);
        parser.add_option("weight-decay", "Set the weight decay for AdamW (default: 0.01)", 1);
        parser.add_option("beta1", "Set AdamW's beta1 coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set AdamW's beta2 coefficient (default: 0.999)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_lm_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: dlib_lm_tokenizer.vocab)", 1);
        parser.add_option("output-file", "Path for generated output (default: generated_text.txt)", 1);
        parser.add_option("external-data", "Path to external text data", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("train") && !parser.option("generate"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 3e-4);
        const size_t batch_size = get_option(parser, "batch-size", 96);
        const long patience = get_option(parser, "patience", 25000);
        const size_t max_epochs = get_option(parser, "max-epochs", 500);
        const double weight_decay = get_option(parser, "weight-decay", 0.01);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.999);
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_moe_model.dat");
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", "dlib_lm_tokenizer.vocab");
        const std::string output_file = get_option(parser, "output-file", "generated_text.txt");

        // Model architecture parameters
        const long num_tokens = 2000;
        const long num_layers = 3;
        const long num_heads = 6;
        const long embedding_dim = 192;
        const long max_seq_len = 128;

        // Define transformer configuration with MoE
        using my_transformer = transformer_config<
            num_tokens,     // vocab_size
            num_layers,     // number of layers
            num_heads,      // number of attention heads
            embedding_dim,  // embedding dimension
            max_seq_len     // maximum sequence length
        > ;

        // Load internal dataset
        cout << "Loading internal training datasets...\n";
        std::vector<dataset_id> text_datasets = {
            dataset_id::BLACK_HOLE_ARTICLE,
            dataset_id::PHYSICS_PARAGRAPHS,
			dataset_id::GENERAL_KNOWLEDGE
        };
        auto text_segments = get_dataset_as_segments(text_datasets);

        // Load external data if provided
        std::string external_corpus_for_tokenizer;
        if (parser.option("external-data")) {
            std::string external_path = parser.option("external-data").argument();

            std::string external_text = load_external_data(external_path, true);
            if (!external_text.empty()) {
                // Store raw text for tokenizer training (if needed later)
                external_corpus_for_tokenizer = external_text;

                // Parse into segments for training
                cout << "Parsing external data into segments...\n";
                auto external_segments = parse_delimited_segments(external_text);
                cout << "Parsed " << external_segments.size() << " external segments\n";

                if (!external_segments.empty()) {
                    // Add to training data
                    size_t original_count = text_segments.size();
                    text_segments.insert(text_segments.end(),
                        external_segments.begin(), external_segments.end());

                    cout << "Training segments: " << original_count
                        << " (internal) + " << external_segments.size()
                        << " (external) = " << text_segments.size() << " (total)\n";
                }
            }
            else {
                cerr << "Warning: no valid external data loaded, continuing with internal datasets only\n";
            }
        }

        // Tokens filename
        const std::string tokens_file = "dlib_datasets_tokens.bin";

        // Tokenizer BPE
        bpe_tokenizer tokenizer;

        // Load pre-trained tokenizer if it exists
        if (file_exists(tokenizer_file)) {
            cout << "Loading pre-trained tokenizer from: " << tokenizer_file << endl;
            deserialize(tokenizer_file) >> tokenizer;
            cout << "Tokenizer loaded successfully with vocabulary size: " << tokenizer.get_vocab_size() << endl;
        }
        else {
            cout << "Pre-trained tokenizer not found at: " << tokenizer_file << endl;
            cout << "Will train a new tokenizer if needed." << endl;
        }

        // For GPU usage (if available)
        std::vector<int> gpus{ 0 };

        // Variables to store tokens (one vector per segment)
        std::vector<std::vector<int>> full_tokens;

        // Training mode
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";            

            // Check if we should load pre-tokenized tokens
            bool tokens_loaded = false;
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens from file...\n";
                try {
                    dlib::deserialize(tokens_file) >> full_tokens;

                    // Calculate total tokens across all segments
                    size_t total_tokens = 0;
                    for (const auto& segment_tokens : full_tokens)
                        total_tokens += segment_tokens.size();

                    cout << "Loaded " << full_tokens.size() << " segments ("
                        << total_tokens << " tokens) from file\n";
                    tokens_loaded = true;
                }
                catch (const std::exception& e) {
                    cerr << "Failed to load tokens from file: " << e.what()
                        << "\nWill tokenize again.\n";
                    full_tokens.clear();
                }
            }

            if (!tokens_loaded) {
                // Train a new tokenizer if needed
                if (!file_exists(tokenizer_file)) {
                    cout << "Training new BPE tokenizer with vocabulary size " << num_tokens << "...\n";

                    // Compose training corpus from multiple datasets
                    std::string delimiter = "@@";
                    std::string tokenizer_corpus =
                        get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE) + delimiter
                        + get_dataset_as_text(dataset_id::PHYSICS_PARAGRAPHS) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTA) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTB) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTC) + delimiter
                        + get_dataset_as_text(dataset_id::GENERAL_KNOWLEDGE);

                    if (!external_corpus_for_tokenizer.empty())
                        tokenizer_corpus += delimiter + external_corpus_for_tokenizer;
                    cout << "Tokenizer corpus: " << tokenizer_corpus.size() << " characters\n";

                    // Replace all "@@" delimiters with spaces
                    size_t pos = 0;
                    while ((pos = tokenizer_corpus.find(delimiter, pos)) != std::string::npos) {
                        tokenizer_corpus.replace(pos, delimiter.length(), " ");
                        pos += 1; // Move past the replacement space
                    }

                    tokenizer.train(tokenizer_corpus, num_tokens, 1e6, true);
                    serialize(tokenizer_file) << tokenizer;
                    cout << "Tokenizer saved to " << tokenizer_file << endl;
                }

                // Tokenize all text segments
                cout << "Tokenizing input text segments...\n";
                int text_start_id = tokenizer.get_special_token_id("<text>"),
                    text_end_id = tokenizer.get_special_token_id("</text>");
                if (text_start_id < 0 || text_end_id < 0) {
                    cerr << "ERROR: Required special tokens not found in tokenizer vocabulary!\n";
                    cerr << "The tokenizer must include: <text>, </text>\n";
                    return 1;
                }

                auto start_time = std::chrono::high_resolution_clock::now();
                full_tokens.clear();

                // Format : <text>content</text>
                size_t total_tokens = 0;
                for (const auto& segment : text_segments) {
                    std::vector<int> segment_tokens;
                    segment_tokens.push_back(text_start_id);
                    auto encoded_tokens = tokenizer.encode(segment);
                    segment_tokens.insert(segment_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
                    segment_tokens.push_back(text_end_id);

                    total_tokens += segment_tokens.size();
                    full_tokens.push_back(std::move(segment_tokens));
                }

                auto end_time = std::chrono::high_resolution_clock::now();
                auto tokenize_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
                cout << "Tokenization complete: " << total_tokens << " tokens in " << tokenize_time << "s.\n";
                text_segments.clear();

                // Save tokens for future use using Dlib serialization
                cout << "Saving tokens to file: " << tokens_file << endl;
                try {
                    serialize(tokens_file) << full_tokens;
                    cout << "Tokens successfully saved for future use.\n";
                }
                catch (const std::exception& e) {
                    cerr << "Warning: Failed to save tokens: " << e.what() << "\n";
                }
            }

            // Prepare training sequences (sliding window)
            cout << "Preparing training sequences...\n";
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            build_single_token_prediction_dataset(full_tokens, max_seq_len,
                tokenizer.get_special_token_id("<pad>"), false,
                samples, labels);
            cout << "Created " << samples.size() << " training samples\n";

            // Augment the dataset with 5% additional noisy samples
            augment_training_dataset(
                samples, labels,
                tokenizer.get_special_token_id("<unk>"),
                tokenizer.get_special_token_id("<pad>"),
                0.05
            );
            std::cout << "Augmented dataset size: " << samples.size() << std::endl;

            // Release memory as we no longer need the tokens at this point
            full_tokens.clear();            

            // Build and train the network
            using net_type = my_transformer::network_type<true>;
            net_type net;
            cout << my_transformer::model_info::describe() << endl;

            // Tokenizer stored with model for simplified inference
            if (file_exists(model_file) &&
                !file_exists("chkpt-" + model_file)) deserialize(model_file) >> net >> tokenizer;            

            // Create trainer
            dnn_trainer<net_type, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(5e-5);
            trainer.set_learning_rate_shrink_factor(0.1);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(5));
            trainer.be_quiet();
            cout << net << endl << endl; // Show the model architecture
            cout << "Starting training...\n";

            size_t epoch = 0, steps = 0;            
            size_t batches_count = 0, batches_seen = 0, samples_seen = 0;
            double total_loss = 0.0;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            // Training loop
            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() 
                && epoch < max_epochs && !g_terminate_flag.load())
            {
                total_loss = 0.0;
                batches_seen = 0, samples_seen = 0;
                epoch_start = std::chrono::high_resolution_clock::now();

                // Shuffle the dataset
                shuffle_training_dataset(samples, labels);

                for (size_t i = 0; i < samples.size() && !g_terminate_flag.load(); i += batch_size)
                {
                    size_t batch_end = std::min(i + batch_size, samples.size());
                    std::vector<matrix<int, 0, 1>> batch_samples(
                        samples.begin() + i, samples.begin() + batch_end);
                    std::vector<unsigned long> batch_labels(
                        labels.begin() + i, labels.begin() + batch_end);

                    trainer.train_one_step(batch_samples, batch_labels);
                    total_loss += trainer.get_average_loss();
                    batches_seen++;
                    samples_seen += batch_samples.size();
                    steps += batch_samples.size();

                    // Progress reporting
                    if (batches_count++ % 50 == 0) {
                        double avg_loss = total_loss / batches_seen;
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - epoch_start).count();
                        double samples_per_sec = samples_seen / (elapsed > 0 ? elapsed : 1);

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " (ksteps: " << (steps / 1000) << ")"
                            << " \t loss: " << avg_loss
                            << " \t patience: " << trainer.get_steps_without_progress()
                            << " \t speed: " << samples_per_sec << " samples/sec\n";
                        cout.flush();
                    }
                }
                epoch++;
            }

            // Save model and tokenizer
			cout << "Training complete. Saving model...\n";
            net.clean();
            serialize(model_file) << net << tokenizer;
            cout << "Model saved to " << model_file << "\n";

            // Evaluate on training set
            {
                cout << "Evaluating model accuracy...\n";
                my_transformer::network_type<false> g_infer;
                deserialize(model_file) >> g_infer >> tokenizer;
                auto predicted = g_infer(samples);
                size_t correct = 0;
                for (size_t i = 0; i < labels.size(); ++i)
                    if (predicted[i] == labels[i]) correct++;
                double accuracy = (double)correct / labels.size();
                cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";
            }
        }

        // Generation mode
        if (parser.option("generate"))
        {
            cout << "=== GENERATION MODE ===\n";

            // Load the model
            using net_infer = my_transformer::network_type<false>;
            net_infer net;
            if (file_exists(model_file)) {
                deserialize(model_file) >> net >> tokenizer;
                cout << "Loaded model from " << model_file << "\n";
            }
            else {
                cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }

            // Display model structure information
            auto param_info = get_moe_param_info<net_infer>(net, num_layers);
            param_info.print();

            // Check that tokenizer is loaded
            if (tokenizer.get_vocab_size() == 0) {
                cerr << "Error: Tokenizer not loaded. Please provide a valid tokenizer file.\n";
                return 0;
            }

            // Load tokenized segments
            std::vector<std::vector<int>> tokenized_segments;
            if (!file_exists(tokens_file)) {
                cerr << "Error: Tokenized file not found. Please run --train first.\n";
                return 0;
            }

            cout << "Loading tokenized segments from: " << tokens_file << endl;
            try {
                deserialize(tokens_file) >> tokenized_segments;
                cout << "Loaded " << tokenized_segments.size() << " tokenized segments.\n";
            }
            catch (const std::exception& e) {
                cerr << "Error loading tokens: " << e.what() << "\n";
                return 0;
            }

            if (tokenized_segments.empty()) {
                cerr << "Error: No segments found in tokens file.\n";
                return 0;
            }

            // Select a segment for generation
            dlib::rand rng(std::chrono::system_clock::now().time_since_epoch().count());
            size_t segment_idx = rng.get_random_32bit_number() % 100;
            cout << "Randomly selected segment #" << segment_idx << " (out of "
                << tokenized_segments.size() << ") for generation\n";
            const auto& selected_segment = tokenized_segments[segment_idx];

            if (selected_segment.size() < (size_t)max_seq_len) {
                cerr << "Error: Selected segment has only " << selected_segment.size()
                    << " tokens, need at least " << max_seq_len << ".\n";
                return 0;
            }

            // Extract prompt tokens (first max_seq_len tokens of the segment)
            std::vector<int> prompt_tokens(selected_segment.begin(),
                selected_segment.begin() + max_seq_len);
            cout << "Using " << prompt_tokens.size() << " tokens for initial prompt.\n";

            // Setup inference context
            inference_context llm_context(max_seq_len, 4, tokenizer.get_special_token_id("<pad>"));
            llm_context.add_tokens(prompt_tokens);
            auto input_seq = llm_context.get_input_window();

            // Open output file
            std::ofstream outfile(output_file, std::ios::binary);
            if (!outfile) {
                cerr << "Error: Cannot open output file: " << output_file << "\n";
                return 0;
            }

            // Write initial text (corresponding to prompt tokens)
            std::string initial_text = tokenizer.decode(prompt_tokens, false);
            outfile.write(initial_text.c_str(), initial_text.size());
            outfile.flush();

            cout << "Starting autoregressive generation...\n";

            // Generation parameters
            const size_t tokens_to_generate = selected_segment.size() - max_seq_len;
            std::vector<int> generated_tokens;
            generated_tokens.reserve(tokens_to_generate);

            auto start_time = std::chrono::high_resolution_clock::now();
            int end_of_text_id = tokenizer.get_special_token_id("</text>");

            // Generate tokens autoregressively
            for (size_t i = 0; i < tokens_to_generate && !g_terminate_flag.load(); ++i) {
                // Predict next token
                int next_token = net(input_seq);
                generated_tokens.push_back(next_token);

                // Update context window
                llm_context.add_token(next_token);
                input_seq = llm_context.get_input_window();

                // Progress reporting every 50 tokens
                if ((i + 1) % 50 == 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    double tokens_per_sec = (i + 1) / (elapsed > 0 ? elapsed : 1);

                    cout << "Generated " << (i + 1) << "/" << tokens_to_generate
                        << " tokens (" << ((i + 1) * 100.0 / tokens_to_generate) << "%) - "
                        << tokens_per_sec << " tokens/sec\r";
                    cout.flush();
                }

                // Stop if end-of-text token is generated
                if (next_token == end_of_text_id) break;
            }

            // Write generated text to file
            std::string generated_text = tokenizer.decode(generated_tokens, false);
            outfile.write(generated_text.c_str(), generated_text.size());
            outfile.flush();
            outfile.close();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();

            cout << "\nGeneration complete in " << total_time << " seconds!\n";
            cout << "Generated " << generated_tokens.size() << " tokens\n";
            cout << "Total output: " << (initial_text.size() + generated_text.size()) << " bytes\n";
            cout << "Output saved to " << output_file << "\n";

            // Compare generated text with original segment for validation
            cout << "\n=== Validation: comparing generated vs. original segment ===\n";

            // Extract reference tokens (the part we tried to regenerate)
            std::vector<int> reference_tokens(selected_segment.begin() + max_seq_len,
                selected_segment.end());

            // Limit comparison to the length of generated tokens
            size_t compare_length = std::min(reference_tokens.size(), generated_tokens.size());
            std::vector<int> reference_subset(reference_tokens.begin(),
                reference_tokens.begin() + compare_length);
            std::vector<int> generated_subset(generated_tokens.begin(),
                generated_tokens.begin() + compare_length);

            cout << "Comparing " << compare_length << " tokens\n";
            cout << "Reference length: " << reference_tokens.size() << " tokens\n";
            cout << "Generated length: " << generated_tokens.size() << " tokens\n\n";

            // Compute and display similarity metrics
            auto similarity = compute_text_similarity(reference_subset, generated_subset);
            similarity.print();

            // Display sample of differences if similarity is not perfect
            if (similarity.edit_similarity < 0.95) {
                cout << "Sample comparison (first 100 tokens):\n";
                size_t sample_len = std::min(size_t(100), compare_length);

                size_t diff_count = 0;
                for (size_t i = 0; i < sample_len; ++i) {
                    if (reference_subset[i] != generated_subset[i]) {
                        if (diff_count < 10) {  // Show first 10 differences
                            std::string ref_word = tokenizer.decode({ reference_subset[i] }, false);
                            std::string gen_word = tokenizer.decode({ generated_subset[i] }, false);
                            cout << "  Position " << i << ": '"
                                << ref_word << "' -> '" << gen_word << "'\n";
                        }
                        diff_count++;
                    }
                }
                cout << "Total differences in sample: " << diff_count << "/" << sample_len << "\n";
            }
            else {
                cout << "Excellent match! Generated text closely follows the original.\n";
            }
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}

/*
 * This program demonstrates production-grade language model training using Dlib's
 * advanced utilities for dataset preparation: shuffle_training_dataset() for
 * randomization and augment_training_dataset() for noise injection. These techniques
 * improve model robustness and generalization, enabling effective training on large
 * volumes of information.
 *
 * - Transformer model configuration:
 *    + vocabulary size: 3500
 *    + layers: 4
 *    + attention heads: 6
 *    + embedding dimension: 228
 *    + max sequence length: 100
 * - Number of parameters: 5,970,554 (training) - 5,432,738 (inference)
 *
 * After training, the model achieves excellent memorization of all internal datasets.
 */