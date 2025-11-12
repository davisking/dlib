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
    - Sparse activation pattern (only top-k experts active per input)
    - Automatic load balancing across experts through auxiliary loss
    - Multi-head self-attention with causal masking for autoregressive generation
    - BPE tokenization for efficient vocabulary management
    - Complete training, generation, and verification pipeline
    - Compile-time mode selection (training vs inference) using tag-based dispatch

    Usage modes:
    --train      Train model on internal dataset with MoE layers
    --generate   Generate text from trained MoE-enhanced model
    --verify     Compare generated output with original dataset

    Configuration:
    - Adjust template parameters in transformer_config for model architecture
    - Modify training parameters via command-line for optimization
    - Set sequence length and expert count according to hardware resources
    - Balance exploration noise and load balancing weights for convergence

    Performance considerations:
    - MoE layers increase model capacity without proportional compute cost
    - Sparse activation reduces inference time compared to dense networks
    - Expert specialization improves model quality on diverse data
    - Memory usage scales with number of experts (4 experts by default)
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
    using expert_net_type = std_ffn<silu, DO, d_model, input<matrix<float>>>;

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
        multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;

    /*!
        Classification head for next-token prediction.
        Uses the new loss_cross_entropy_per_logit loss function which:
        - Works directly with sequence outputs (no flattening needed)
        - Computes loss only on the last sequence position
        - Optimized for autoregressive language modeling
    !*/
    template <long num_logits, long embedding_dim, typename SUBNET>
    using classification_head = loss_cross_entropy_per_logit<linear<num_logits,
        linear<embedding_dim / 2, rms_norm<SUBNET>>>>;

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
            classification_head<VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, t_transformer_block,
            token_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>,
            classification_head<VOCAB_SIZE, EMBEDDING_DIM,
            repeat<NUM_LAYERS, i_transformer_block,
            token_embeddings<VOCAB_SIZE, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>;

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
                    << "- Experts per layer: 4 (auto top-k selection)";
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
std::string generate_tokens_filename(size_t max_bytes)
{
    if (max_bytes > 0) {
        return "dlib_dataset_" + std::to_string(max_bytes) + "_tokens.bin";
    }
    return "dlib_dataset_tokens.bin";
}

bool save_tokens_to_file(const std::vector<int>& tokens, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    uint64_t num_tokens = tokens.size();
    file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));

    for (int token : tokens) {
        uint32_t t = static_cast<uint32_t>(token);
        file.write(reinterpret_cast<const char*>(&t), sizeof(t));
    }

    return file.good();
}

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

std::string read_file_content(const std::string& filename, size_t max_bytes = 0)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string content;
    if (max_bytes > 0) {
        content.resize(max_bytes);
        file.read(&content[0], max_bytes);
        content.resize(file.gcount());
    }
    else {
        content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    return content;
}

bool verify_match(const std::string& original, const std::string& generated)
{
    if (original.size() != generated.size()) {
        cout << "Size mismatch: original=" << original.size()
            << ", generated=" << generated.size() << "\n";
        return false;
    }

    size_t mismatch_count = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != generated[i]) {
            if (mismatch_count < 10) {
                cout << "Mismatch at byte " << i << ": expected='" << original[i]
                    << "' (0x" << std::hex << (int)(unsigned char)original[i] << std::dec
                    << "), got='" << generated[i]
                    << "' (0x" << std::hex << (int)(unsigned char)generated[i] << std::dec << ")\n";
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0) {
        cout << "Total mismatches: " << mismatch_count << "\n";
        return false;
    }

    cout << "Files match perfectly. All " << original.size() << " bytes are identical.\n";
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
    long top_k;                     // Number of active experts during inference
    float efficiency_ratio;         // Ratio of inference/training params

    void print() const
    {
        std::cout << "=== MoE network parameter analysis ===\n"
            << "Architecture:\n"
            << "  MoE layers: " << num_moe_layers << "\n"
            << "  Experts per layer: " << num_experts << "\n"
            << "  Active experts (top-k): " << top_k << "\n\n"
            << "Parameter breakdown per MoE layer:\n"
            << "  Single expert: " << expert_params << " params\n"
            << "Total network parameters:\n"
            << "  Other layers (attn, embed, etc.): " << other_params << " params\n"
            << "  Training (all experts): " << total_training_params << " params\n"
            << "  Inference (top-k experts): " << total_inference_params << " params\n\n"
            << "Efficiency:\n"
            << "  Inference uses " << (efficiency_ratio * 100.0f) << "% of training params\n"
            << "  Savings: " << ((1.0f - efficiency_ratio) * 100.0f) << "% fewer active params\n\n";
    }
};

// Computes detailed parameter counts for MoE-enhanced networks
template <typename net_type>
moe_param_info get_moe_param_info(const net_type& net, long num_layers)
{
    moe_param_info info;

    // Access first MoE layer
    const auto& moe_layer = layer<5>(net).layer_details();

    // Get MoE configuration
    info.num_experts = moe_layer.num_experts();
    info.num_moe_layers = num_layers;

    // Count parameters in one expert network
    if (info.num_experts > 0) {
        info.expert_params = count_parameters(moe_layer.get_expert(0));

        // Determine top_k (either fixed or auto-calculated as 20% of experts)
        info.top_k = std::max(1L, static_cast<long>(std::floor(info.num_experts * 0.2f)));
    }
    else {
        info.expert_params = 0;
        info.top_k = 0;
    }

    // Count other parameters (embeddings, attention layers, output layer)
    info.other_params = count_parameters(net);

    // Calculate total parameters for training (all experts in all MoE layers)
    size_t moe_training_params = info.num_moe_layers *
        (info.num_experts * info.expert_params);
    info.total_training_params = info.other_params + moe_training_params;

    // Calculate active parameters during inference (only top-k experts)
    size_t moe_inference_params = info.num_moe_layers *
        (info.top_k * info.expert_params);
    info.total_inference_params = info.other_params + moe_inference_params;

    // Calculate efficiency ratio
    if (info.total_training_params > 0) {
        info.efficiency_ratio = static_cast<float>(info.total_inference_params) /
            static_cast<float>(info.total_training_params);
    }
    else {
        info.efficiency_ratio = 1.0f;
    }

    return info;
}

int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        setup_interrupt_handler();

        command_line_parser parser;
        parser.add_option("train", "Train a transformer model on internal dataset");
        parser.add_option("generate", "Generate text from a previously trained model");
        parser.add_option("verify", "Verify generated output against original dataset");
        parser.add_option("learning-rate", "Set the learning rate (default: 2e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 64)", 1);
        parser.add_option("patience", "Iterations without progress before early stopping (default: 7500)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 500)", 1);
        parser.add_option("alpha", "Set the weight decay for Adam (default: 0.004)", 1);
        parser.add_option("beta1", "Set Adam's first moment coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set Adam's second moment coefficient (default: 0.999)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_lm_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: slm_tokenizer.vocab)", 1);
        parser.add_option("output-file", "Path for generated output (default: generated_text.txt)", 1);
        parser.add_option("max-tokens", "Maximum number of tokens to process (default: all)", 1);
        parser.add_option("max-bytes", "Maximum number of bytes to process (default: all)", 1);
        parser.add_option("percent", "Percentage of bytes to process (0-100 - default: all)", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("train") && !parser.option("generate") &&
            !parser.option("verify"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 2e-4);
        const size_t batch_size = get_option(parser, "batch-size", 64);
        const long patience = get_option(parser, "patience", 7500);
        const size_t max_epochs = get_option(parser, "max-epochs", 500);
        const double alpha = get_option(parser, "alpha", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.999);
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_moe_model.dat");
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", "slm_tokenizer.vocab");
        const std::string output_file = get_option(parser, "output-file", "generated_text.txt");

        // Model architecture parameters
        const long num_tokens = 2500;
        const long num_layers = 4;
        const long num_heads = 6;
        const long embedding_dim = 228;
        const long max_seq_len = 60;

        // Define transformer configuration with MoE
        using my_transformer = transformer_config<
            num_tokens,     // vocab_size
            num_layers,     // number of layers
            num_heads,      // number of attention heads
            embedding_dim,  // embedding dimension
            max_seq_len     // maximum sequence length
        > ;

        // Load internal dataset
        cout << "Loading internal training dataset...\n";
        std::string training_text = get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE);
        size_t original_size = training_text.size();
        cout << "Loaded " << original_size << " bytes from internal dataset\n";

        // Calculate max bytes to process
        size_t max_bytes = 0, max_tokens_limit = 0;
        if (parser.option("max-tokens"))
            max_tokens_limit = std::stoul(parser.option("max-tokens").argument());
        if (parser.option("max-bytes")) {
            max_bytes = std::stoul(parser.option("max-bytes").argument());
        }
        else if (parser.option("percent")) {
            double percent = std::stod(parser.option("percent").argument());
            max_bytes = static_cast<size_t>(original_size * percent / 100.0);
            cout << "Processing " << percent << "% of dataset = " << max_bytes << " bytes\n";
        }

        // Apply size limits to dataset
        if (max_bytes > 0 && max_bytes < training_text.size()) {
            training_text.resize(max_bytes);
            cout << "Limited to " << training_text.size() << " bytes\n";
        }

        // Determine tokens filename
        const std::string tokens_file = generate_tokens_filename(max_bytes);

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

        // Variables to store tokens
        std::vector<int> full_tokens;

        // Training mode
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";

            bool tokens_loaded = false;

            // Check if we should load pre-tokenized tokens
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens from file...\n";
                if (load_tokens_from_file(full_tokens, tokens_file)) {
                    cout << "Loaded " << full_tokens.size() << " tokens from file.\n";
                    if (max_tokens_limit > 0 && max_tokens_limit < full_tokens.size()) {
                        full_tokens.resize(max_tokens_limit);
                        cout << "Limited to " << full_tokens.size() << " tokens for training.\n";
                    }
                    tokens_loaded = true;
                }
                else {
                    cerr << "Failed to load tokens from file. Will tokenize again.\n";
                }
            }

            if (!tokens_loaded) {
                // Train a new tokenizer if needed
                if (!file_exists(tokenizer_file)) {
                    cout << "Training new BPE tokenizer with vocabulary size " << num_tokens << "...\n";

                    // Compose training corpus from multiple datasets
                    std::string tokenizer_corpus = get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE) + " " +
                        get_dataset_raw(dataset_id::PHYSICS_PARAGRAPHS) + " " + get_dataset_raw(dataset_id::BLACK_HOLE_QA_PARTA) +
                        get_dataset_raw(dataset_id::BLACK_HOLE_QA_PARTB) + get_dataset_raw(dataset_id::BLACK_HOLE_QA_PARTC);

                    // Replace all "@@" delimiters with spaces
                    std::string delimiter = "@@";
                    size_t pos = 0;
                    while ((pos = tokenizer_corpus.find(delimiter, pos)) != std::string::npos) {
                        tokenizer_corpus.replace(pos, delimiter.length(), " ");
                        pos += 1; // Move past the replacement space
                    }

                    tokenizer.train(tokenizer_corpus, num_tokens, 1e6, true);
                    serialize(tokenizer_file) << tokenizer;
                    cout << "Tokenizer saved to " << tokenizer_file << endl;
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
                auto encoded_tokens = tokenizer.encode(training_text);
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

            build_single_token_prediction_dataset({ full_tokens }, max_seq_len,
                tokenizer.get_special_token_id("<pad>"), false,
                samples, labels);
            full_tokens.clear();
            cout << "Created " << samples.size() << " training samples\n";

            // Build and train the network
            using net_type = my_transformer::network_type<true>;
            net_type net;
            cout << my_transformer::model_info::describe() << endl;

            // Tokenizer stored with model for simplified inference
            if (file_exists(model_file)) deserialize(model_file) >> net >> tokenizer;
            cout << net << endl << endl; // Show the model architecture

            // Create trainer
            dnn_trainer<net_type, adam> trainer(net, adam(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.be_quiet();
            cout << "Starting training...\n";

            size_t epoch = 0, steps = 0;            
            size_t batches_count = 0, batches_seen = 0, samples_seen = 0;
            double total_loss = 0.0;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            // Training loop
            while (trainer.get_learning_rate() >= 1e-6 && epoch < max_epochs && !g_terminate_flag.load())
            {
                total_loss = 0.0;
                batches_seen = 0, samples_seen = 0;
                epoch_start = std::chrono::high_resolution_clock::now();

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

            // Save model
            net.clean();
            serialize(model_file) << net << tokenizer;
            cout << "Model saved to " << model_file << "\n";

            // Evaluate on training set
            {
                if (!g_terminate_flag.load()) {
                    cout << "Evaluating model accuracy...\n";
                    using net_infer = my_transformer::network_type<false>;
                    net_infer g_infer;
                    deserialize(model_file) >> g_infer >> tokenizer;
                    auto predicted = g_infer(samples);
                    size_t correct = 0;
                    for (size_t i = 0; i < labels.size(); ++i)
                        if (predicted[i] == labels[i]) correct++;
                    double accuracy = (double)correct / labels.size();
                    cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";

                    // We need perfect accuracy to reconstruct the internal dataset
                    if (accuracy < 0.999) {
                        cout << "WARNING: Model accuracy is less than 99.90%. The model may not "
                            << "perfectly reconstruct the input text.\n";
                    }
                }
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

            // Read beginning of the dataset for prompt
            std::vector<int> prompt_tokens;

            // Check if we have pre-tokenized tokens
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens for prompt...\n";

                std::ifstream file(tokens_file, std::ios::binary);
                if (!file) {
                    cerr << "Failed to open tokens file: " << tokens_file << endl;
                }
                else {
                    // Read total number of tokens
                    uint64_t num_tokens_in_file;
                    file.read(reinterpret_cast<char*>(&num_tokens_in_file), sizeof(num_tokens_in_file));

                    // Read only the first max_seq_len tokens
                    size_t tokens_to_read = std::min(static_cast<size_t>(max_seq_len),
                        static_cast<size_t>(num_tokens_in_file));
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
                cout << "Tokenizing initial prompt from internal dataset...\n";

                // Use beginning of internal dataset for prompt
                std::string prompt_text = training_text.substr(0, std::min(training_text.size(),
                    static_cast<size_t>(max_seq_len * 10)));

                int text_start_id = tokenizer.get_special_token_id("<text>");
                prompt_tokens.clear();
                prompt_tokens.push_back(text_start_id);
                auto encoded_tokens = tokenizer.encode(prompt_text);
                prompt_tokens.insert(prompt_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
            }

            // Limit to requested number of tokens
            if (prompt_tokens.size() > (size_t)max_seq_len) {
                prompt_tokens.resize(max_seq_len);
            }
            else if (prompt_tokens.size() < (size_t)max_seq_len) {
                cerr << "Warning: Not enough tokens in prompt. Got " << prompt_tokens.size()
                    << ", needed " << max_seq_len << ".\n";
                return 0;
            }
            cout << "Using " << prompt_tokens.size() << " tokens for initial prompt\n";

            // Put prompt in input sequence
            inference_context llm_context(max_seq_len, 4, tokenizer.get_special_token_id("<pad>"));
            llm_context.add_tokens(prompt_tokens);
            auto input_seq = llm_context.get_input_window();

            // Determine text size to generate
            size_t target_size = (max_bytes > 0) ? max_bytes : training_text.size();
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
            int end_of_text = tokenizer.get_special_token_id("</text>"), next_token = 0;
            while (total_bytes < target_size && next_token != end_of_text && !g_terminate_flag.load()) {
                // Predict next token
                next_token = net(input_seq);
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
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    double tokens_per_second = (token_count - input_seq.size()) / (elapsed > 0 ? elapsed : 1);

                    cout << "Generated " << (token_count - input_seq.size()) << " tokens, "
                        << total_bytes << " bytes ("
                        << (total_bytes * 100.0 / target_size) << "%) - "
                        << tokens_per_second << " tokens/sec - "
                        << "Est. completion: "
                        << (int)((target_size - total_bytes) / (tokens_per_second * (chunk.size() / (double)buffer_size)))
                        << " seconds\r";
                }
                if (max_tokens_limit > 0 && token_count >= max_tokens_limit) break;
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

            cout << "\nGeneration complete in " << total_time << " seconds! (100%)\n";
            cout << "Generated " << (token_count - input_seq.size()) << " tokens after prompt, "
                << total_bytes << " bytes total\n";
            cout << "Output saved to " << output_file << "\n";
        }

        // Verification mode - Compare original and generated file
        if (parser.option("verify"))
        {
            cout << "=== VERIFICATION MODE ===\n";

            if (!file_exists(output_file)) {
                cerr << "Error: Generated file not found at " << output_file << "\n";
                return 0;
            }

            // Read generated file
            cout << "Reading generated file...\n";
            std::string generated = read_file_content(output_file);

            // Read the same portion of original dataset
            cout << "Reading original dataset (same size as generated)...\n";
            std::string original = training_text.substr(0, std::min(training_text.size(), generated.size()));

            cout << "Verifying byte-for-byte match...\n";
            bool verify = verify_match(original, generated);

            if (verify)
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