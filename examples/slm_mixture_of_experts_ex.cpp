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
        multihead_attention<ACT, DO, seq_len, d_model, num_heads, SUBNET>>;

    /*!
        Classification head for next-token prediction.
        Uses the new loss_cross_entropy_per_logit loss function which:
        - Works directly with sequence outputs (no flattening needed)
        - Computes loss only on the last sequence position
        - Optimized for autoregressive language modeling
    !*/
    template <long num_logits, long embedding_dim, typename SUBNET>
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
    }
};

// Computes detailed parameter counts for MoE-enhanced networks
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

    return info;
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
        parser.add_option("learning-rate", "Set the learning rate (default: 2e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 64)", 1);
        parser.add_option("patience", "Iterations without progress before early stopping (default: 15000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 500)", 1);
        parser.add_option("alpha", "Set the weight decay for Adam (default: 0.004)", 1);
        parser.add_option("beta1", "Set Adam's first moment coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set Adam's second moment coefficient (default: 0.999)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_lm_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: dlib_lm_tokenizer.vocab)", 1);
        parser.add_option("output-file", "Path for generated output (default: generated_text.txt)", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("train") && !parser.option("generate"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 2e-4);
        const size_t batch_size = get_option(parser, "batch-size", 64);
        const long patience = get_option(parser, "patience", 15000);
        const size_t max_epochs = get_option(parser, "max-epochs", 500);
        const double alpha = get_option(parser, "alpha", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.999);
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_moe_model.dat");
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", "dlib_lm_tokenizer.vocab");
        const std::string output_file = get_option(parser, "output-file", "generated_text.txt");

        // Model architecture parameters
        const long num_tokens = 3500;
        const long num_layers = 6;
        const long num_heads = 8;
        const long embedding_dim = 256;
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
        std::vector<dataset_id> datasets = {
            dataset_id::BLACK_HOLE_ARTICLE,
            dataset_id::PHYSICS_PARAGRAPHS,
			dataset_id::GENERAL_KNOWLEDGE
        };
        auto training_segments = get_dataset_as_segments(datasets);

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
                        << total_tokens << " total tokens) from file.\n";
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
                if (text_start_id < 0 || text_end_id < 0)
                    cout << "Warning: Special tokens not found in tokenizer vocabulary.\n";

                auto start_time = std::chrono::high_resolution_clock::now();
                full_tokens.clear();

                // Process each segment independently with delimiters
                size_t total_tokens = 0;
                for (const auto& segment : training_segments) {
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
                cout << "Tokenization complete: " << total_tokens << " total tokens from "
                    << training_segments.size() << " segments in " << tokenize_time << "s.\n";

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
                samples,
                labels,
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
            cout << net << endl << endl; // Show the model architecture

            // Create trainer
            dnn_trainer<net_type, adam> trainer(net, adam(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(15));
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

            // Select a segment for generation (use first segment by default)
            size_t segment_idx = 0;
            cout << "Using segment #" << segment_idx << " for generation.\n";
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

                // Stop if end-of-text token is generated
                if (next_token == end_of_text_id)
                    break;

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
 *    + layers: 6
 *    + attention heads: 8
 *    + embedding dimension: 256
 *    + max sequence length: 128
 * - Number of parameters: 4,614,137 (training) - 3,597,749 (inference)
 *
 * After training, the model achieves good memorization of all internal datasets.
 */