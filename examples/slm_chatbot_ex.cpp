/*!
    @file slm_chatbot_ex.cpp
    @brief Transformer-based chatbot with staged fine-tuning

    This program demonstrates how to build a specialized chatbot using transformer
    architecture with Mixture-of-Experts layers. The fine-tuning process is used to
    specialize the model for conversational Q&A tasks using formatted prompt-response
    pairs with special tags.

    Key features:
    - Layer-wise learning rate multipliers for selective fine-tuning
    - Learning rate scheduler with warmup and cosine decay
    - Padding-aware causal attention via tril_padding_context
    - Stochastic text generation with temperature, top-k, nucleus sampling
    - Repetition penalty and min-p filtering for improved generation quality

    The chatbot is designed to answer questions about black holes and
    related astrophysics topics, demonstrating how proper data formatting and
    tagging can specialize a language model for specific domains.

    Usage modes:
    --fine-tune          Fine-tune on Q&A pairs for chatbot specialization
    --prompt             Interactive prompting mode

    Data format for fine-tuning:
    <question><text>What is a black hole?</text>
    <answer><text>A black hole is a region of spacetime...</text>

    The special tags help the model learn the conversational structure and
    role-based response patterns.
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
#include <sstream>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/tokenizer/bpe_tokenizer.h>
#include <dlib/misc_api.h>

// Include internal dataset
#include "slm_data.h"

using namespace std;
using namespace dlib;

namespace dlib
{
    // Expert network architecture for MoE layer
    template <template <typename> class DO, long d_model>
    using expert_net_type = swiglu<DO, d_model, input_tensor>;

    // Complete transformer block with MoE-based feed-forward layer
    template <template <typename> class ACT, template <typename> class DO,
        long d_model, long num_heads, typename MODE, typename SUBNET>
    using trans_moe_block =
        moe_ffn<expert_net_type<DO, d_model>, 4, 0, MODE, DO,
        add_prev1<multihead_attention<ACT, DO, d_model, num_heads, rms_norm<tag1<SUBNET>>>>>;

    // Classification head for next-token prediction in conversational context
    template <long num_logits, typename SUBNET>
    using classification_head = loss_cross_entropy_per_logit<linear<num_logits, rms_norm<SUBNET>>>;

    // Chatbot model configuration
    template<
        long vocab_size = 2000,
        long num_layers = 3,
        long num_heads = 6,
        long embedding_dim = 192,
        template <typename> class activation_func = gelu,
        template <typename> class dropout_policy = dropout_10
    >
    struct chatbot_config {
        static constexpr long VOCAB_SIZE = vocab_size;
        static constexpr long NUM_LAYERS = num_layers;
        static constexpr long NUM_HEADS = num_heads;
        static constexpr long EMBEDDING_DIM = embedding_dim;

        struct validation {
            static_assert(VOCAB_SIZE > 0, "Vocabulary size must be positive");
            static_assert(NUM_LAYERS > 0, "Number of layers must be positive");
            static_assert(NUM_HEADS > 0, "Number of attention heads must be positive");
            static_assert(EMBEDDING_DIM% NUM_HEADS == 0, "Embedding dimension must be divisible by number of heads");
        };

        // Network component definitions for training (with dropout)
        template <typename SUBNET>
        using t_transformer_block =
            trans_moe_block<activation_func, dropout_policy, EMBEDDING_DIM, NUM_HEADS,
            training_mode_tag, SUBNET>;

        // Network component definitions for inference (using multiply)
        template <typename SUBNET>
        using i_transformer_block =
            trans_moe_block<activation_func, multiply, EMBEDDING_DIM, NUM_HEADS,
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
                ss << "Chatbot configuration:\n"
                    << "- vocabulary: " << VOCAB_SIZE << " tokens\n"
                    << "- layers: " << NUM_LAYERS << " transformer layers with MoE\n"
                    << "- attention heads: " << NUM_HEADS << "\n"
                    << "- embedding dimension: " << EMBEDDING_DIM << "\n"
                    << "- experts per layer: 4 (auto top-n selection)";
                return ss.str();
            }
        };
    };
}

// Signal handling for clean termination
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

    void setup_interrupt_handler() {
        SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
    }
#else
    void signal_handler(int signal) {
        if (signal == SIGINT) {
            g_terminate_flag.store(true);
            cout << "\nCtrl+C detected, cleaning up..." << endl;
        }
    }

    void setup_interrupt_handler() {
        std::signal(SIGINT, signal_handler);
    }
#endif
}

// ----------------------------------------------------------------------------------------

void display_random_qa_samples(size_t num_samples = 3)
{
    try {
        // Load Q&A dataset
        auto qa_pairs = get_dataset_as_pairs({ dataset_id::BLACK_HOLE_QA_PARTA });
        if (qa_pairs.empty()) {
            cout << "Warning: No Q&A pairs found in dataset\n";
            return;
        }

        cout << "=== SAMPLE QUESTIONS FROM TRAINING DATA ===\n";
        cout << "Total Q&A pairs in dataset <part.a>: " << qa_pairs.size() << "\n\n";

        // Generate random indices
        dlib::rand rng(std::time(0));
        std::vector<size_t> indices;
        for (size_t i = 0; i < qa_pairs.size(); ++i)
            indices.push_back(i);

        // Shuffle indices
        for (size_t i = indices.size() - 1; i > 0; --i) {
            size_t j = rng.get_random_32bit_number() % (i + 1);
            std::swap(indices[i], indices[j]);
        }

        // Display random samples (questions only)
        num_samples = std::min(num_samples, qa_pairs.size());
        for (size_t i = 0; i < num_samples; ++i) {
            size_t idx = indices[i];
            cout << "Example " << (i + 1) << " - ";
            cout << "Q: " << qa_pairs[idx].first << "\n";
        }

        cout << "=========================================\n\n";
    }
    catch (const std::exception& e) {
        cerr << "Error loading Q&A samples: " << e.what() << "\n";
    }
}

// Visitor for setting learning rate multiplier on computational layers
struct lr_mult_visitor
{
    double mult;

    lr_mult_visitor(double m) : mult(m) {}

    template <typename layer_type>
    void operator()(size_t, layer_type& l) const
    {
        set_learning_rate_multiplier_impl(l, mult);
    }

private:
    template <typename T>
    static auto set_learning_rate_multiplier_impl(T& layer, double m)
        -> decltype(layer.layer_details().set_learning_rate_multiplier(m), void())
    {
        layer.layer_details().set_learning_rate_multiplier(m);
    }

    template <typename T>
    static void set_learning_rate_multiplier_impl(T&, ...)
    {
        // No-op for layers without this method
    }
};

int main(int argc, char** argv)
{
    try
    {
        setup_interrupt_handler();

        command_line_parser parser;
        parser.add_option("fine-tune", "Fine-tune model on Q&A pairs for chatbot specialization");
        parser.add_option("prompt", "Enter interactive prompting mode");
        parser.add_option("learning-rate", "Set the learning rate (default: 1e-5)", 1);
        parser.add_option("batch-size", "Set mini-batch size (default: 32)", 1);
        parser.add_option("max-epochs", "Set maximum training epochs (default: 150)", 1);
        parser.add_option("weight-decay", "Set the weight decay for AdamW (default: 0.01)", 1);
        parser.add_option("beta1", "Set AdamW's beta1 coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set AdamW's beta2 coefficient (default: 0.999)", 1);
        parser.add_option("patience", "Set iterations without progress threshold (default: 15000)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_lm_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: dlib_lm_tokenizer.vocab)", 1);
        parser.add_option("temperature", "Set sampling temperature, higher = more creative (default: 0.8)", 1);
        parser.add_option("top-k", "Set top-k filtering, max tokens to consider (default: 50)", 1);
        parser.add_option("top-p", "Set nucleus sampling threshold (default: 0.9)", 1);
        parser.add_option("repeat-penalty", "Set repetition penalty (default: 1.2)", 1);
        parser.add_option("min-p", "Set relative minimum probability threshold (default: 0.05)", 1);
        parser.add_option("deterministic", "Force deterministic generation mode (Argmax)");
        parser.parse(argc, argv);

        if (!parser.option("fine-tune") && !parser.option("prompt")) {
            cout << "Transformer-based chatbot with staged fine-tuning\n\n";
            parser.print_options();
            return 0;
        }

        // Training hyperparameters
        const double learning_rate = get_option(parser, "learning-rate", 1e-5);
        const size_t batch_size = get_option(parser, "batch-size", 32);
        const size_t max_epochs = get_option(parser, "max-epochs", 150);
        const long patience = get_option(parser, "patience", 15000);
        const double weight_decay = get_option(parser, "weight-decay", 0.01);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.999);

        // File paths
        const std::string model_file = get_option(parser, "model-file", std::string("dlib_lm_moe_model.dat"));
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", std::string("dlib_lm_tokenizer.vocab"));

        // Configuration parameters
        const long vocab_size = 2000;
        const long max_seq_len = 128;
        using config = chatbot_config<vocab_size>;
        using train_net = config::network_type<true>;
        using infer_net = config::network_type<false>;
        cout << config::model_info::describe() << "\n\n";

        // GPU configuration
        std::vector<int> gpus{ 0 };
        if (parser.option("fine-tune"))
        {
            cout << "=== FINE-TUNING MODE ===\n";
            cout << "Objective: specialize model for conversational Q&A with proper formatting\n\n";

            // Setup trainer for fine-tuning
            std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.'))
                + "_finetuned.dat";
            train_net net;
            dnn_trainer<train_net, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-7);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_max_num_epochs(max_epochs);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + finetuned_model, std::chrono::minutes(25));
            trainer.be_quiet();

            // Load tokenizer & model
            bpe_tokenizer tokenizer;
            if (file_exists(model_file) &&
                !file_exists("chkpt-" + finetuned_model)) deserialize(model_file) >> net >> tokenizer;
            else if (file_exists(finetuned_model) &&
                !file_exists("chkpt-" + finetuned_model)) deserialize(finetuned_model) >> net >> tokenizer;
            else if (file_exists(tokenizer_file))
                deserialize(tokenizer_file) >> tokenizer;
            else {
                cout << "Pre-trained tokenizer not found at: " << tokenizer_file << endl;
                return 1;
            }
            const int pad_token = tokenizer.get_special_token_id("<pad>");
            layer<0>(net).loss_details().set_ignore_index(pad_token);

            // Load Q&A datasets for fine-tuning
            cout << "Loading Q&A training datasets...\n";
            std::vector<dataset_id> qa_datasets = {
                dataset_id::BLACK_HOLE_QA_PARTA,
                dataset_id::BLACK_HOLE_QA_PARTB,
                dataset_id::BLACK_HOLE_QA_PARTC
            };
            auto all_qa_pairs = get_dataset_as_pairs(qa_datasets);

            cout << "Loaded " << all_qa_pairs.size() << " Q&A pairs\n";
            cout << "Format: uses special tags for role-based learning\n\n";

            // Tokenize Q&A segments with markers
            cout << "Tokenizing Q&A segments...\n";
            int text_start_id = tokenizer.get_special_token_id("<text>"),
                text_end_id = tokenizer.get_special_token_id("</text>"),
                question_id = tokenizer.get_special_token_id("<question>"),
                answer_id = tokenizer.get_special_token_id("<answer>");

            std::vector<std::vector<int>> qa_tokens;
            size_t total_tokens = 0;
            for (const auto& qa_pair : all_qa_pairs) {
                std::vector<int> pair_tokens;

                // Format: <question><text>question_text</text>
                pair_tokens.push_back(question_id);
                pair_tokens.push_back(text_start_id);
                auto q_tokens = tokenizer.encode(qa_pair.first);
                pair_tokens.insert(pair_tokens.end(), q_tokens.begin(), q_tokens.end());
                pair_tokens.push_back(text_end_id);

                // Format: <answer><text>answer_text</text>
                pair_tokens.push_back(answer_id);
                pair_tokens.push_back(text_start_id);
                auto a_tokens = tokenizer.encode(qa_pair.second);
                pair_tokens.insert(pair_tokens.end(), a_tokens.begin(), a_tokens.end());
                pair_tokens.push_back(text_end_id);

                total_tokens += pair_tokens.size();
                qa_tokens.push_back(std::move(pair_tokens));
            }
            cout << "Tokenization complete: " << total_tokens << " total Q&A tokens\n\n";

            // Prepare fine-tuning dataset
            cout << "Building fine-tuning dataset...\n";
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;
            build_single_token_prediction_dataset(
                qa_tokens,
                max_seq_len,
                tokenizer.get_special_token_id("<pad>"),
                true,
                samples,
                labels
            );
            cout << "Fine-tuning samples: " << samples.size() << "\n";
            if (samples.empty()) {
                cerr << "Error: No fine-tuning samples generated\n";
                return 1;
            }

            // Release memory
            qa_tokens.clear();

            // Strategy: Freeze embeddings and lower transformer layers, fine-tune upper layers
            // - Embeddings: frozen (preserve learned token representations)
            // - Lower transformer blocks: frozen or very slow (preserve general language understanding)
            // - Upper transformer blocks: slow learning (adapt to domain)
            // - Classification head: normal learning (specialize for task)
            cout << "Applying freezing strategy for fine-tuning\n";
            // Step 1: freeze everything first (multiplier = 0)
            set_all_learning_rate_multipliers(net, 0.0);
            // Step 2: unfreeze classification head (layers 1-2: linear + rms_norm)
            layer<1>(net).layer_details().set_learning_rate_multiplier(1.0);  // linear (classification)
            layer<2>(net).layer_details().set_learning_rate_multiplier(1.0);  // rms_norm
            // Step 3: partially unfreeze upper transformer layers with gradual unfreezing
            // For a 3-layer transformer, unfreeze the last 1-2 blocks with reduced LR
            // Layer indices depend on architecture - adjust based on `net` output
            // Top transformer block: moderate learning
            visit_layers_range<3, 40>(net, lr_mult_visitor(0.3));
            // Middle transformer block: slower learning  
            visit_layers_range<40, 75>(net, lr_mult_visitor(0.1));
            cout << net << endl;

            size_t epoch = 0;
            size_t batches_count = 0, batches_seen = 0, samples_seen = 0;
            double total_loss = 0.0;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            // Setup learning rate scheduler with warmup
            const size_t steps_per_epoch = (samples.size() + batch_size - 1) / batch_size;
            const size_t total_steps = steps_per_epoch * max_epochs;
            const size_t warmup_steps = std::min(size_t(500), total_steps / 10);  // 10% or 500 steps max

            lr_scheduler scheduler(
                learning_rate,          // peak_lr
                warmup_steps,           // warmup_steps
                total_steps,            // total_steps
                1e-7,                   // min_lr
                lr_decay_type::COSINE   // decay_type
            );

            // Restore scheduler state if exists
            const std::string scheduler_state_file = "scheduler-" + finetuned_model;
            if (file_exists(scheduler_state_file)) {
                deserialize(scheduler_state_file) >> scheduler;
                cout << "Scheduler resumed: step " << scheduler.get_current_step()
                    << ", phase: " << scheduler.get_phase_name()
                    << ", learning rate: " << scheduler.get_learning_rate() << "\n";
            }

            cout << "Learning rate schedule:\n"
                << "  peak learning rate: " << scheduler.get_peak_lr() << "\n"
                << "  min learning rate: " << scheduler.get_min_lr() << "\n"
                << "  warmup steps: " << scheduler.get_warmup_steps() << "\n"
                << "  total steps: " << scheduler.get_total_steps() << "\n"
                << "  current step: " << scheduler.get_current_step() << "\n"
                << "  current phase: " << scheduler.get_phase_name() << "\n"
                << "  decay type: COSINE\n\n";

            // Training loop
            cout << "Starting fine-tuning...\n";
            while (!scheduler.is_training_complete()
                && epoch < max_epochs && !g_terminate_flag.load())
            {
                total_loss = 0.0;
                batches_seen = 0;
                samples_seen = 0;
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

                    // Update learning rate from scheduler
                    double current_lr = scheduler.get_learning_rate();
                    trainer.set_learning_rate(current_lr);

                    std::vector<long> pad_lengths(batch_samples.size());
                    for (size_t j = 0; j < batch_samples.size(); ++j)
                        pad_lengths[j] = count_leading_padding(batch_samples[j], pad_token);
                    tril_padding_context::set_from_lengths(pad_lengths);

                    // Train
                    trainer.train_one_step(batch_samples, batch_labels);

                    // Advance scheduler
                    scheduler.step();

                    total_loss += trainer.get_average_loss();
                    batches_seen++;
                    samples_seen += batch_samples.size();

                    // Progress reporting
                    if (batches_count++ % 100 == 0) {
                        double avg_loss = total_loss / batches_seen;
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - epoch_start).count();
                        double samples_per_sec = samples_seen / (elapsed > 0 ? elapsed : 1);

                        std::ios_base::fmtflags old_flags = cout.flags();
                        std::streamsize old_precision = cout.precision();

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " \t loss: " << std::fixed << std::setprecision(3) << avg_loss
                            << " \t lr: " << std::scientific << std::setprecision(2) << current_lr
                            << " \t phase: " << scheduler.get_phase_name()
                            << " \t progress: " << std::fixed << std::setprecision(1)
                            << (scheduler.get_total_progress() * 100) << "%"
                            << " \t speed: " << samples_per_sec << " samples/sec\n";
                        cout.flush();

                        cout.flags(old_flags);
                        cout.precision(old_precision);

                        // Save scheduler checkpoint periodically
                        serialize(scheduler_state_file) << scheduler;
                    }

                    // Check if scheduler indicates training is complete
                    if (scheduler.is_training_complete()) break;
                }
                epoch++;
            }
            tril_padding_context::clear();

            // Save fine-tuned model
            set_all_learning_rate_multipliers(net, 1.0);  // Reset multipliers before saving
            cout << "\nFine-tuning complete, saving specialized model...\n";
            cout << "Final step: " << scheduler.get_current_step()
                << ", final learning rate: " << scheduler.get_learning_rate() << "\n";
            net.clean();

            serialize(finetuned_model) << net << tokenizer;
            cout << "Fine-tuned model saved to " << finetuned_model << "\n";

            cout << "\nFine-tuning completed successfully\n";
            cout << "The model is now specialized for chatbot Q&A interactions\n";
            cout << "Next step: use --prompt to interact with the Chatbot\n";
        }

        // PROMPTING MODE
        else if (parser.option("prompt"))
        {
            cout << "=== INTERACTIVE PROMPTING MODE ===\n";
            cout << "Chat specialized in astrophysics and black holes\n\n";

            // Display 3 random sample questions from training data
            display_random_qa_samples(5);
            cout << "Type 'quit' to exit\n\n";

            // Sampling parameters for text generation
            size_t top_k = get_option(parser, "top-k", 50);
            float top_p = get_option(parser, "top-p", 0.9f);
            float repeat_penalty = get_option(parser, "repeat-penalty", 1.2f);
            float min_p = get_option(parser, "min-p", 0.05f);
            bool deterministic_mode = parser.option("deterministic");
            float temperature = deterministic_mode ? 1.0f : get_option(parser, "temperature", 0.8f);
            dlib::rand rng(std::time(0));

            // Load fine-tuned model
            bpe_tokenizer tokenizer;
            softmaxm<multiply<infer_net::subnet_type>> generator(multiply_(1.0 / temperature));
            {
                infer_net net;
                std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.'))
                    + "_finetuned.dat";
                if (!file_exists(finetuned_model)) {
                    cerr << "Error: fine-tuned model not found: " << finetuned_model << "\n";
                    cerr << "Please run --fine-tune first.\n";
                    return 1;
                }
                deserialize(finetuned_model) >> net >> tokenizer;
                cout << "Fine-tuned model loaded from " << finetuned_model << "\n\n";
                generator.subnet().subnet() = net.subnet();
            }            

            // Get special token IDs
            int text_start_id = tokenizer.get_special_token_id("<text>");
            int text_end_id = tokenizer.get_special_token_id("</text>");
            int question_id = tokenizer.get_special_token_id("<question>");
            int answer_id = tokenizer.get_special_token_id("<answer>");

            // Setup inference context
            const int pad_token = tokenizer.get_special_token_id("<pad>");
            inference_context ctx(max_seq_len, 3, pad_token);

            // Interactive loop
            while (!g_terminate_flag.load())
            {
                // Get user input
                cout << "You: ";
                cout.flush();

                std::string user_input;
                if (!std::getline(std::cin, user_input)) break;

                // Trim whitespace
                user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
                user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);
                if (user_input.empty()) continue;

                // Check for quit command
                if (user_input == "quit" || user_input == "exit") {
                    cout << "Goodbye!\n";
                    break;
                }

                // Tokenize user input with proper formatting
                // Format: <question><text>user_input</text>
                std::vector<int> input_tokens;
                input_tokens.push_back(question_id);
                input_tokens.push_back(text_start_id);
                auto q_tokens = tokenizer.encode(user_input);
                input_tokens.insert(input_tokens.end(), q_tokens.begin(), q_tokens.end());
                input_tokens.push_back(text_end_id);

                // Add to context
                ctx.add_tokens(input_tokens);

                // Prepare for bot response
                // Format: <answer><text>
                ctx.add_token(answer_id);
                ctx.add_token(text_start_id);

                // Generate response token by token
                cout << "CHATBOT: ";
                cout.flush();

                // Top-k/top-p (nucleus) sampling for non-deterministic text generation.
                // This function applies temperature scaling, repetition penalty, min-p filtering, 
                // top-k filtering, and nucleus sampling to select the next token.
                auto top_k_p_sample = [&rng, &ctx, &text_end_id](
                    const float* probs, size_t N, size_t k,
                    float p, float repeat_penalty, float min_p) -> size_t
                    {
                        // Copy probabilities
                        std::vector<float> p_copy(probs, probs + N);

                        // Step 1: Apply repetition penalty ONCE
                        if (repeat_penalty > 1.0f) {
                            const auto& context_tokens = ctx.get_full_context();

                            // Penalize only recent tokens (last 20%)
                            size_t recent_size = std::max(size_t(1),
                                static_cast<size_t>(context_tokens.size() * 0.2));
                            size_t start_idx = (context_tokens.size() > recent_size)
                                ? context_tokens.size() - recent_size : 0;

                            for (size_t i = start_idx; i < context_tokens.size(); ++i) {
                                int token_id = context_tokens[i];
                                if (token_id >= 0 && static_cast<size_t>(token_id) < N) {
                                    p_copy[token_id] /= repeat_penalty;
                                }
                            }
                        }

                        // Step 2: Renormalize after penalty
                        float sum_after_penalty = 0.0f;
                        for (size_t i = 0; i < N; ++i) {
                            sum_after_penalty += p_copy[i];
                        }
                        if (sum_after_penalty > 1e-8f) {
                            for (size_t i = 0; i < N; ++i) {
                                p_copy[i] /= sum_after_penalty;
                            }
                        }

                        // Step 3: Find max probability for min-p filtering
                        float max_prob = *std::max_element(p_copy.begin(), p_copy.end());
                        float min_p_threshold = max_prob * min_p;

                        // Step 4: Build candidate list with min-p filter
                        std::vector<std::pair<size_t, float>> candidates;
                        candidates.reserve(N);

                        for (size_t i = 0; i < N; ++i) {
                            if (p_copy[i] >= min_p_threshold) {
                                candidates.push_back({ i, p_copy[i] });
                            }
                        }

                        if (candidates.empty()) {
                            return text_end_id;  // Fallback
                        }

                        // Step 5: Sort and apply top-k
                        k = std::min(k, candidates.size());
                        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                            [](const auto& a, const auto& b) { return a.second > b.second; });

                        // Step 6: Apply top-p (nucleus sampling)
                        float cumsum = 0.0f;
                        size_t cutoff = 0;
                        for (size_t i = 0; i < k; ++i) {
                            cumsum += candidates[i].second;
                            cutoff = i;
                            if (cumsum >= p) break;
                        }

                        // Step 7: Renormalize filtered distribution
                        float final_sum = 0.0f;
                        for (size_t i = 0; i <= cutoff; ++i) {
                            final_sum += candidates[i].second;
                        }

                        if (final_sum < 1e-8f) {
                            return candidates[0].first;  // Return most probable
                        }

                        // Step 8: Sample from normalized distribution
                        float r = rng.get_random_float() * final_sum;
                        float cs = 0.0f;
                        for (size_t i = 0; i <= cutoff; ++i) {
                            cs += candidates[i].second;
                            if (r <= cs) {
                                return candidates[i].first;
                            }
                        }

                        return candidates[0].first;  // Fallback
                    };

                int next_token, max_response_tokens = 3 * max_seq_len;
                for (int i = 0; i < max_response_tokens && !g_terminate_flag.load(); ++i)
                {
                    // Get current context window and predict next token
                    auto input_window = ctx.get_input_window();
                    long pad_len = count_leading_padding(input_window, pad_token);
                    tril_padding_context::set_uniform(pad_len, 1);
                    auto& probs_tensor = generator(input_window);

                    // Extract dimensions
                    const long seq_len = probs_tensor.nr();
                    const long vocab_size = probs_tensor.nc();
                    const long last_pos = seq_len - 1;

                    // Get pointer to probabilities at last position
                    const long offset = tensor_index(probs_tensor, 0, 0, last_pos, 0);
                    const float* probs = probs_tensor.host() + offset;

                    if (deterministic_mode) {
                        // Argmax: select most probable token
                        const float* max_ptr = std::max_element(probs, probs + vocab_size);
                        next_token = static_cast<int>(std::distance(probs, max_ptr));
                    }
                    else {
                        // Stochastic sampling
                        next_token = top_k_p_sample(probs, vocab_size, top_k, top_p, repeat_penalty, min_p);
                    }

                    ctx.add_token(next_token);

                    // Decode and display token
                    std::string token_text = tokenizer.decode(next_token, false);
                    cout << token_text;
                    cout.flush();

                    // Stop if end token is found
                    if (next_token == text_end_id) break;
                }
                cout << "\n\n";
            }
            tril_padding_context::clear();
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}