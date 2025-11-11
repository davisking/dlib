/*
    @file slm_basic_train_ex.cpp
    @brief Minimal character-level Transformer language model for training and text generation

    This program demonstrates a minimal example of a Very Small Language Model (VSLM)
    using dlib's deep learning tools. It includes two modes:

    1) --train  : Train a small Transformer-based language model on a character-based
                  corpus extracted from "slm_data.h".

    2) --generate: Generate new text from a trained model, given an initial prompt
                   extracted from "slm_data.h" (named shakespeare_prompt).

    Character-level tokenization is used here. Each character is directly transformed
    into an integer token. The model attempts to learn the sequence of characters in
    shakespeare_text. Then you can ask the model to generate new text from a short
    prompt.

    This model is intentionally kept small (few neurons/parameters) to ensure
    simplicity and efficiency. As a result, it may not generalize well to unseen
    patterns or concepts. However, it effectively illustrates the principle of
    attention and the ability to perfectly memorize and reproduce sequences from
    the training data. This makes it a useful educational tool for understanding
    the mechanics of Transformer models.
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>

// Include internal dataset
#include "slm_data.h"

using namespace std;
using namespace dlib;

// We treat each character as a token ID in [0..255]
const int MAX_TOKEN_ID = 255;
const int PAD_TOKEN = 256; // Extra "pad" token if needed

const std::string shakespeare_text = get_dataset_as_text(dataset_id::SHAKESPEARE_EXTRACT);
const std::string prompt_text = get_dataset_as_text(dataset_id::SHAKESPEARE_PROMPT);

// For simplicity, we assume each line from shakespeare_text is appended, ignoring them
std::vector<int> char_based_tokenize(const std::string& text)
{
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (const int c : text)
        tokens.push_back(std::min(c, MAX_TOKEN_ID));

    return tokens;
}

// Function to shuffle samples and labels in sync
void shuffle_samples_and_labels(
    std::vector<matrix<int, 0, 1>>& samples,
    std::vector<unsigned long>& labels)
{
    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ..., N-1
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

    // Create temporary vectors to hold shuffled data
    std::vector<matrix<int, 0, 1>> shuffled_samples(samples.size());
    std::vector<unsigned long> shuffled_labels(labels.size());

    // Apply the shuffle
    for (size_t i = 0; i < indices.size(); ++i)
    {
        shuffled_samples[i] = samples[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }

    // Replace the original data with shuffled data
    samples = std::move(shuffled_samples);
    labels = std::move(shuffled_labels);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("train", "Train a small transformer on the built-in Shakespeare text");
        parser.add_option("generate", "Generate text from a previously trained model (needs shakespeare_prompt)");
        parser.add_option("learning-rate", "Set the learning rate for training (default: 1e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size for training (default: 64)", 1);
        parser.add_option("generation-length", "Set the length of generated text (default: 550)", 1);
        parser.add_option("alpha", "Set the weight decay for Adam optimizer (default: 0.004)", 1);
        parser.add_option("beta1", "Set the first moment coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set the second moment coefficient (default: 0.999)", 1);
        parser.add_option("max-samples", "Set the maximum number of training samples (default: 50000)", 1);
        parser.add_option("shuffle", "Shuffle training sequences and labels before training (default: false)");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train") && !parser.option("generate"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 1e-4);
        const long batch_size = get_option(parser, "batch-size", 64);
        const int generation_length = get_option(parser, "generation-length", 550);
        const double alpha = get_option(parser, "alpha", 0.004);             // Initial learning rate for Adam
        const double beta1 = get_option(parser, "beta1", 0.9);               // Decay rate for the first moment estimate
        const double beta2 = get_option(parser, "beta2", 0.999);             // Decay rate for the second moment estimate
        const size_t max_samples = get_option(parser, "max-samples", 50000); // Default maximum number of training samples

        // We define a minimal config for demonstration
        const long vocab_size = (MAX_TOKEN_ID + 1) + 1; // 256 for chars + 1 pad token
        const long num_layers = 3;
        const long num_heads = 4;
        const long embedding_dim = 64;
        const long max_seq_len = 50; // Small sequence length for the example

        using train_fused_transformer =
            loss_multiclass_log<fc<vocab_size, rms_norm<
            fused_transformer::transformer_stack<num_layers, gelu, dropout_10, max_seq_len, embedding_dim, num_heads,
            token_embeddings<vocab_size, embedding_dim, input<matrix<int, 0, 1>>>>>>>;

        using infer_fused_transformer =
            loss_multiclass_log<fc<vocab_size, rms_norm<
            fused_transformer::transformer_stack<num_layers, gelu, multiply, max_seq_len, embedding_dim, num_heads,
            token_embeddings<vocab_size, embedding_dim, input<matrix<int, 0, 1>>>>>>>;

        // For GPU usage (if any), set gpus = {0} for a single GPU, etc.
        std::vector<int> gpus{ 0 };

        // The model file to store or load
        const std::string model_file = "dlib_lm_chars_model.dat";

        // Training mode
        if (parser.option("train"))
        {
            cout << "=== TRAIN MODE ===\n";

            // 1) Prepare training data using language_model_data utilities
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            build_single_token_prediction_dataset(
                std::vector<std::vector<int>>{ char_based_tokenize(shakespeare_text) },
                max_seq_len,
                PAD_TOKEN,
                false,  // use_left_padding = false (skip sequences shorter than window)
                samples,
                labels
            );

            // Check if the text is too short
            size_t max_sequences = samples.size();
            cout << "Total number of sequences: " << max_sequences << "\n";
            if (max_sequences == 0)
            {
                cerr << "ERROR: The Shakespeare text is too short for training. It must contain at least "
                    << (max_seq_len + 1) << " characters.\n";
                return 0;
            }

            // Limit samples if requested
            if (max_sequences > max_samples)
            {
                cout << "Limiting to " << max_samples << " samples (from " << max_sequences << ")\n";
                samples.resize(max_samples);
                labels.resize(max_samples);
            }

            // Shuffle samples and labels if the --shuffle option is enabled
            if (parser.option("shuffle"))
            {
                cout << "Shuffling training sequences and labels...\n";
                shuffle_samples_and_labels(samples, labels);
            }

            // 2) Construct the network in training mode
            train_fused_transformer net;
            if (file_exists(model_file))
            {
                cout << "Loading existing model from " << model_file << "\n";
                deserialize(model_file) >> net;
            }

            // 3) Create dnn_trainer
            dnn_trainer<train_fused_transformer, adam> trainer(net, adam(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(5000);
            trainer.set_max_num_epochs(100);
            trainer.be_verbose();

            // 4) Train
            trainer.train(samples, labels);

            // 5) Evaluate quickly on the training set
            auto predicted = net(samples);
            size_t correct = 0;
            for (size_t i = 0; i < labels.size(); ++i)
                if (predicted[i] == labels[i]) correct++;
            double accuracy = (double)correct / labels.size();
            cout << "Training accuracy (on this sample set): " << accuracy << "\n";

            // 6) Save the model
            net.clean();
            serialize(model_file) << net;
            cout << "Model saved to " << model_file << "\n";
        }

        // Generation mode
        if (parser.option("generate"))
        {
            cout << "=== GENERATE MODE ===\n";

            // 1) Load the trained model
            infer_fused_transformer net;
            if (file_exists(model_file))
            {
                deserialize(model_file) >> net;
                cout << "Loaded model from " << model_file << "\n";
            }
            else
            {
                cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }
            cout << "Model parameters: " << count_parameters(net) << endl << endl;

            // 2) Get the prompt from the included slm_data.h
            if (prompt_text.empty())
            {
                cerr << "No prompt found in slm_data.h.\n";
                return 0;
            }

            // 3) Initialize inference context
            inference_context ctx(max_seq_len, 1, PAD_TOKEN);

            // Add prompt tokens to context
            ctx.add_tokens(char_based_tokenize(prompt_text));

            cout << "\nInitial prompt:\n" << prompt_text << "\n\n";
            cout << "Generated text:\n" << prompt_text;

            // 4) Generate new text using inference_context
            for (int i = 0; i < generation_length; ++i)
            {
                // Get input window from context
                auto input_seq = ctx.get_input_window();

                // Predict next token
                const unsigned long next_token = net(input_seq);

                // Print the generated character
                cout << static_cast<char>(std::min(static_cast<int>(next_token), MAX_TOKEN_ID)) << flush;

                // Add predicted token to context (automatic sliding window)
                ctx.add_token(next_token);
            }

            cout << "...\n\n(end of generation)\n";
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
 * This program demonstrates the training of a language model on about 14.6k sequences.
 * The training process produces a data file of approximately 20MB on disk.
 *
 * - Transformer model configuration:
 *    + vocabulary size: 257
 *    + layers: 3
 *    + attention heads: 4
 *    + embedding dimension: 64
 *    + max sequence length: 50
 * - Number of parameters: 5,185,864
 *
 * The training can be performed using the following command line:
 * > ./slm_basic_train_ex --train --shuffle
 *
 * After this phase, the model achieves perfect prediction accuracy (i.e acc~99.98%).
 * The generation option produces text that is very similar or identical to the original
 * training data, as illustrated by the example below:
 * 
 * > Generated text:
 * > KING RICHARD III:
 * > Bear her my true love's kiss; and so, farewell.
 * > Relenting fool, and shallow, changing woman!
 * > How now! what news?
 * >
 * > RATCLIFF:
 * > My gracious sovereign, on the western coast
 * > Rideth a puissant navy; to the shore
 * > Throng many doubtful hollow-hearted friends,
 * > Unarm'd, and unresolved to beat them back:
 * > 'Tis thought that Richmond is their admiral;
 * > And there they hull, expecting but the aid
 * > Of Buckingham to welcome them ashore.
 * 
 * > KING RICHARD III:
 * > Some light-foot friend post to the Duke of Norfolk:
 * > Ratcliff, thyself, or Cate...
 */