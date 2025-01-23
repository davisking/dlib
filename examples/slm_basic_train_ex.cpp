/*
    This program demonstrates a minimal example of a Very Small Language Model (VSLM)
    using dlib's deep learning tools. It includes two modes:

    1) --train  : Train a small Transformer-based language model on a character-based
                  corpus extracted from "slm_data.h" (named shakespeare_text).

    2) --generate: Generate new text from a trained model, given an initial prompt
                   extracted from "slm_data.h" (named shakespeare_prompt).

    The "slm_dels.h" header is expected to provide a comprehensive Transformer
    definition with the following key elements:
      - A configurable transformer_config
      - The use of classification_head to output a single token
      - The network_type<true> or network_type<false> for training vs inference
      - The typical dlib constructs (input<matrix<int>>, etc.)

    Character-level tokenization is used here. Each character is directly transformed
    into an integer token. The model attempts to learn the sequence of characters in
    shakespeare_text. Then you can ask the model to generate new text from a short
    prompt.

    This model is intentionally kept small (few neurons/parameters) to ensure
    simplicity and efficiency. As a result, it may not generalize well to unseen
    patterns or concepts. However, it effectively illustrates the principle of
    attention and the ability to perfectly memorize and reproduce sequences from
    the training data. This makes it a useful educational tool for understanding
    the mechanics of Transformer models, even if it lacks the capacity for
    sophisticated language understanding.
*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>

// Include Transformer definitions
#include "slm_defs.h"

// This header "slm_data.h" is assumed to contain:
//   const std::string shakespeare_text;
//   const std::string shakespeare_prompt;
#include "slm_data.h"

// ----------------------------------------------------------------------------------------

using namespace std;
using namespace dlib;

// We treat each character as a token ID in [0..255].
const int MAX_TOKEN_ID = 255;
const int PAD_TOKEN = 256; // an extra "pad" token if needed

// For simplicity, we assume each line from shakespeare_text is appended, ignoring them.
std::vector<int> char_based_tokenize(const std::string& text)
{
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (const int c : text)
    {
        tokens.push_back(std::min(c, MAX_TOKEN_ID));
    }
    return tokens;
}

// Function to shuffle samples and labels in sync
void shuffle_samples_and_labels(std::vector<matrix<int, 0, 1>>& samples, std::vector<unsigned long>& labels) {
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
        parser.add_option("generation-length", "Set the length of generated text (default: 400)", 1);
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
        const int generation_length = get_option(parser, "generation-length", 400);
        const double alpha = get_option(parser, "alpha", 0.004);       // Initial learning rate for Adam
        const double beta1 = get_option(parser, "beta1", 0.9);         // Decay rate for the first moment estimate
        const double beta2 = get_option(parser, "beta2", 0.999);       // Decay rate for the second moment estimate
        const size_t max_samples = get_option(parser, "max-samples",50000); // Default maximum number of training samples

        // We define a minimal config for demonstration
        const long vocab_size = MAX_TOKEN_ID + 1 + 1;   // 256 for chars + 1 pad token
        const long num_layers = 3;
        const long num_heads = 4;
        const long embedding_dim = 64;
        const long max_seq_len = 80;   // a small sequence length for the example
        const bool use_squeezing = false;

        using my_transformer_cfg = transformer::transformer_config<
            vocab_size,
            num_layers,
            num_heads,
            embedding_dim,
            max_seq_len,
            use_squeezing,
            gelu,
            dropout_10
        >;

        // For GPU usage (if any), set gpus = {0} for a single GPU, etc.
        std::vector<int> gpus{ 0 };

        // The model file to store or load
        const std::string model_file = "shakespeare_lm_char_model.dat";

        // ----------------------------------------------------------------------------------------
        // Train mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("train"))
        {
            cout << "=== TRAIN MODE ===\n";

            // 1) Prepare training data (simple approach)
            // We will store characters from shakespeare_text into a vector
            // and then produce training samples of length (max_seq_len+1),
            // where the last token is the label to predict from the preceding max_seq_len.
            auto full_tokens = char_based_tokenize(shakespeare_text);
            if (full_tokens.empty())
            {
                cerr << "ERROR: The Shakespeare text is empty. Please provide a valid training text.\n";
                return 0;
            }

            // Calculate the maximum number of sequences
            size_t max_sequences = (full_tokens.size() > (size_t)max_seq_len + 1)
                ? (full_tokens.size() - ((size_t)max_seq_len + 1))
                : 0;

            // Display the size of the training text and the number of sequences
            cout << "Training text size: " << full_tokens.size() << " characters\n";
            cout << "Maximum number of sequences: " << max_sequences << "\n";

            // Check if the text is too short
            if (max_sequences == 0)
            {
                cerr << "ERROR: The Shakespeare text is too short for training. It must contain at least "
                    << (max_seq_len + 1) << " characters.\n";
                return 0;
            }

            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            // Let's create a training set of about (N) samples from the text
            // Each sample: [x0, x1, ..., x_(max_seq_len-1)] -> y
            // We'll store them in "samples" and "labels".
            const size_t N = (max_sequences < max_samples) ? max_sequences : max_samples;
            for (size_t start = 0; start < N; ++start)
            {
                matrix<int, 0, 1> seq(max_seq_len, 1);
                for (long t = 0; t < max_seq_len; ++t)
                    seq(t, 0) = full_tokens[start + t];
                samples.push_back(seq);
                labels.push_back(full_tokens[start + max_seq_len]);
            }

            // Shuffle samples and labels if the --shuffle option is enabled
            if (parser.option("shuffle"))
            {
                cout << "Shuffling training sequences and labels...\n";
                shuffle_samples_and_labels(samples, labels);
            }

            // 3) Construct the network in training mode
            using net_type = my_transformer_cfg::network_type<true>;
            net_type net;
            if (file_exists(model_file))
                deserialize(model_file) >> net;

            // 4) Create dnn_trainer
            dnn_trainer<net_type, adam> trainer(net, adam(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(15000);
            trainer.set_max_num_epochs(400);
            trainer.be_verbose();

            // 5) Train
            trainer.train(samples, labels);

            // 6) Evaluate quickly on the training set
            auto predicted = net(samples);
            size_t correct = 0;
            for (size_t i = 0; i < labels.size(); ++i)
                if (predicted[i] == labels[i])
                    correct++;
            double accuracy = (double)correct / labels.size();
            cout << "Training accuracy (on this sample set): " << accuracy << "\n";

            // 7) Save the model
            net.clean();
            serialize(model_file) << net;
            cout << "Model saved to " << model_file << "\n";
        }

        // ----------------------------------------------------------------------------------------
        // Generate mode
        // ----------------------------------------------------------------------------------------
        if (parser.option("generate"))
        {
            cout << "=== GENERATE MODE ===\n";
            // 1) Load the trained model
            using net_infer = my_transformer_cfg::network_type<false>;
            net_infer net;
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
            cout << my_transformer_cfg::model_info::describe() << endl;
            cout << "Model parameters: " << count_parameters(net) << endl << endl;

            // 2) Get the prompt from the included slm_data.h
            std::string prompt_text = shakespeare_prompt;
            if (prompt_text.empty())
            {
                cerr << "No prompt found in slm_data.h.\n";
                return 0;
            }
            // If prompt is longer than max_seq_len, we keep only the first window
            if (prompt_text.size() > (size_t)max_seq_len)
                prompt_text.erase(prompt_text.begin() + max_seq_len, prompt_text.end());

            // Convert prompt to a token sequence
            const auto prompt_tokens = char_based_tokenize(prompt_text);

            // Put into a dlib matrix
            matrix<int, 0, 1> input_seq(max_seq_len, 1);
            // Fill with pad if prompt is shorter than max_seq_len
            for (long i = 0; i < max_seq_len; ++i)
            {
                if ((size_t)i < prompt_tokens.size())
                    input_seq(i, 0) = prompt_tokens[i];
                else
                    input_seq(i, 0) = PAD_TOKEN;
            }

            cout << "\nInitial prompt:\n" << prompt_text << " (...)\n\n\nGenerated text:\n" << prompt_text;

            // 3) Generate new text
            // We'll predict one character at a time, then shift the window
            for (int i = 0; i < generation_length; ++i)
            {
                const int next_char = net(input_seq); // single inference

                // Print the generated character
                cout << static_cast<char>(std::min(next_char, MAX_TOKEN_ID)) << flush;

                // Shift left by 1
                for (long i = 0; i < max_seq_len - 1; ++i)
                    input_seq(i, 0) = input_seq(i + 1, 0);
                input_seq(max_seq_len - 1, 0) = std::min(next_char, MAX_TOKEN_ID);
            }

            cout << "\n\n(end of generation)\n";
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
 * This program demonstrates the training of a language model on about 15k sequences.
 * The training process produces a data file of approximately 32MB on disk.
 *
 * - Transformer model configuration:
 *    + vocabulary size: 257
 *    + layers: 3
 *    + attention heads: 4
 *    + embedding dimension: 64
 *    + max sequence length: 80
 * - Number of parameters: 8,247,496
 *
 * The training cab be done using the following command line:
 * >./slm_basic_train_ex --train --shuffle
 *
 * After this phase, the model achieves perfect prediction accuracy (i.e acc=1).
 * The generation option produces text that is very close to the original training data,
 * as illustrated by the example below:
 * > Generated text:
 * > QUEEN ELIZABETH:
 * > But thou didst kill my children.
 * >
 * > KING RICHARD III:
 * > But in your daughter's womb I bury them:
 * > Where in that nest of spicery they shall breed
 * > Selves of themselves, to your recomforture.
 * >
 * > QUEEN ELIZABETH:
 * > Shall I go win my daughter to thy will?
 * >
 * > KING RICHARD III:
 * > And be a happy mother by the deed.
 * >
 * > QUEEN ELIZABETH:
 * > I go. Write to me very shortly.
 * > And you shall understand from me her mind.
 */
