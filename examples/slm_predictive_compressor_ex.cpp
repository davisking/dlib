/*
    @file slm_predictive_compressor_ex.cpp
    @brief Byte-level predictive compression using Transformer-based prediction

    This example demonstrates an advanced application of AI generative models beyond
    traditional chatbot use cases. It implements a compression/decompression system
    for any file type using a small Transformer model to predict the next byte.

    The compression scheme uses a single bit to indicate prediction success, reducing
    data size when the predictor is accurate. This showcases how AI can be applied to
    practical optimization problems like data compression.

    The model uses direct byte values (0-255) as token IDs for embeddings, making it
    reusable and continuously improvable across different files. For unpredicted bytes,
    a file-specific vocabulary enables optimal bit encoding (unless vocabulary requires
    8 bits, in which case direct 8-bit encoding is used without vocabulary storage).

    Key Features:
    - Adaptive learning: model can be trained or fine-tuned with each compression
    - Flexible embedding: model can be embedded in compressed file or stored separately
    - Universal compression: works with any file type (text, binary, executables, etc.)
    - Incremental improvement: each compression can improve the predictor for future use

    Compression format:
    - Header: magic number "DLIB", vocabulary size, vocabulary (if size < 256),
      original size, CRC32, compressed model size (0 if not embedded)
    - Compressed serialized model (if embedded, using dlib compress_stream)
    - Initial window (16 bytes)
    - Compressed body size
    - Compressed body (using dlib compress_stream on bit-encoded stream)

    Usage:
    --compress --input <file> --output <file> : Compress file (default: train + embed model)
    --compress --input <file> --output <file> --no-train : Compress without training
    --compress --input <file> --output <file> --no-embed-model : Don't embed model in output
    --decompress --input <file> --output <file> : Decompress file
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <chrono>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/crc32.h>
#include <dlib/compress_stream.h>

using namespace std;
using namespace dlib;

// Use maximum compression level
typedef dlib::compress_stream::kernel_1ec stream_compressor;

// Constants
const uint32_t MAGIC_NUMBER_DLIB = 0x444C4942; // "DLIB" in big-endian

const int WINDOW_SIZE = 16;                 // Prediction window size
const long MAX_VOCAB_SIZE = 257;            // 256 byte values + 1 PAD token
const int PAD_TOKEN = 256;                  // Padding token
const std::string MODEL_SAVE_FILE = "dlib_predictive_compressor.dat";
const uint32_t FULL_VOCAB_MARKER = 256;     // Marker for full 8-bit vocabulary
const size_t MAX_TRAINING_TOKENS = 5000000; // Maximum tokens for training

// Network architecture parameters
const long NUM_LAYERS = 1;
const long NUM_HEADS = 4;
const long EMBEDDING_DIM = 16;

// ========================================================================================
// Helper Functions
// ========================================================================================

// Calculate minimum number of bits needed to represent vocab_size values
int calculate_bits_per_byte(int vocab_size)
{
    if (vocab_size <= 1)
        return 1;
    return static_cast<int>(std::ceil(std::log2(vocab_size)));
}

// Format duration in hours, minutes, seconds
std::string format_duration(double seconds)
{
    int hours = static_cast<int>(seconds / 3600);
    int minutes = static_cast<int>((seconds - hours * 3600) / 60);
    int secs = static_cast<int>(seconds - hours * 3600 - minutes * 60);

    std::ostringstream oss;
    if (hours > 0)
        oss << hours << "h " << minutes << "m " << secs << "s";
    else if (minutes > 0)
        oss << minutes << "m " << secs << "s";
    else
        oss << secs << "s";

    return oss.str();
}

// Calculate max epochs for the trainer based on dataset size
size_t calculate_max_epochs(size_t num_samples)
{
    const size_t MIN_TRAINING_TOKENS = 20000;
    double S = static_cast<double>(std::max(num_samples, MIN_TRAINING_TOKENS));
    double S_min = static_cast<double>(MIN_TRAINING_TOKENS);
    double S_max = static_cast<double>(MAX_TRAINING_TOKENS);
    double V_min = 1.0;
    double V_max = 50.0;

    S = std::max(S_min, std::min(S, S_max));

    double log_range = std::log(S_max) - std::log(S_min);
    double log_value = std::log(S) - std::log(S_min);
    double normalized_log = log_value / log_range;
    double V = V_max - (normalized_log * (V_max - V_min));

    return static_cast<size_t>(std::ceil(V));
}

// Calculate throughput in bytes/s
double calculate_throughput(size_t bytes, double seconds)
{
    return (bytes / (seconds + 1e-8));
}

// Format throughput for display
std::string format_throughput(double bytes_per_sec)
{
    std::ostringstream oss;
    if (bytes_per_sec >= 1048576.0)
        oss << std::fixed << std::setprecision(2) << (bytes_per_sec / 1048576.0) << " MB/s";
    else if (bytes_per_sec >= 1024.0)
        oss << std::fixed << std::setprecision(2) << (bytes_per_sec / 1024.0) << " KB/s";
    else
        oss << std::fixed << std::setprecision(0) << bytes_per_sec << " B/s";
    return oss.str();
}

// Display progress bar
void show_progress(size_t current, size_t total, const std::string& prefix = "Progress")
{
    if (total == 0) return;

    const int bar_width = 30;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    cout << "\r" << prefix << ": [";
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "%" << std::flush;

    if (current >= total)
        cout << "\n";
}

// ========================================================================================
// Bit stream writer/reader
// ========================================================================================

class bit_stream_writer
{
private:
    std::vector<uint8_t>& buffer;
    uint8_t current_byte;
    int bit_position;

public:
    bit_stream_writer(std::vector<uint8_t>& buf)
        : buffer(buf), current_byte(0), bit_position(0) {
    }

    void write_bit(bool bit)
    {
        if (bit)
            current_byte |= (1 << bit_position);

        bit_position++;
        if (bit_position == 8)
        {
            buffer.push_back(current_byte);
            current_byte = 0;
            bit_position = 0;
        }
    }

    void write_bits(uint8_t value, int num_bits)
    {
        for (int i = 0; i < num_bits; ++i)
            write_bit((value >> i) & 1);
    }

    void flush()
    {
        if (bit_position > 0)
            buffer.push_back(current_byte);
    }
};

class bit_stream_reader
{
private:
    const std::vector<uint8_t>& buffer;
    size_t byte_position;
    int bit_position;

public:
    bit_stream_reader(const std::vector<uint8_t>& buf)
        : buffer(buf), byte_position(0), bit_position(0) {
    }

    bool read_bit()
    {
        if (byte_position >= buffer.size())
            throw std::runtime_error("End of bit stream reached");

        bool bit = (buffer[byte_position] >> bit_position) & 1;
        bit_position++;
        if (bit_position == 8)
        {
            bit_position = 0;
            byte_position++;
        }
        return bit;
    }

    uint8_t read_bits(int num_bits)
    {
        uint8_t value = 0;
        for (int i = 0; i < num_bits; ++i)
            if (read_bit())
                value |= (1 << i);
        return value;
    }
};

// ========================================================================================
// Vocabulary (for optimal encoding of unpredicted bytes only)
// ========================================================================================

class vocabulary
{
private:
    std::vector<uint8_t> index_to_byte;
    std::map<uint8_t, int> byte_to_index;

public:
    // Build vocabulary from binary data
    void build_from_data(const std::string& data)
    {
        std::set<uint8_t> unique_bytes;
        for (size_t i = 0; i < data.size(); ++i)
            unique_bytes.insert(static_cast<uint8_t>(data[i]));

        index_to_byte.assign(unique_bytes.begin(), unique_bytes.end());
        std::sort(index_to_byte.begin(), index_to_byte.end());

        byte_to_index.clear();
        for (size_t i = 0; i < index_to_byte.size(); ++i)
            byte_to_index[index_to_byte[i]] = static_cast<int>(i);
    }

    int size() const { return static_cast<int>(index_to_byte.size()); }

    int byte_to_compact_index(uint8_t byte_val) const
    {
        auto it = byte_to_index.find(byte_val);
        return (it != byte_to_index.end()) ? it->second : 0;
    }

    uint8_t compact_index_to_byte(int index) const
    {
        return (index >= 0 && index < size()) ? index_to_byte[index] : 0;
    }

    std::string serialize() const
    {
        std::string result;
        result.reserve(index_to_byte.size());
        for (uint8_t b : index_to_byte)
            result.push_back(static_cast<char>(b));
        return result;
    }

    void deserialize(const std::string& data)
    {
        index_to_byte.clear();
        byte_to_index.clear();
        index_to_byte.reserve(data.size());

        for (size_t i = 0; i < data.size(); ++i)
            index_to_byte.push_back(static_cast<uint8_t>(data[i]));

        for (size_t i = 0; i < index_to_byte.size(); ++i)
            byte_to_index[index_to_byte[i]] = static_cast<int>(i);
    }
};

// ========================================================================================
// Tokenization
// ========================================================================================

std::vector<int> tokenize(const std::string& data)
{
    std::vector<int> tokens;
    tokens.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        tokens.push_back(static_cast<int>(static_cast<uint8_t>(data[i])));
    return tokens;
}

// ========================================================================================
// Predictor network
// ========================================================================================

template<int vocab_size>
using train_predictor =
loss_multiclass_log<fc<vocab_size, rms_norm<
    fused_transformer::transformer_stack<NUM_LAYERS, gelu, dropout_10, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS,
    token_embeddings<vocab_size, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>;

template<int vocab_size>
using infer_predictor =
loss_multiclass_log<fc<vocab_size, rms_norm<
    fused_transformer::transformer_stack<NUM_LAYERS, gelu, multiply, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS,
    token_embeddings<vocab_size, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>;

// ========================================================================================
// Compression function
// ========================================================================================

void compress_file(const std::string& input_path, const std::string& output_path,
    bool train_model, bool embed_model)
{
    cout << "=== COMPRESSION MODE ===\n";
    cout << "Input file: " << input_path << "\n";
    cout << "Output file: " << output_path << "\n";
    cout << "Training: " << (train_model ? "enabled" : "disabled") << "\n";
    cout << "Model embedding: " << (embed_model ? "enabled" : "disabled") << "\n\n";

    // Read input file in binary mode
    std::ifstream input(input_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open input file");

    std::string file_data;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    file_data.resize(file_size);
    input.read(&file_data[0], file_size);
    input.close();

    cout << "Original file size: " << file_data.size() << " bytes\n";
    if (file_data.size() == 0) {
        cout << "Input file is empty\n";
        return;
    }

    // Build vocabulary
    vocabulary vocab;
    vocab.build_from_data(file_data);

    int bits_per_byte = calculate_bits_per_byte(vocab.size());
    bool use_full_vocab = (bits_per_byte >= 8);

    cout << "Vocabulary size: " << vocab.size() << " unique byte values\n";
    if (use_full_vocab)
        cout << "Using direct 8-bit encoding (vocabulary not stored)\n";
    else
        cout << "Bits per unpredicted byte: " << bits_per_byte << "\n";

    // Tokenize
    std::vector<int> tokens = tokenize(file_data);

    // Prepare training data if needed
    train_predictor<MAX_VOCAB_SIZE> net;
    bool model_exists = file_exists(MODEL_SAVE_FILE);

    if (train_model || !model_exists)
    {
        std::vector<int> tokens_for_training;
        if (tokens.size() > MAX_TRAINING_TOKENS) {
            tokens_for_training.assign(tokens.begin(), tokens.begin() + MAX_TRAINING_TOKENS);
            cout << "Limiting training to first " << MAX_TRAINING_TOKENS << " bytes\n";
        }
        else {
            tokens_for_training = tokens;
        }

        cout << "\nTraining predictor model...\n";

        std::vector<matrix<int, 0, 1>> samples;
        std::vector<unsigned long> labels;

        build_single_token_prediction_dataset(
            std::vector<std::vector<int>>{tokens_for_training},
            WINDOW_SIZE,
            PAD_TOKEN,
            false,
            samples,
            labels
        );

        cout << "Training samples: " << samples.size() << "\n";

        if (model_exists) {
            cout << "Loading pre-trained model: " << MODEL_SAVE_FILE << endl;
            deserialize(MODEL_SAVE_FILE) >> net;
            cout << "Performing fine-tuning\n" << endl;
        }
        else {
            cout << "Training new model from scratch\n" << endl;
        }

        dnn_trainer<train_predictor<MAX_VOCAB_SIZE>, adam> trainer(net, adam(0.004, 0.9, 0.999));
        trainer.set_learning_rate(1e-3);
        trainer.set_min_learning_rate(1e-6);
        trainer.set_mini_batch_size(128);
        trainer.set_iterations_without_progress_threshold(15000);

        size_t max_num_epochs = calculate_max_epochs(samples.size());
        cout << "Max epochs: " << max_num_epochs << endl;
        trainer.set_max_num_epochs(max_num_epochs);
        trainer.be_verbose();
        trainer.train(samples, labels);

        auto predicted = net(samples);
        size_t correct = 0;
        for (size_t i = 0; i < labels.size(); ++i)
            if (predicted[i] == labels[i]) correct++;

        double accuracy = static_cast<double>(correct) / labels.size();
        cout << "Predictor accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%\n";

        net.clean();
        serialize(MODEL_SAVE_FILE) << net;
        cout << "Model saved to " << MODEL_SAVE_FILE << "\n";
    }
    else
    {
        cout << "\nLoading existing model without training: " << MODEL_SAVE_FILE << "\n";
        deserialize(MODEL_SAVE_FILE) >> net;
        net.clean();
    }

    // Convert to inference mode
    infer_predictor<MAX_VOCAB_SIZE> infer_net;
    infer_net = net;

    // Prepare model data for embedding if requested
    std::string compressed_model;
    if (embed_model)
    {
        cout << "\nSerializing model for embedding...\n";
        std::ostringstream model_stream;
        serialize(infer_net, model_stream);
        std::string serialized_model = model_stream.str();
        cout << "Serialized model size: " << serialized_model.size() << " bytes\n";

        cout << "Compressing model...\n";
        std::istringstream model_input(serialized_model);
        std::ostringstream model_compressed;

        stream_compressor model_comp;
        model_comp.compress(model_input, model_compressed);

        compressed_model = model_compressed.str();
        cout << "Compressed model size: " << compressed_model.size() << " bytes";
        double model_compression_ratio = (1.0 - static_cast<double>(compressed_model.size()) / serialized_model.size()) * 100.0;
        cout << " (saved " << std::fixed << std::setprecision(1) << model_compression_ratio << "%)\n";
    }
    else
    {
        cout << "\nModel will not be embedded in compressed file\n";
    }

    // Compress data
    auto compression_start = std::chrono::high_resolution_clock::now();

    cout << "\nCompressing data...\n";
    std::vector<uint8_t> compressed_data;
    bit_stream_writer writer(compressed_data);

    inference_context ctx(WINDOW_SIZE, 1, PAD_TOKEN);

    size_t predictions_correct = 0;
    size_t predictions_total = 0;

    for (int i = 0; i < std::min(WINDOW_SIZE, static_cast<int>(tokens.size())); ++i)
        ctx.add_token(tokens[i]);

    size_t tokens_to_compress = tokens.size() - WINDOW_SIZE;
    for (size_t i = WINDOW_SIZE; i < tokens.size(); ++i)
    {
        if ((i - WINDOW_SIZE) % 1000 == 0 || i == tokens.size() - 1)
            show_progress(i - WINDOW_SIZE + 1, tokens_to_compress, "Compressing");

        auto input_seq = ctx.get_input_window();
        unsigned long predicted_token = infer_net(input_seq);

        predictions_total++;
        if (predicted_token == static_cast<unsigned long>(tokens[i]))
        {
            writer.write_bit(true);
            predictions_correct++;
        }
        else
        {
            writer.write_bit(false);
            uint8_t byte_val = static_cast<uint8_t>(tokens[i]);

            if (use_full_vocab)
                writer.write_bits(byte_val, 8);
            else
            {
                int compact_index = vocab.byte_to_compact_index(byte_val);
                writer.write_bits(static_cast<uint8_t>(compact_index), bits_per_byte);
            }
        }

        ctx.add_token(tokens[i]);
    }

    writer.flush();

    auto compression_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compression_duration = compression_end - compression_start;
    double throughput = calculate_throughput(file_data.size(), compression_duration.count());

    cout << "Prediction success ratio: " << std::fixed << std::setprecision(2)
        << (predictions_correct * 100.0 / predictions_total) << "%\n";
    cout << "Uncompressed body size: " << compressed_data.size() << " bytes\n";
    cout << "Compression time: " << format_duration(compression_duration.count())
        << " (" << format_throughput(throughput) << ")\n";

    // Apply final compression
    cout << "\nApplying final compression...\n";
    std::string uncompressed_body(compressed_data.begin(), compressed_data.end());
    std::istringstream body_input(uncompressed_body);
    std::ostringstream body_compressed;

    stream_compressor body_comp;
    body_comp.compress(body_input, body_compressed);

    std::string final_compressed_body = body_compressed.str();
    cout << "Final compressed body size: " << final_compressed_body.size() << " bytes";
    double body_compression_ratio = (1.0 - static_cast<double>(final_compressed_body.size()) / compressed_data.size()) * 100.0;
    cout << " (saved " << std::fixed << std::setprecision(1) << body_compression_ratio << "%)\n";

    uint32_t checksum = dlib::crc32(file_data);

    // Write output file
    std::ofstream output(output_path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Cannot create output file");

    // Write header
    output.write(reinterpret_cast<const char*>(&MAGIC_NUMBER_DLIB), sizeof(MAGIC_NUMBER_DLIB));

    uint32_t vocab_size_u32 = use_full_vocab ? FULL_VOCAB_MARKER : vocab.size();
    output.write(reinterpret_cast<const char*>(&vocab_size_u32), sizeof(vocab_size_u32));

    if (!use_full_vocab)
    {
        std::string vocab_data = vocab.serialize();
        output.write(vocab_data.data(), vocab_data.size());
    }

    uint64_t original_size = file_data.size();
    output.write(reinterpret_cast<const char*>(&original_size), sizeof(original_size));

    output.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));

    // Write model size (0 if not embedded)
    uint64_t model_size = embed_model ? compressed_model.size() : 0;
    output.write(reinterpret_cast<const char*>(&model_size), sizeof(model_size));

    if (embed_model)
        output.write(compressed_model.data(), compressed_model.size());

    // Write initial window
    for (int i = 0; i < std::min(WINDOW_SIZE, static_cast<int>(tokens.size())); ++i)
    {
        uint8_t token_byte = static_cast<uint8_t>(tokens[i]);
        output.write(reinterpret_cast<const char*>(&token_byte), 1);
    }

    // Write compressed body
    uint64_t body_size = final_compressed_body.size();
    output.write(reinterpret_cast<const char*>(&body_size), sizeof(body_size));
    output.write(final_compressed_body.data(), final_compressed_body.size());

    std::streampos final_size = output.tellp();
    output.close();

    cout << "\nFinal compressed file size: " << final_size << " bytes\n";
    cout << "Overall compression ratio: " << std::fixed << std::setprecision(2)
        << ((1.0 - static_cast<double>(final_size) / file_data.size()) * 100.0) << "%\n";
    cout << "Compression complete!\n";
}

// ========================================================================================
// Decompression function
// ========================================================================================

void decompress_file(const std::string& input_path, const std::string& output_path)
{
    cout << "=== DECOMPRESSION MODE ===\n";
    cout << "Input file: " << input_path << "\n";
    cout << "Output file: " << output_path << "\n\n";

    std::ifstream input(input_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open input file");

    // Read and verify magic number
    uint32_t magic;
    input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC_NUMBER_DLIB)
        throw std::runtime_error("Invalid file format (bad magic number)");

    // Read vocabulary
    uint32_t vocab_size_u32;
    input.read(reinterpret_cast<char*>(&vocab_size_u32), sizeof(vocab_size_u32));

    bool use_full_vocab = (vocab_size_u32 == FULL_VOCAB_MARKER);
    int bits_per_byte = use_full_vocab ? 8 : calculate_bits_per_byte(vocab_size_u32);

    vocabulary vocab;

    if (!use_full_vocab)
    {
        std::string vocab_data;
        vocab_data.resize(vocab_size_u32);
        input.read(&vocab_data[0], vocab_size_u32);
        vocab.deserialize(vocab_data);
        cout << "Vocabulary size: " << vocab.size() << " byte values\n";
        cout << "Bits per unpredicted byte: " << bits_per_byte << "\n";
    }
    else
    {
        cout << "Using direct 8-bit encoding\n";
    }

    // Read original size and CRC
    uint64_t original_size;
    input.read(reinterpret_cast<char*>(&original_size), sizeof(original_size));

    uint32_t stored_crc;
    input.read(reinterpret_cast<char*>(&stored_crc), sizeof(stored_crc));

    cout << "Original file size: " << original_size << " bytes\n";

    // Read model size
    uint64_t compressed_model_size;
    input.read(reinterpret_cast<char*>(&compressed_model_size), sizeof(compressed_model_size));

    infer_predictor<MAX_VOCAB_SIZE> infer_net;

    if (compressed_model_size > 0)
    {
        // Model is embedded in file
        cout << "Compressed model size: " << compressed_model_size << " bytes\n";

        if (compressed_model_size > 10000000)
            throw std::runtime_error("Invalid compressed model size");

        std::vector<char> compressed_model_buffer(compressed_model_size);
        input.read(compressed_model_buffer.data(), compressed_model_size);

        if (!input.good())
            throw std::runtime_error("Failed to read compressed model");

        cout << "Decompressing embedded model...\n";
        std::string compressed_model(compressed_model_buffer.begin(), compressed_model_buffer.end());
        std::istringstream compressed_input(compressed_model);
        std::ostringstream decompressed_output;

        stream_compressor model_decomp;
        model_decomp.decompress(compressed_input, decompressed_output);

        std::string serialized_model = decompressed_output.str();
        cout << "Decompressed model size: " << serialized_model.size() << " bytes\n";

        cout << "Loading model...\n";
        try {
            std::istringstream model_stream(serialized_model);
            deserialize(infer_net, model_stream);
            cout << "Model loaded successfully\n";
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to deserialize model: ") + e.what());
        }
    }
    else
    {
        // Model not embedded, load from file
        cout << "Model not embedded in compressed file\n";
        cout << "Loading model from: " << MODEL_SAVE_FILE << "\n";

        if (!file_exists(MODEL_SAVE_FILE))
            throw std::runtime_error("Model file not found: " + MODEL_SAVE_FILE);

        try {
            deserialize(MODEL_SAVE_FILE) >> infer_net;
            cout << "Model loaded successfully\n";
        }
        catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to load model: ") + e.what());
        }
    }

    // Read initial window
    std::vector<int> initial_tokens;
    int window_size = std::min(WINDOW_SIZE, static_cast<int>(original_size));
    for (int i = 0; i < window_size; ++i)
    {
        uint8_t token_byte;
        input.read(reinterpret_cast<char*>(&token_byte), 1);
        if (!input.good())
            throw std::runtime_error("Failed to read initial window");
        initial_tokens.push_back(static_cast<int>(token_byte));
    }

    // Read compressed body
    uint64_t compressed_body_size;
    input.read(reinterpret_cast<char*>(&compressed_body_size), sizeof(compressed_body_size));
    cout << "Compressed body size: " << compressed_body_size << " bytes\n";

    if (compressed_body_size == 0)
        throw std::runtime_error("Invalid compressed body size");

    std::vector<char> compressed_body_buffer(compressed_body_size);
    input.read(compressed_body_buffer.data(), compressed_body_size);
    input.close();

    if (!input.good())
        throw std::runtime_error("Failed to read compressed body");

    // Decompress body
    cout << "Decompressing body...\n";
    std::string compressed_body(compressed_body_buffer.begin(), compressed_body_buffer.end());
    std::istringstream body_compressed_input(compressed_body);
    std::ostringstream body_decompressed_output;

    stream_compressor body_decomp;
    body_decomp.decompress(body_compressed_input, body_decompressed_output);

    std::string decompressed_body = body_decompressed_output.str();
    cout << "Decompressed body size: " << decompressed_body.size() << " bytes\n";

    std::vector<uint8_t> compressed_data(decompressed_body.begin(), decompressed_body.end());

    // Decompress data
    inference_context ctx(WINDOW_SIZE, 1, PAD_TOKEN);
    for (int token : initial_tokens)
        ctx.add_token(token);

    auto decompression_start = std::chrono::high_resolution_clock::now();

    cout << "\nDecompressing data...\n";
    std::string decompressed_data;
    decompressed_data.reserve(original_size);

    for (int token : initial_tokens)
        decompressed_data.push_back(static_cast<char>(token));

    if (compressed_data.size() > 0)
    {
        bit_stream_reader reader(compressed_data);

        try
        {
            size_t bytes_to_decompress = original_size - window_size;
            while (decompressed_data.size() < original_size)
            {
                size_t current_pos = decompressed_data.size() - window_size;
                if (current_pos % 1000 == 0 || decompressed_data.size() == original_size - 1)
                    show_progress(current_pos + 1, bytes_to_decompress, "Decompressing");

                bool prediction_success = reader.read_bit();

                int next_token;
                if (prediction_success)
                {
                    auto input_seq = ctx.get_input_window();
                    next_token = infer_net(input_seq);
                }
                else
                {
                    if (use_full_vocab)
                        next_token = reader.read_bits(8);
                    else
                    {
                        int compact_index = reader.read_bits(bits_per_byte);
                        next_token = static_cast<int>(vocab.compact_index_to_byte(compact_index));
                    }
                }

                decompressed_data.push_back(static_cast<char>(next_token));
                ctx.add_token(next_token);
            }
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(std::string("Decompression error: ") + e.what());
        }
    }

    auto decompression_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompression_duration = decompression_end - decompression_start;
    double throughput = calculate_throughput(decompressed_data.size(), decompression_duration.count());

    cout << "Decompression time: " << format_duration(decompression_duration.count())
        << " (" << format_throughput(throughput) << ")\n";

    // Verify
    if (decompressed_data.size() != original_size)
    {
        cerr << "Warning: Decompressed size (" << decompressed_data.size()
            << ") differs from expected (" << original_size << ")\n";
    }

    uint32_t computed_crc = dlib::crc32(decompressed_data);
    if (computed_crc != stored_crc)
    {
        cerr << "Warning: CRC mismatch! File may be corrupted.\n";
        cerr << "Expected: 0x" << std::hex << stored_crc << "\n";
        cerr << "Computed: 0x" << std::hex << computed_crc << "\n";
    }
    else
    {
        cout << "CRC verification: OK\n";
    }

    // Write output
    std::ofstream output(output_path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Cannot create output file");

    output.write(decompressed_data.data(), decompressed_data.size());
    output.close();

    cout << "\nDecompression complete!\n";
    cout << "Output file size: " << decompressed_data.size() << " bytes\n";
}

// ========================================================================================
// Main function
// ========================================================================================

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("compress", "Compress any file using AI predictive encoding");
        parser.add_option("decompress", "Decompress a compressed file");
        parser.add_option("input", "Input file path", 1);
        parser.add_option("output", "Output file path", 1);
        parser.add_option("no-train", "Skip model training/fine-tuning (use existing model as-is)");
        parser.add_option("no-embed-model", "Don't embed model in compressed file (requires external model file for decompression)");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("compress") && !parser.option("decompress"))
        {
            cout << "This example demonstrates how AI generative models can be applied to\n";
            cout << "practical optimization problems beyond traditional chatbot applications.\n";
            cout << "It uses a small Transformer model to predict and compress data.\n\n";
            parser.print_options();
            cout << "\nExample usage:\n";
            cout << "  Compress with training:   " << argv[0] << " --compress --input file.bin --output file.dlib\n";
            cout << "  Compress without training:" << argv[0] << " --compress --input file.bin --output file.dlib --no-train\n";
            cout << "  Compress without embed:   " << argv[0] << " --compress --input file.bin --output file.dlib --no-embed-model\n";
            cout << "  Decompress:               " << argv[0] << " --decompress --input file.dlib --output file.bin\n";
            cout << "\nNotes:\n";
            cout << "  - Model uses direct byte values (0-255) as embeddings, reusable across files\n";
            cout << "  - Model saved to '" << MODEL_SAVE_FILE << "' for continuous improvement\n";
            cout << "  - Use --no-train to compress faster with existing model\n";
            cout << "  - Use --no-embed-model for smaller compressed files (requires model file separately)\n";
            return 0;
        }

        const std::string input_file = get_option(parser, "input", "");
        const std::string output_file = get_option(parser, "output", "");

        if (input_file.empty() || output_file.empty())
        {
            cerr << "Error: Both --input and --output options are required\n";
            return 1;
        }

        if (parser.option("compress"))
        {
            bool train_model = !parser.option("no-train");
            bool embed_model = !parser.option("no-embed-model");
            compress_file(input_file, output_file, train_model, embed_model);
        }
        else if (parser.option("decompress"))
        {
            decompress_file(input_file, output_file);
        }

        return 0;
    }
    catch (std::exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}