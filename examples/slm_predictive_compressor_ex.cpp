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
    - File type detection: automatically detects text vs binary for optimized processing

    Compression format:
    - Header: magic number "DLIB", file type flag (text/binary), vocabulary size,
      vocabulary (if size < 256), original size, CRC32, compressed model size (0 if not embedded)
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
const uint32_t MAGIC_NUMBER = 0x444C4942;   // "DLIB" in big-endian
const uint8_t FILE_TYPE_BINARY = 0x00;      // Binary file marker
const uint8_t FILE_TYPE_TEXT = 0x01;        // Text file marker

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

// Format throughput
std::string format_throughput(double bytes_per_sec)
{
    if (bytes_per_sec >= 1048576)
        return std::to_string(static_cast<int>(bytes_per_sec / 1048576)) + " MB/s";
    else if (bytes_per_sec >= 1024)
        return std::to_string(static_cast<int>(bytes_per_sec / 1024)) + " KB/s";
    else
        return std::to_string(static_cast<int>(bytes_per_sec)) + " B/s";
}

// Calculate throughput
double calculate_throughput(size_t bytes, double seconds)
{
    if (seconds == 0.0)
        return 0.0;
    return static_cast<double>(bytes) / seconds;
}

// Format compression ratio with sign
std::string format_compression_ratio(size_t compressed, size_t original)
{
    if (original == 0)
        return "N/A";

    double ratio = ((1.0 - static_cast<double>(compressed) / original) * 100.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (ratio >= 0)
        oss << "-" << ratio << "%";
    else
        oss << "+" << ratio << "%";

    return oss.str();
}

// Show progress
void show_progress(size_t current, size_t total, const std::string& label)
{
    double percent = (static_cast<double>(current) / total) * 100.0;
    std::cout << "\r" << label << ": " << current << "/" << total
        << " (" << std::fixed << std::setprecision(1) << percent << "%)     " << std::flush;
    if (current == total)
        std::cout << "\n";
}

// Calculate max epochs for the trainer based on dataset size
size_t calculate_max_epochs(size_t num_samples)
{
    const size_t MIN_TRAINING_TOKENS = 5000;
    double S = static_cast<double>(std::max(num_samples, MIN_TRAINING_TOKENS));
    double S_min = static_cast<double>(MIN_TRAINING_TOKENS);
    double S_max = static_cast<double>(MAX_TRAINING_TOKENS);
    double V_min = 1.0;
    double V_max = 100.0;

    S = std::max(S_min, std::min(S, S_max));

    double log_range = std::log(S_max) - std::log(S_min);
    double log_value = std::log(S) - std::log(S_min);
    double normalized_log = log_value / log_range;
    double V = V_max - (normalized_log * (V_max - V_min));

    return static_cast<size_t>(std::ceil(V));
}

// ========================================================================================
// Bit Stream Classes
// ========================================================================================

class bit_stream_writer
{
public:
    bit_stream_writer(std::vector<uint8_t>& output) : output_(output), current_byte_(0), bit_pos_(0)
    {
    }

    void write_bit(bool bit)
    {
        if (bit)
            current_byte_ |= (1 << (7 - bit_pos_));
        bit_pos_++;
        if (bit_pos_ == 8)
        {
            output_.push_back(current_byte_);
            current_byte_ = 0;
            bit_pos_ = 0;
        }
    }

    void write_bits(uint8_t value, int num_bits)
    {
        for (int i = num_bits - 1; i >= 0; --i)
            write_bit((value >> i) & 1);
    }

    void flush()
    {
        if (bit_pos_ > 0)
        {
            output_.push_back(current_byte_);
            current_byte_ = 0;
            bit_pos_ = 0;
        }
    }

private:
    std::vector<uint8_t>& output_;
    uint8_t current_byte_;
    int bit_pos_;
};

class bit_stream_reader
{
public:
    bit_stream_reader(const std::vector<uint8_t>& data) : data_(data), byte_pos_(0), bit_pos_(0)
    {
    }

    bool read_bit()
    {
        if (byte_pos_ >= data_.size())
            throw std::runtime_error("Unexpected end of bit stream");

        bool bit = (data_[byte_pos_] >> (7 - bit_pos_)) & 1;
        bit_pos_++;
        if (bit_pos_ == 8)
        {
            bit_pos_ = 0;
            byte_pos_++;
        }
        return bit;
    }

    uint8_t read_bits(int num_bits)
    {
        uint8_t value = 0;
        for (int i = 0; i < num_bits; ++i)
            value = (value << 1) | (read_bit() ? 1 : 0);
        return value;
    }

private:
    const std::vector<uint8_t>& data_;
    size_t byte_pos_;
    int bit_pos_;
};

// ========================================================================================
// Vocabulary Management
// ========================================================================================

class vocabulary
{
public:
    vocabulary() = default;

    void build(const std::vector<uint8_t>& data)
    {
        std::set<uint8_t> unique_bytes(data.begin(), data.end());
        sorted_bytes_.assign(unique_bytes.begin(), unique_bytes.end());
        std::sort(sorted_bytes_.begin(), sorted_bytes_.end());

        byte_to_index_.clear();
        for (size_t i = 0; i < sorted_bytes_.size(); ++i)
            byte_to_index_[sorted_bytes_[i]] = i;
    }

    int byte_to_compact_index(uint8_t byte) const
    {
        auto it = byte_to_index_.find(byte);
        if (it == byte_to_index_.end())
            throw std::runtime_error("Byte not in vocabulary");
        return it->second;
    }

    uint8_t compact_index_to_byte(int index) const
    {
        if (index < 0 || index >= static_cast<int>(sorted_bytes_.size()))
            throw std::runtime_error("Index out of range");
        return sorted_bytes_[index];
    }

    size_t size() const { return sorted_bytes_.size(); }

    std::string serialize() const
    {
        return std::string(sorted_bytes_.begin(), sorted_bytes_.end());
    }

    void deserialize(const std::string& data)
    {
        sorted_bytes_.assign(data.begin(), data.end());
        byte_to_index_.clear();
        for (size_t i = 0; i < sorted_bytes_.size(); ++i)
            byte_to_index_[sorted_bytes_[i]] = i;
    }

private:
    std::vector<uint8_t> sorted_bytes_;
    std::map<uint8_t, int> byte_to_index_;
};

// ========================================================================================
// Neural Network Architecture
// ========================================================================================

template<int vocab_size>
using train_predictor =
loss_multiclass_log<fc<vocab_size, fc<EMBEDDING_DIM, rms_norm<
    fused_transformer::transformer_stack<NUM_LAYERS, gelu, dropout_10, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS,
    token_embeddings<vocab_size, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>>;

template<int vocab_size>
using infer_predictor =
loss_multiclass_log<fc<vocab_size, fc<EMBEDDING_DIM, rms_norm<
    fused_transformer::transformer_stack<NUM_LAYERS, gelu, multiply, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS,
    token_embeddings<vocab_size, EMBEDDING_DIM, input<matrix<int, 0, 1>>>>>>>>;

// ========================================================================================
// Compression function
// ========================================================================================

void compress_file(const std::string& input_path, const std::string& output_path,
    bool train_model, bool embed_model)
{
    cout << "=== COMPRESSION MODE ===\n";
    cout << "Input file: " << input_path << "\n";
    cout << "Output file: " << output_path << "\n";
    cout << "Training: " << (train_model ? "Yes" : "No") << "\n";
    cout << "Embed model: " << (embed_model ? "Yes" : "No") << "\n\n";

    // Detect file type
    cout << "Detecting file type...\n";
    file_content_type detected_type;
    bool is_text = detect_file_type(input_path, detected_type);

    cout << "File type detected: ";
    if (is_text)
    {
        cout << "TEXT";
        if (detected_type == file_content_type::TEXT_XML)
            cout << " (XML/HTML)";
    }
    else
    {
        cout << "BINARY";
        switch (detected_type)
        {
        case file_content_type::IMAGE: cout << " (Image)"; break;
        case file_content_type::VIDEO: cout << " (Video)"; break;
        case file_content_type::AUDIO: cout << " (Audio)"; break;
        case file_content_type::EXECUTABLE: cout << " (Executable)"; break;
        case file_content_type::COMPRESSED: cout << " (Compressed Archive)"; break;
        case file_content_type::PDF: cout << " (PDF)"; break;
        case file_content_type::OFFICE: cout << " (Office Document)"; break;
        case file_content_type::UNKNOWN: cout << " (Unknown Format)"; break;
        default: break;
        }
    }
    cout << "\n\n";

    uint8_t file_type_flag = is_text ? FILE_TYPE_TEXT : FILE_TYPE_BINARY;

    // Load input file
    cout << "Loading input file...\n";
    std::ifstream input_file(input_path, std::ios::binary);
    if (!input_file)
        throw std::runtime_error("Cannot open input file");

    std::vector<uint8_t> file_data((std::istreambuf_iterator<char>(input_file)),
        std::istreambuf_iterator<char>());
    input_file.close();

    cout << "Input file size: " << file_data.size() << " bytes\n";

    if (file_data.empty())
        throw std::runtime_error("Input file is empty");

    const size_t original_size = file_data.size();

    // Apply text-specific preprocessing if needed
    std::vector<uint8_t> preprocessed_data;
    if (is_text)
    {
        cout << "Applying text-specific preprocessing...\n";
        // TODO: Implement text-specific preprocessing here
        // For now, just copy the data
        preprocessed_data = file_data;
    }
    else
    {
        preprocessed_data = file_data;
    }

    // Build vocabulary
    vocabulary vocab;
    vocab.build(preprocessed_data);
    cout << "Unique byte values: " << vocab.size() << "\n";

    bool use_full_vocab = (vocab.size() == 256);
    int bits_per_byte = use_full_vocab ? 8 : calculate_bits_per_byte(vocab.size());

    if (!use_full_vocab)
        cout << "Using optimized vocabulary (" << bits_per_byte << " bits per unpredicted byte)\n";
    else
        cout << "Using direct 8-bit encoding (full vocabulary)\n";

    // Convert to tokens
    std::vector<int> tokens;
    tokens.reserve(preprocessed_data.size());
    for (uint8_t byte : preprocessed_data)
        tokens.push_back(static_cast<int>(byte));

    // Train or load model
    train_predictor<MAX_VOCAB_SIZE> net;
    bool model_exists = file_exists(MODEL_SAVE_FILE);

    if (train_model)
    {
        std::vector<int> tokens_for_training;
        if (tokens.size() > MAX_TRAINING_TOKENS) {
            cout << "\nFile is large (" << tokens.size() << " tokens).\n";
            cout << "Using random " << MAX_TRAINING_TOKENS << " tokens for training.\n";

            dlib::rand rng(std::time(0));
            std::set<size_t> selected_indices;
            while (selected_indices.size() < MAX_TRAINING_TOKENS)
                selected_indices.insert(rng.get_random_32bit_number() % tokens.size());

            tokens_for_training.reserve(selected_indices.size());
            for (size_t idx : selected_indices)
                tokens_for_training.push_back(tokens[idx]);
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
    size_t model_overhead = 0;

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
        model_overhead = compressed_model.size();
        cout << "Compressed model size: " << model_overhead << " bytes "
            << format_compression_ratio(model_overhead, serialized_model.size()) << "\n";
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
    cout << "Encoded body size: " << compressed_data.size() << " bytes "
        << format_compression_ratio(compressed_data.size(), original_size) << "\n";
    cout << "Encoding time: " << format_duration(compression_duration.count())
        << " (" << format_throughput(throughput) << ")\n";

    // Apply final compression
    cout << "\nApplying final compression...\n";
    std::string uncompressed_body(compressed_data.begin(), compressed_data.end());
    std::istringstream body_input(uncompressed_body);
    std::ostringstream body_compressed;

    stream_compressor body_comp;
    body_comp.compress(body_input, body_compressed);

    std::string final_compressed_body = body_compressed.str();
    cout << "Final compressed body: " << final_compressed_body.size() << " bytes "
        << format_compression_ratio(final_compressed_body.size(), compressed_data.size()) << "\n";

    uint32_t checksum = dlib::crc32(std::string(file_data.begin(), file_data.end()));

    // Calculate header overhead
    size_t header_size = sizeof(MAGIC_NUMBER) + sizeof(file_type_flag) + sizeof(uint32_t);
    if (!use_full_vocab)
        header_size += vocab.size();
    header_size += sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t);
    header_size += WINDOW_SIZE + sizeof(uint64_t);

    // Write output file
    std::ofstream output(output_path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Cannot create output file");

    // Write header
    output.write(reinterpret_cast<const char*>(&MAGIC_NUMBER), sizeof(MAGIC_NUMBER));

    // Write file type flag
    output.write(reinterpret_cast<const char*>(&file_type_flag), sizeof(file_type_flag));

    uint32_t vocab_size_u32 = use_full_vocab ? FULL_VOCAB_MARKER : vocab.size();
    output.write(reinterpret_cast<const char*>(&vocab_size_u32), sizeof(vocab_size_u32));

    if (!use_full_vocab)
    {
        std::string vocab_data = vocab.serialize();
        output.write(vocab_data.data(), vocab_data.size());
    }

    uint64_t original_size_u64 = file_data.size();
    output.write(reinterpret_cast<const char*>(&original_size_u64), sizeof(original_size_u64));

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

    cout << "\n=== COMPRESSION SUMMARY ===\n";
    cout << "Original size:       " << original_size << " bytes\n";
    cout << "Compressed size:     " << final_size << " bytes\n";
    cout << "  Header + metadata: " << header_size << " bytes\n";
    if (embed_model)
        cout << "  Embedded model:    " << model_overhead << " bytes\n";
    cout << "  Compressed data:   " << final_compressed_body.size() << " bytes\n";
    cout << "Overall compression: " << format_compression_ratio(static_cast<size_t>(final_size), original_size) << "\n";
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
    if (magic != MAGIC_NUMBER)
        throw std::runtime_error("Invalid file format (bad magic number)");

    // Read file type flag
    uint8_t file_type_flag;
    input.read(reinterpret_cast<char*>(&file_type_flag), sizeof(file_type_flag));

    bool is_text = (file_type_flag == FILE_TYPE_TEXT);
    cout << "File type: " << (is_text ? "TEXT" : "BINARY") << "\n";

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

    // Decompress body
    cout << "\nDecompressing body...\n";
    std::string compressed_body(compressed_body_buffer.begin(), compressed_body_buffer.end());
    std::istringstream body_compressed_input(compressed_body);
    std::ostringstream body_decompressed_output;

    stream_compressor body_decomp;
    body_decomp.decompress(body_compressed_input, body_decompressed_output);

    std::string decompressed_body_str = body_decompressed_output.str();
    std::vector<uint8_t> decompressed_body(decompressed_body_str.begin(), decompressed_body_str.end());

    cout << "Decompressed body size: " << decompressed_body.size() << " bytes\n";

    // Decode data
    auto decompression_start = std::chrono::high_resolution_clock::now();

    cout << "\nDecoding data...\n";
    bit_stream_reader reader(decompressed_body);
    inference_context ctx(WINDOW_SIZE, 1, PAD_TOKEN);

    std::vector<uint8_t> output_data;
    output_data.reserve(original_size);

    for (int token : initial_tokens)
    {
        output_data.push_back(static_cast<uint8_t>(token));
        ctx.add_token(token);
    }

    size_t tokens_to_decode = original_size - window_size;

    try {
        for (size_t i = 0; i < tokens_to_decode; ++i)
        {
            if (i % 1000 == 0 || i == tokens_to_decode - 1)
                show_progress(i + 1, tokens_to_decode, "Decoding");

            bool prediction_correct = reader.read_bit();
            unsigned long token;

            if (prediction_correct)
            {
                auto input_seq = ctx.get_input_window();
                token = infer_net(input_seq);
            }
            else
            {
                if (use_full_vocab)
                {
                    token = reader.read_bits(8);
                }
                else
                {
                    int compact_index = reader.read_bits(bits_per_byte);
                    token = vocab.compact_index_to_byte(compact_index);
                }
            }

            output_data.push_back(static_cast<uint8_t>(token));
            ctx.add_token(token);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Decoding error: ") + e.what());
    }

    auto decompression_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompression_duration = decompression_end - decompression_start;
    double throughput = calculate_throughput(original_size, decompression_duration.count());

    cout << "Decompression time: " << format_duration(decompression_duration.count())
        << " (" << format_throughput(throughput) << ")\n";

    // Apply text-specific postprocessing if needed
    std::vector<uint8_t> final_output_data;
    if (is_text)
    {
        cout << "Applying text-specific postprocessing...\n";
        // TODO: Implement text-specific postprocessing here
        // For now, just copy the data
        final_output_data = output_data;
    }
    else
    {
        final_output_data = output_data;
    }

    // Verify checksum
    uint32_t computed_crc = dlib::crc32(std::string(final_output_data.begin(), final_output_data.end()));
    if (computed_crc != stored_crc)
    {
        cerr << "WARNING: CRC mismatch! Data may be corrupted.\n";
        cerr << "Expected CRC: " << std::hex << stored_crc << "\n";
        cerr << "Computed CRC: " << std::hex << computed_crc << "\n";
    }
    else
    {
        cout << "CRC32 verification: OK\n";
    }

    // Write output file
    std::ofstream output_file(output_path, std::ios::binary);
    if (!output_file)
        throw std::runtime_error("Cannot create output file");

    output_file.write(reinterpret_cast<const char*>(final_output_data.data()), final_output_data.size());
    output_file.close();

    cout << "\nDecompression complete!\n";
    cout << "Output file: " << output_path << " (" << final_output_data.size() << " bytes)\n";
}

// ========================================================================================
// Main
// ========================================================================================

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("compress", "Compress a file");
        parser.add_option("decompress", "Decompress a file");
        parser.add_option("input", "Input file path", 1);
        parser.add_option("output", "Output file path", 1);
        parser.add_option("no-train", "Skip training (use existing model)");
        parser.add_option("no-embed-model", "Don't embed model in compressed file");

        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("compress") && !parser.option("decompress"))
        {
            cout << "Dlib Predictive Compressor Example\n\n";
            parser.print_options();
            cout << "\nExamples:\n";
            cout << "  Compress with training:\n";
            cout << "    " << argv[0] << " --compress --input data.txt --output data.dpc\n\n";
            cout << "  Compress without training:\n";
            cout << "    " << argv[0] << " --compress --input data.txt --output data.dpc --no-train\n\n";
            cout << "  Decompress:\n";
            cout << "    " << argv[0] << " --decompress --input data.dpc --output data_restored.txt\n";
            return 0;
        }

        if (parser.option("compress") && parser.option("decompress"))
            throw std::runtime_error("Cannot specify both --compress and --decompress");

        if (!parser.option("compress") && !parser.option("decompress"))
            throw std::runtime_error("Must specify either --compress or --decompress");

        if (!parser.option("input"))
            throw std::runtime_error("Missing --input parameter");

        if (!parser.option("output"))
            throw std::runtime_error("Missing --output parameter");

        std::string input_path = parser.option("input").argument();
        std::string output_path = parser.option("output").argument();

        if (parser.option("compress"))
        {
            bool train_model = !parser.option("no-train");
            bool embed_model = !parser.option("no-embed-model");
            compress_file(input_path, output_path, train_model, embed_model);
        }
        else
        {
            decompress_file(input_path, output_path);
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
}