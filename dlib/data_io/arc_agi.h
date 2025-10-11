// Copyright (C) 2025  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ARC_AGI_H_
#define DLIB_ARC_AGI_H_

#include "arc_agi_abstract.h"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include "../matrix.h"
#include "../dir_nav.h"
#include "../serialize.h"

namespace dlib
{

    // ----------------------------------------------------------------------------------------
    // Type aliases and constants
    // ----------------------------------------------------------------------------------------

    /*!
        Type aliases for ARC-AGI data structures. Grids are represented as matrices
        of unsigned char values (0-9), and token sequences are column vectors of long.
    !*/
    using arc_grid_t = matrix<unsigned char>;
    using arc_token_sequence_t = matrix<long, 0, 1>;

    /*!
        Maximum sequence length for LLM-style training. This constant defines the
        upper bound for token sequences that can be processed by the model.
    !*/
    constexpr long ARC_MAX_SEQUENCE_LENGTH = 4096;

    // ----------------------------------------------------------------------------------------
    // Token vocabulary
    // ----------------------------------------------------------------------------------------

    /*!
        Token vocabulary for the Hierarchical Reasoning Model. The vocabulary includes:
        - COLOR_0 to COLOR_9: Grid cell colors (10 values)
        - TOKEN_SEP_IO: Separator between input and output grids
        - TOKEN_SEP_PAIR: Separator between demonstration pairs
        - TOKEN_QUERY_START: Marks the beginning of a test query
        - TOKEN_GEN_START: Marks the beginning of generation phase
        - TOKEN_END_OF_OUTPUT: Marks the end of generated output
        - TOKEN_PADDING: Padding token for variable-length sequences
        - TOKEN_ROW_END: Marks the end of a grid row (for dimension encoding)
    !*/
    enum arc_token_id : long
    {
        COLOR_0 = 0, COLOR_1 = 1, COLOR_2 = 2, COLOR_3 = 3, COLOR_4 = 4,
        COLOR_5 = 5, COLOR_6 = 6, COLOR_7 = 7, COLOR_8 = 8, COLOR_9 = 9,
        TOKEN_SEP_IO = 10,
        TOKEN_SEP_PAIR = 11,
        TOKEN_QUERY_START = 12,
        TOKEN_GEN_START = 13,
        TOKEN_END_OF_OUTPUT = 14,
        TOKEN_PADDING = 15,
        TOKEN_ROW_END = 16
    };

    /*!
        Vocabulary size constants for the token set.
    !*/
    constexpr long ARC_VOCAB_SIZE_COLORS = 10;
    constexpr long ARC_VOCAB_SIZE_TOTAL = 17;

    // ----------------------------------------------------------------------------------------
    // ARC-AGI task data structures
    // ----------------------------------------------------------------------------------------

    /*!
        Represents a single input-output pair in an ARC-AGI task. Each pair consists
        of an input grid and its corresponding output grid, along with their dimensions.
    !*/
    struct arc_task_pair
    {
        arc_grid_t input;
        arc_grid_t output;
        long input_rows;
        long input_cols;
        long output_rows;
        long output_cols;

        friend void serialize(const arc_task_pair& item, std::ostream& out)
        {
            dlib::serialize(item.input, out);
            dlib::serialize(item.output, out);
            dlib::serialize(item.input_rows, out);
            dlib::serialize(item.input_cols, out);
            dlib::serialize(item.output_rows, out);
            dlib::serialize(item.output_cols, out);
        }

        friend void deserialize(arc_task_pair& item, std::istream& in)
        {
            dlib::deserialize(item.input, in);
            dlib::deserialize(item.output, in);
            dlib::deserialize(item.input_rows, in);
            dlib::deserialize(item.input_cols, in);
            dlib::deserialize(item.output_rows, in);
            dlib::deserialize(item.output_cols, in);
        }
    };

    /*!
        Represents a complete ARC-AGI task. Each task contains:
        - A unique task identifier
        - A set of training demonstration pairs
        - A set of test pairs (where outputs are to be predicted)
    !*/
    struct arc_task
    {
        std::string task_id;
        std::vector<arc_task_pair> train_pairs;
        std::vector<arc_task_pair> test_pairs;

        friend void serialize(const arc_task& item, std::ostream& out)
        {
            dlib::serialize(item.task_id, out);
            dlib::serialize(item.train_pairs, out);
            dlib::serialize(item.test_pairs, out);
        }

        friend void deserialize(arc_task& item, std::istream& in)
        {
            dlib::deserialize(item.task_id, in);
            dlib::deserialize(item.train_pairs, in);
            dlib::deserialize(item.test_pairs, in);
        }
    };

    // ----------------------------------------------------------------------------------------
    // Internal JSON parsing utilities
    // ----------------------------------------------------------------------------------------

    namespace internal
    {
        using raw_arc_grid_t = std::vector<std::vector<int>>;

        // ------------------------------------------------------------------------------------

        inline std::string read_file_to_string(const std::string& path)
        /*!
            ensures
                - Reads the entire contents of a file and returns it as a string
                - Throws std::runtime_error if the file cannot be opened
        !*/
        {
            std::ifstream file(path);
            if (!file.is_open())
                throw std::runtime_error("Failed to open file: " + path);
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        // ------------------------------------------------------------------------------------

        inline std::vector<int> parse_int_array(const std::string& str)
        /*!
            ensures
                - Parses a comma-separated string of integers
                - Returns a vector containing the parsed integers
                - Whitespace around numbers is automatically stripped
        !*/
        {
            std::vector<int> result;
            std::stringstream ss(str);
            std::string segment;
            while (std::getline(ss, segment, ','))
            {
                segment.erase(0, segment.find_first_not_of(" \t\n\r"));
                segment.erase(segment.find_last_not_of(" \t\n\r") + 1);
                if (!segment.empty())
                    result.push_back(std::stoi(segment));
            }
            return result;
        }

        // ------------------------------------------------------------------------------------

        inline raw_arc_grid_t parse_arc_grid(std::string::const_iterator& it,
            const std::string::const_iterator& end)
        /*!
            ensures
                - Parses a 2D grid from JSON array-of-arrays format
                - Advances the iterator 'it' past the parsed content
                - Returns a vector of vectors representing the grid rows
                - Throws std::runtime_error on malformed input
        !*/
        {
            raw_arc_grid_t grid;

            // Locate the opening bracket of the outer array
            it = std::find(it, end, '[');
            if (it == end) return grid;
            ++it;

            // Skip any leading whitespace
            while (it != end && std::isspace(*it)) ++it;

            // Verify we have an array of arrays (second '[')
            if (it == end || *it != '[') return grid;

            // Parse each row in the grid
            while (it != end)
            {
                // Skip whitespace between rows
                while (it != end && std::isspace(*it)) ++it;

                // Check for end of outer array
                if (it == end || *it == ']') break;

                // Expect a '[' at the start of each row
                if (*it != '[') {
                    ++it;
                    continue;
                }
                ++it;

                // Find the closing ']' for this row
                auto inner_end = std::find(it, end, ']');
                if (inner_end == end)
                    throw std::runtime_error("Missing inner array closing bracket");

                // Parse the integers in this row
                std::string row_str(it, inner_end);
                auto row = parse_int_array(row_str);

                if (!row.empty())
                    grid.push_back(row);

                it = inner_end;
                ++it;

                // Skip trailing whitespace, commas, and newlines
                while (it != end && (*it == ' ' || *it == ',' || *it == '\n' ||
                    *it == '\r' || *it == '\t'))
                    ++it;
            }

            // Advance past the closing ']' of the outer array
            if (it != end && *it == ']') ++it;

            return grid;
        }

        // ------------------------------------------------------------------------------------

        inline std::string::const_iterator find_key_value_start(
            const std::string& content,
            const std::string& key,
            std::string::const_iterator start_it)
        /*!
            ensures
                - Searches for a JSON key-value pair starting from start_it
                - Returns an iterator pointing to the first character of the value
                - Returns content.end() if the key is not found
        !*/
        {
            std::string search_str = "\"" + key + "\":";
            auto pos = std::search(start_it, content.end(),
                search_str.begin(), search_str.end());
            if (pos == content.end()) return content.end();
            pos += search_str.length();
            while (pos != content.end() && std::isspace(*pos)) ++pos;
            return pos;
        }

        // ------------------------------------------------------------------------------------

        inline std::string extract_task_id_from_filename(const std::string& filename)
        /*!
            ensures
                - Extracts the task ID from a filename by removing the file extension
                - If no extension is found, returns the filename unchanged
        !*/
        {
            size_t dot_pos = filename.find_last_of('.');
            if (dot_pos == std::string::npos)
                return filename;
            return filename.substr(0, dot_pos);
        }

    } // namespace internal

    // ----------------------------------------------------------------------------------------
    // arc_agi_manager class
    // ----------------------------------------------------------------------------------------

    /*!
        The arc_agi_manager class provides functionality to:
        - Load ARC-AGI tasks from JSON files
        - Manage training and evaluation datasets
        - Convert grids to token sequences for LLM training
        - Generate training batches with sliding window context
        - Serialize and deserialize task data

        THREAD SAFETY
            This class is not thread-safe. External synchronization is required
            if accessing the same instance from multiple threads.

        TOKENIZATION STRATEGY
            Grids are tokenized row-by-row with TOKEN_ROW_END markers to preserve
            dimensional information. This allows the model to learn the structure
            of non-square grids (ranging from 1x1 to 30x30) without explicit
            dimension encoding.
    !*/
    class arc_agi_manager
    {
    private:
        std::vector<arc_task> training_tasks;
        std::vector<arc_task> evaluation_tasks;
        std::map<std::string, size_t> training_task_id_map;
        std::map<std::string, size_t> evaluation_task_id_map;

        // ------------------------------------------------------------------------------------

        static void append_flat_grid(std::vector<long>& sequence, const arc_grid_t& grid)
        /*!
            requires
                - grid contains valid color values (0-9)
            ensures
                - Appends the grid to the sequence in row-major order
                - Each row is terminated with TOKEN_ROW_END
                - This encoding preserves grid dimensions for reconstruction
        !*/
        {
            for (long r = 0; r < grid.nr(); ++r)
            {
                for (long c = 0; c < grid.nc(); ++c)
                    sequence.push_back(static_cast<long>(grid(r, c)));

                // Mark the end of this row to encode dimensional information
                sequence.push_back(TOKEN_ROW_END);
            }
        }

        // ------------------------------------------------------------------------------------

        static arc_grid_t to_dlib_matrix(const internal::raw_arc_grid_t& grid)
        /*!
            requires
                - grid is a valid 2D array with consistent row lengths
                - all values are in the range [0, 9]
            ensures
                - Converts a raw vector-of-vectors grid to a dlib matrix
                - Returns an empty matrix if the input grid is empty
            throws
                - DLIB_CASSERT if row lengths are inconsistent
                - DLIB_CASSERT if pixel values are outside [0, 9]
        !*/
        {
            if (grid.empty()) return arc_grid_t(0, 0);
            long rows = static_cast<long>(grid.size());
            long cols = static_cast<long>(grid[0].size());
            arc_grid_t mat(rows, cols);

            for (long r = 0; r < rows; ++r)
            {
                DLIB_CASSERT(static_cast<long>(grid[r].size()) == cols,
                    "Inconsistent column size in grid");
                for (long c = 0; c < cols; ++c)
                {
                    DLIB_CASSERT(grid[r][c] >= 0 && grid[r][c] <= 9,
                        "Invalid pixel value (must be 0-9)");
                    mat(r, c) = static_cast<unsigned char>(grid[r][c]);
                }
            }
            return mat;
        }

        // ------------------------------------------------------------------------------------

        arc_task parse_arc_task_from_content(const std::string& content,
            const std::string& filename)
        /*!
            ensures
                - Parses a complete ARC task from JSON content
                - Returns an arc_task structure with all training and test pairs
                - Task ID is extracted from the filename
            throws
                - std::runtime_error on malformed JSON or missing required fields
        !*/
        {
            arc_task task;
            task.task_id = internal::extract_task_id_from_filename(filename);

            auto parse_pairs = [&](const std::string& key,
                std::vector<arc_task_pair>& pairs)
                {
                    auto it = internal::find_key_value_start(content, key, content.begin());
                    if (it == content.end() || *it != '[')
                        throw std::runtime_error("'" + key + "' array not found");
                    ++it;

                    // Iterate through each object in the array
                    while (it != content.end())
                    {
                        // Skip inter-object whitespace
                        while (it != content.end() && std::isspace(*it)) ++it;

                        // Check if we've reached the end of the array
                        if (it == content.end() || *it == ']') break;

                        // Locate the opening brace of this object
                        if (*it != '{') {
                            ++it;
                            continue;
                        }

                        // Mark boundaries for scoped key searches
                        auto object_start = it;
                        ++it;

                        // Find the matching closing brace
                        int brace_depth = 1;
                        auto object_end = it;
                        while (object_end != content.end() && brace_depth > 0)
                        {
                            if (*object_end == '{') ++brace_depth;
                            else if (*object_end == '}') --brace_depth;
                            ++object_end;
                        }

                        if (object_end == content.end())
                            throw std::runtime_error("Missing object closing bracket");

                        arc_task_pair pair;

                        // Parse the "input" field within this object's scope
                        auto input_it = internal::find_key_value_start(content, "input", object_start);
                        if (input_it == content.end() || input_it >= object_end)
                            throw std::runtime_error("'input' not found in " + key + " object");

                        auto raw_input = internal::parse_arc_grid(input_it, object_end);
                        pair.input = to_dlib_matrix(raw_input);
                        pair.input_rows = pair.input.nr();
                        pair.input_cols = pair.input.nc();

                        // Parse the "output" field (search starts after input)
                        auto output_it = internal::find_key_value_start(content, "output", input_it);
                        if (output_it == content.end() || output_it >= object_end)
                            throw std::runtime_error("'output' not found in " + key + " object");

                        auto raw_output = internal::parse_arc_grid(output_it, object_end);
                        pair.output = to_dlib_matrix(raw_output);
                        pair.output_rows = pair.output.nr();
                        pair.output_cols = pair.output.nc();

                        pairs.push_back(pair);

                        // Advance iterator past this object
                        it = object_end;
                    }
                };

            parse_pairs("train", task.train_pairs);
            parse_pairs("test", task.test_pairs);
            return task;
        }

        // ------------------------------------------------------------------------------------

        std::vector<arc_task> load_all_tasks(const std::string& directory_path,
            std::map<std::string, size_t>& id_map)
        /*!
            ensures
                - Loads all .json files from the specified directory
                - Each file is parsed as an ARC task
                - Returns a vector of successfully loaded tasks
                - Populates id_map with task_id to index mappings
                - Outputs diagnostic information to stdout/stderr
        !*/
        {
            std::vector<arc_task> tasks;
            std::cout << "Loading tasks from: " << directory_path << std::endl;

            try {
                const dlib::directory dir(directory_path);
                std::vector<dlib::file> all_files = dir.get_files();

                std::cout << "Found " << all_files.size() << " files in directory" << std::endl;

                // Filter for JSON files only
                std::vector<dlib::file> json_files;
                for (const auto& file : all_files)
                {
                    const std::string& filename = file.name();
                    if (filename.size() >= 5 &&
                        filename.substr(filename.size() - 5) == ".json")
                    {
                        json_files.push_back(file);
                    }
                }

                std::cout << "Found " << json_files.size() << " .json files" << std::endl;

                if (json_files.empty()) {
                    std::cout << "WARNING: No .json files found in "
                        << directory_path << std::endl;
                    return tasks;
                }

                size_t success_count = 0;
                size_t error_count = 0;

                // Attempt to load each JSON file
                for (const auto& file : json_files)
                {
                    try {
                        std::string content = internal::read_file_to_string(file.full_name());
                        arc_task task = parse_arc_task_from_content(content, file.name());
                        id_map[task.task_id] = tasks.size();
                        tasks.push_back(task);
                        ++success_count;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "ERROR parsing " << file.name()
                            << ": " << e.what() << std::endl;
                        ++error_count;
                    }
                }

                std::cout << "Successfully loaded " << success_count << " tasks" << std::endl;
                if (error_count > 0) {
                    std::cout << "Failed to load " << error_count << " tasks" << std::endl;
                }

            }
            catch (const dlib::directory::dir_not_found& e) {
                std::cerr << "ERROR: Directory not found: " << directory_path << std::endl;
                std::cerr << "Details: " << e.info << std::endl;
            }
            catch (const dlib::directory::listing_error& e) {
                std::cerr << "ERROR: Cannot list directory: " << directory_path << std::endl;
                std::cerr << "Details: " << e.info << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR during directory navigation: " << e.what() << std::endl;
            }
            return tasks;
        }

    public:
        arc_agi_manager() = default;

        // ------------------------------------------------------------------------------------

        void load_data(const std::string& training_path,
            const std::string& evaluation_path)
        /*!
            ensures
                - Loads all ARC tasks from training and evaluation directories
                - Clears any previously loaded data
                - Outputs a summary of loaded tasks to stdout
        !*/
        {
            training_task_id_map.clear();
            evaluation_task_id_map.clear();

            training_tasks = load_all_tasks(training_path, training_task_id_map);
            evaluation_tasks = load_all_tasks(evaluation_path, evaluation_task_id_map);

            std::cout << "--- ARC Data Loading Summary ---" << std::endl;
            std::cout << "Loaded " << training_tasks.size() << " training tasks" << std::endl;
            std::cout << "Loaded " << evaluation_tasks.size() << " evaluation tasks" << std::endl;
            std::cout << "--------------------------------" << std::endl;
        }

        // ------------------------------------------------------------------------------------

        const arc_task& get_training_task(size_t index) const
        /*!
            requires
                - index < num_training_tasks()
            ensures
                - Returns a const reference to the training task at the given index
            throws
                - DLIB_CASSERT if index is out of bounds
        !*/
        {
            DLIB_CASSERT(index < training_tasks.size(),
                "Training task index out of bounds"
                << "\n\tRequested index: " << index
                << "\n\tAvailable tasks: " << training_tasks.size());
            return training_tasks[index];
        }

        // ------------------------------------------------------------------------------------

        const arc_task& get_evaluation_task(size_t index) const
        /*!
            requires
                - index < num_evaluation_tasks()
            ensures
                - Returns a const reference to the evaluation task at the given index
            throws
                - DLIB_CASSERT if index is out of bounds
        !*/
        {
            DLIB_CASSERT(index < evaluation_tasks.size(),
                "Evaluation task index out of bounds");
            return evaluation_tasks[index];
        }

        // ------------------------------------------------------------------------------------

        const arc_task& get_training_task_by_id(const std::string& task_id) const
        /*!
            ensures
                - Returns a const reference to the training task with the given ID
            throws
                - std::runtime_error if task_id is not found
        !*/
        {
            auto it = training_task_id_map.find(task_id);
            if (it == training_task_id_map.end())
                throw std::runtime_error("Training task ID not found: " + task_id);
            return training_tasks[it->second];
        }

        // ------------------------------------------------------------------------------------

        const arc_task& get_evaluation_task_by_id(const std::string& task_id) const
        /*!
            ensures
                - Returns a const reference to the evaluation task with the given ID
            throws
                - std::runtime_error if task_id is not found
        !*/
        {
            auto it = evaluation_task_id_map.find(task_id);
            if (it == evaluation_task_id_map.end())
                throw std::runtime_error("Evaluation task ID not found: " + task_id);
            return evaluation_tasks[it->second];
        }

        // ------------------------------------------------------------------------------------

        size_t num_training_tasks() const { return training_tasks.size(); }
        size_t num_evaluation_tasks() const { return evaluation_tasks.size(); }

        // ------------------------------------------------------------------------------------

        void serialize(std::ostream& out) const
        /*!
            ensures
                - Serializes the entire dataset to the output stream
                - Format is versioned for forward compatibility
        !*/
        {
            dlib::serialize("arc_agi_v1", out);
            dlib::serialize(training_tasks, out);
            dlib::serialize(evaluation_tasks, out);
            dlib::serialize(training_task_id_map, out);
            dlib::serialize(evaluation_task_id_map, out);
        }

        // ------------------------------------------------------------------------------------

        void deserialize(std::istream& in)
        /*!
            ensures
                - Deserializes a dataset from the input stream
                - Replaces any existing data in this object
            throws
                - serialization_error if version mismatch is detected
        !*/
        {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "arc_agi_v1")
                throw serialization_error("Unexpected version in arc_agi_manager");
            dlib::deserialize(training_tasks, in);
            dlib::deserialize(evaluation_tasks, in);
            dlib::deserialize(training_task_id_map, in);
            dlib::deserialize(evaluation_task_id_map, in);
        }

        // ----------------------------------------------------------------------------------------
        // Tokenization for LLM-style training
        // ----------------------------------------------------------------------------------------

        static arc_token_sequence_t tokenize_input_context(const arc_task& task,
            const arc_task_pair& test_pair)
        /*!
            ensures
                - Creates a token sequence representing the input context for a test pair
                - Format: [train_input, SEP_IO, train_output, SEP_PAIR]* QUERY_START test_input GEN_START
                - Each grid is tokenized with TOKEN_ROW_END markers preserving dimensions
                - Returns a column vector of tokens
        !*/
        {
            std::vector<long> sequence;

            // Encode all training demonstration pairs
            for (const auto& pair : task.train_pairs)
            {
                append_flat_grid(sequence, pair.input);
                sequence.push_back(TOKEN_SEP_IO);
                append_flat_grid(sequence, pair.output);
                sequence.push_back(TOKEN_SEP_PAIR);
            }

            // Encode the test query
            sequence.push_back(TOKEN_QUERY_START);
            append_flat_grid(sequence, test_pair.input);
            sequence.push_back(TOKEN_GEN_START);

            // Convert to dlib column vector
            arc_token_sequence_t result(static_cast<long>(sequence.size()));
            for (long i = 0; i < static_cast<long>(sequence.size()); ++i)
                result(i) = sequence[i];
            return result;
        }

        // ------------------------------------------------------------------------------------

        static arc_token_sequence_t tokenize_target_output(const arc_task_pair& test_pair)
        /*!
            ensures
                - Creates a token sequence for the target output grid
                - Format: output_grid END_OF_OUTPUT
                - Output grid includes TOKEN_ROW_END markers
                - Returns a column vector of tokens
        !*/
        {
            std::vector<long> sequence;
            append_flat_grid(sequence, test_pair.output);
            sequence.push_back(TOKEN_END_OF_OUTPUT);

            arc_token_sequence_t result(static_cast<long>(sequence.size()));
            for (long i = 0; i < static_cast<long>(sequence.size()); ++i)
                result(i) = sequence[i];
            return result;
        }

        // ------------------------------------------------------------------------------------

        static void prepare_training_data_batch(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch)
        /*!
            requires
                - window_len > 1
            ensures
                - Generates training samples using a sliding window approach
                - Each X sample contains window_len tokens of context
                - Each Y label is the next token following the context window
                - Padding tokens are used when the window extends beyond sequence boundaries
                - training_X_batch[i] is a column vector of length window_len
                - training_Y_batch[i] is the target token for training_X_batch[i]
                - Processes all test pairs in the task
            throws
                - DLIB_CASSERT if window_len <= 1

            IMPLEMENTATION NOTES
                This function implements causal language modeling for ARC tasks.
                For each position in the concatenated [context + target] sequence,
                it creates a training example where:
                - X = [t_{pos-window_len+1}, ..., t_{pos}]
                - Y = t_{pos+1}

                The sliding window ensures the model learns to predict each token
                given the appropriate amount of left context.
        !*/
        {
            DLIB_CASSERT(window_len > 1, "Window length must be greater than 1");

            training_X_batch.clear();
            training_Y_batch.clear();

            for (const arc_task_pair& test_pair : task.test_pairs)
            {
                // Tokenize the full sequence: context + target
                arc_token_sequence_t input_context = tokenize_input_context(task, test_pair);
                arc_token_sequence_t target_output = tokenize_target_output(test_pair);

                long L_in = input_context.size();
                long L_out = target_output.size();
                long L_full = L_in + L_out;

                // Build the complete token sequence
                std::vector<long> S_vec;
                S_vec.reserve(static_cast<size_t>(L_full));

                for (long i = 0; i < L_in; ++i)
                    S_vec.push_back(input_context(i));
                for (long i = 0; i < L_out; ++i)
                    S_vec.push_back(target_output(i));

                // Generate sliding window samples
                // For each position, create a context window of length window_len
                for (long pos = 0; pos < L_full; ++pos)
                {
                    arc_token_sequence_t X_window(window_len);

                    // Fill the context window
                    // Window spans from (pos - window_len + 1) to pos inclusive
                    for (long i = 0; i < window_len; ++i)
                    {
                        long context_idx = pos - window_len + 1 + i;

                        // Use padding for positions before sequence start or after end
                        if (context_idx < 0 || context_idx >= L_full)
                            X_window(i) = TOKEN_PADDING;
                        else
                            X_window(i) = S_vec[static_cast<size_t>(context_idx)];
                    }

                    // The target is the next token after the window
                    long y_token = (pos + 1 < L_full) ?
                        S_vec[static_cast<size_t>(pos + 1)] : TOKEN_PADDING;

                    training_X_batch.push_back(std::move(X_window));
                    training_Y_batch.push_back(y_token);
                }
            }
        }

        // ----------------------------------------------------------------------------------------
        // Detokenization utilities
        // ----------------------------------------------------------------------------------------

        static arc_grid_t detokenize_to_grid(const arc_token_sequence_t& tokens,
            long start_idx = 0)
        /*!
            ensures
                - Reconstructs a grid from a tokenized sequence
                - Uses TOKEN_ROW_END markers to determine row boundaries
                - Stops at TOKEN_END_OF_OUTPUT, TOKEN_SEP_IO, or TOKEN_SEP_PAIR
                - Returns a matrix with the reconstructed grid
                - Returns an empty matrix if no valid grid is found
            throws
                - DLIB_CASSERT if row lengths are inconsistent

            IMPLEMENTATION NOTES
                This function recovers grid dimensions from the token stream by
                counting tokens between TOKEN_ROW_END markers. This allows the
                model to generate grids of arbitrary dimensions (1x1 to 30x30)
                without explicit dimension specification.
        !*/
        {
            // Extract rows from the token sequence
            std::vector<std::vector<unsigned char>> rows;
            std::vector<unsigned char> current_row;

            for (long i = start_idx; i < tokens.size(); ++i)
            {
                long token = tokens(i);

                if (token == TOKEN_ROW_END)
                {
                    // End of current row - save it if non-empty
                    if (!current_row.empty())
                    {
                        rows.push_back(current_row);
                        current_row.clear();
                    }
                }
                else if (token == TOKEN_END_OF_OUTPUT ||
                    token == TOKEN_SEP_IO ||
                    token == TOKEN_SEP_PAIR)
                {
                    // End of grid section
                    break;
                }
                else if (token >= COLOR_0 && token <= COLOR_9)
                {
                    // Valid color token - add to current row
                    current_row.push_back(static_cast<unsigned char>(token));
                }
                // Ignore other tokens (padding, etc.)
            }

            // Build the output matrix
            if (rows.empty())
                return arc_grid_t(0, 0);

            long n_rows = static_cast<long>(rows.size());
            long n_cols = static_cast<long>(rows[0].size());

            arc_grid_t grid(n_rows, n_cols);
            for (long r = 0; r < n_rows; ++r)
            {
                DLIB_CASSERT(static_cast<long>(rows[r].size()) == n_cols,
                    "Inconsistent row length during detokenization"
                    << "\n\tRow " << r << " has " << rows[r].size() << " columns"
                    << "\n\tExpected " << n_cols << " columns");
                for (long c = 0; c < n_cols; ++c)
                    grid(r, c) = rows[r][c];
            }

            return grid;
        }
    };

} // namespace dlib

#endif // DLIB_ARC_AGI_H_