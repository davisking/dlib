// Copyright (C) 2025  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ARC_AGI_ABSTRACT_H_
#ifdef DLIB_ARC_AGI_ABSTRACT_H_

#include <string>
#include <vector>
#include <map>
#include "../matrix.h"
#include "../serialize.h"

namespace dlib
{
    // Type aliases for ARC-AGI data structures
    using arc_grid_t = matrix<unsigned char>;
    using arc_token_sequence_t = matrix<long, 0, 1>;

    // Maximum sequence length for LLM-style training
    constexpr long ARC_MAX_SEQUENCE_LENGTH = 4096;

    // Token vocabulary for the Hierarchical Reasoning Model
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

    // Vocabulary size constants
    constexpr long ARC_VOCAB_SIZE_COLORS = 10;
    constexpr long ARC_VOCAB_SIZE_TOTAL = 17;

    struct arc_task_pair
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Represents a single Input/Output example pair within an ARC task.
                Each pair demonstrates a transformation pattern that the model must learn.
        !*/

        arc_grid_t input;
        /*!
            The input grid (2D matrix of color values 0-9)
        !*/

        arc_grid_t output;
        /*!
            The corresponding output grid showing the transformed result
        !*/

        long input_rows;
        long input_cols;
        long output_rows;
        long output_cols;
        /*!
            Dimensions of the input and output grids
        !*/
    };

    struct arc_task
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Represents a complete ARC-AGI reasoning task containing:
                - Multiple training pairs demonstrating a pattern
                - One or more test pairs where the model must predict outputs
        !*/

        std::string task_id;
        /*!
            Unique identifier extracted from the JSON filename
        !*/

        std::vector<arc_task_pair> train_pairs;
        /*!
            Training examples demonstrating the pattern to learn
        !*/

        std::vector<arc_task_pair> test_pairs;
        /*!
            Test cases where the model must predict the output
        !*/
    };

    class arc_agi_manager
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object provides utilities for loading, accessing, and preparing
                ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)
                dataset for training Transformer-based models such as the Hierarchical
                Reasoning Model (HRM).

                The ARC-AGI dataset consists of visual reasoning tasks where each task
                contains:
                - Training pairs: Input/Output grid examples demonstrating a pattern
                - Test pairs: Input grids where the model must predict the output

                Each grid is a 2D matrix of integers (0-9) representing colors/symbols,
                with maximum dimensions of 30x30.

                TOKENIZATION STRATEGY
                    Grids are tokenized row-by-row with TOKEN_ROW_END markers inserted
                    at the end of each row. This encoding preserves dimensional information
                    implicitly, allowing the model to learn and generate grids of arbitrary
                    dimensions (1x1 to 30x30, including non-square grids) without requiring
                    explicit dimension specification.

                The dataset is available from: https://github.com/fchollet/ARC-AGI
        !*/

    public:
        arc_agi_manager();
        /*!
            ensures
                - Constructs an empty arc_agi_manager object
        !*/

        void load_data(
            const std::string& training_path,
            const std::string& evaluation_path
        );
        /*!
            ensures
                - Attempts to load the ARC-AGI dataset from the specified directories
                - training_path should contain JSON files for training tasks
                - evaluation_path should contain JSON files for evaluation tasks
                - Each JSON file represents one task with training and test pairs
                - Task IDs are extracted from filenames (without .json extension)
            throws
                - std::runtime_error if directories cannot be accessed or files
                  cannot be parsed
        !*/

        const arc_task& get_training_task(size_t index) const;
        /*!
            requires
                - index < num_training_tasks()
            ensures
                - Returns the training task at the specified index
            throws
                - std::out_of_range if index is out of bounds
        !*/

        const arc_task& get_evaluation_task(size_t index) const;
        /*!
            requires
                - index < num_evaluation_tasks()
            ensures
                - Returns the evaluation task at the specified index
            throws
                - std::out_of_range if index is out of bounds
        !*/

        const arc_task& get_training_task_by_id(const std::string& task_id) const;
        /*!
            requires
                - task_id is a valid task identifier
            ensures
                - Returns the training task with the specified task_id
            throws
                - std::runtime_error if task_id is not found
        !*/

        const arc_task& get_evaluation_task_by_id(const std::string& task_id) const;
        /*!
            requires
                - task_id is a valid task identifier
            ensures
                - Returns the evaluation task with the specified task_id
            throws
                - std::runtime_error if task_id is not found
        !*/

        size_t num_training_tasks() const;
        /*!
            ensures
                - Returns the number of loaded training tasks
        !*/

        size_t num_evaluation_tasks() const;
        /*!
            ensures
                - Returns the number of loaded evaluation tasks
        !*/

        void serialize(std::ostream& out) const;
        /*!
            ensures
                - Writes the entire dataset to the output stream in Dlib's
                  serialization format
                - Can be saved to a .dat file for faster loading
        !*/

        void deserialize(std::istream& in);
        /*!
            ensures
                - Loads the entire dataset from the input stream
                - Stream must contain data previously written by serialize()
            throws
                - serialization_error if data format is invalid
        !*/

        static arc_token_sequence_t tokenize_input_context(
            const arc_task& task,
            const arc_task_pair& test_pair
        );
        /*!
            ensures
                - Converts the task's training pairs and the specified test input
                  into a token sequence suitable for LLM-style training
                - Returns a sequence: [grid_tokens..., ROW_END, SEP_IO,
                  grid_tokens..., ROW_END, SEP_PAIR, ..., QUERY_START,
                  test_input_tokens..., ROW_END, GEN_START]
                - Each grid is encoded with TOKEN_ROW_END markers at the end of
                  each row to preserve dimensional information
                - This represents the context that the model uses to predict the output
        !*/

        static arc_token_sequence_t tokenize_target_output(
            const arc_task_pair& test_pair
        );
        /*!
            ensures
                - Converts the test output grid into a token sequence
                - Returns a sequence: [grid_tokens..., ROW_END, ..., END_OF_OUTPUT]
                - Each row is terminated with TOKEN_ROW_END to preserve dimensions
                - This represents the ground truth that the model should predict
        !*/

        static void prepare_training_data_batch(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch
        );
        /*!
            requires
                - window_len > 1
            ensures
                - Prepares training data in the format required by dlib::dnn::trainer
                  using a sliding window approach for causal language modeling
                - For each test pair in the task, generates training samples where:
                  * Each X sample is a context window of size window_len containing
                    the previous window_len tokens
                  * Each Y label is the next token that should follow the context
                - #training_X_batch.size() == #training_Y_batch.size()
                - Each training_X_batch[i] is a column vector (matrix<long, 0, 1>)
                  of size window_len x 1
                - Each training_Y_batch[i] is a single token (long) representing
                  the target to predict
                - Implements left-padding with TOKEN_PADDING when the context window
                  extends before the sequence start, preserving recent context on
                  the right side (standard for causal language models)
                - The concatenated sequence is: [input_context, target_output]
            throws
                - std::invalid_argument if window_len <= 1

            EXAMPLE
                For a sequence [A, B, C, D, E] with window_len=3:
                X[0] = [PAD, PAD, A]  => Y[0] = B
                X[1] = [PAD, A, B]    => Y[1] = C
                X[2] = [A, B, C]      => Y[2] = D
                X[3] = [B, C, D]      => Y[3] = E
                X[4] = [C, D, E]      => Y[4] = PAD
        !*/

        static arc_grid_t detokenize_to_grid(
            const arc_token_sequence_t& tokens,
            long start_idx = 0
        );
        /*!
            requires
                - tokens contains a valid tokenized grid sequence with TOKEN_ROW_END markers
            ensures
                - Reconstructs a grid from a tokenized sequence
                - Uses TOKEN_ROW_END markers to determine row boundaries and infer
                  grid dimensions
                - Parsing stops at TOKEN_END_OF_OUTPUT, TOKEN_SEP_IO, or TOKEN_SEP_PAIR
                - Returns a matrix containing the reconstructed grid
                - Returns an empty matrix (0x0) if no valid grid is found
                - Grid dimensions are automatically determined from the token stream:
                  * Number of rows = count of TOKEN_ROW_END markers
                  * Number of columns = tokens between consecutive TOKEN_ROW_END markers
            throws
                - DLIB_CASSERT if row lengths are inconsistent (indicating malformed data)

            EXAMPLE
                Input tokens: [1, 2, 3, ROW_END, 4, 5, 6, ROW_END, END_OF_OUTPUT]
                Returns: 2x3 grid = [[1, 2, 3], [4, 5, 6]]
        !*/
    };

} // namespace dlib

#endif // DLIB_ARC_AGI_ABSTRACT_H_