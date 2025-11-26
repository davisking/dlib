// Copyright (C) 2025  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_LANGUAGE_MODEL_DATA_ABSTRACT_H_
#ifdef DLIB_LANGUAGE_MODEL_DATA_ABSTRACT_H_

#include <iostream>
#include <string>
#include <vector>
#include "../matrix.h"
#include "../serialize.h"

namespace dlib
{
    // ---------------------------------------------------------------------------------

    enum class file_content_type
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                Enumeration of recognized file content types for classification purposes.
                Used by detect_file_type() to identify the nature of file contents.

            VALUES
                TEXT_PLAIN   - Plain text files (including CSV, source code, logs, etc.)
                TEXT_XML     - XML or HTML markup documents
                IMAGE        - Image formats (PNG, JPEG, GIF, TIFF, BMP, WEBP, etc.)
                VIDEO        - Video formats (MP4, AVI, MKV, etc.)
                AUDIO        - Audio formats (MP3, WAV, FLAC, OGG, etc.)
                EXECUTABLE   - Executable binary files (EXE, DLL, ELF, Mach-O)
                COMPRESSED   - Compressed archives (ZIP, GZIP, 7Z, RAR, etc.)
                PDF          - PDF documents
                OFFICE       - Office documents (DOCX, XLSX, PPTX)
                UNKNOWN      - File type could not be determined or is not recognized

            NOTES
                - Detection is based on file content analysis, not file extensions
                - Magic number signatures are checked first for binary formats
                - Entropy analysis and heuristics are used for text vs binary classification
        !*/
    };

    // ---------------------------------------------------------------------------------

    inline bool detect_file_type(
        const std::string& filename,
        file_content_type& detected_type
    );
    /*!
        ensures
            - Efficiently detects the content type of a file by analyzing its internal
              structure using magic number signatures and entropy-based heuristics
            - Opens and reads the first 8KB of the file for analysis
            - Returns true if file contains text-based content (TEXT_PLAIN or TEXT_XML)
            - Returns false if file contains binary content or cannot be opened
            - Sets detected_type to the most specific content type that could be identified
            - If file cannot be opened, returns false and sets detected_type to UNKNOWN

        FILE DETECTION METHODOLOGY
            The function uses a multi-stage detection process:

            Stage 1: magic number detection (Binary Formats)
                - Checks for ~30 common file format signatures (magic numbers)
                - Supported formats include:
                  * Images: PNG, JPEG (4 variants), GIF (87a/89a), TIFF (LE/BE), BMP, WEBP
                  * Documents: PDF
                  * Compressed: ZIP, GZIP, 7Z, RAR
                  * Executables: Windows PE (EXE/DLL), Unix ELF, macOS Mach-O (32/64-bit)
                  * Audio: MP3 (ID3/FF), WAV, FLAC, OGG
                  * Video: MP4, AVI, MKV
                - Special handling for container formats:
                  * RIFF containers (WAV/AVI/WEBP) are distinguished by format identifier
                  * ZIP files are checked against filename to detect Office documents (DOCX/XLSX/PPTX)
                - If magic number is found, returns false (binary) with appropriate type

            Stage 2: XML/HTML detection
                - Checks for XML declarations (<?xml) and HTML markers
                - Case-insensitive matching for robustness
                - Returns true with TEXT_XML if detected

            Stage 3: entropy analysis
                - Calculates Shannon entropy: H = -sum(p * log2(p))
                - Entropy ranges from 0 (completely uniform) to 8 (maximum randomness)
                - Used to distinguish text from compressed/encrypted content

            Stage 4: text content heuristics
                - Analyzes character distribution:
                  * Counts printable ASCII/UTF-8 characters
                  * Counts whitespace and control characters
                  * Supports multi-byte UTF-8 sequences
                - Text classification criteria:
                  * >90% printable characters
                  * <10% control characters
                  * Entropy < 5.5 (high confidence text)
                  * Entropy < 6.5 (text with special characters)
                  * Entropy >= 6.8 (likely binary/compressed/encrypted)

        TYPICAL USAGE
            file_content_type type;

            // Detect file type
            bool is_text = detect_file_type("document.pdf", type);

            if (type == file_content_type::PDF)
                std::cout << "PDF document detected\n";
            else if (type == file_content_type::IMAGE)
                std::cout << "Image file detected\n";
            else if (is_text)
                std::cout << "Text file detected\n";
            else
                std::cout << "Binary file or unknown format\n";

            // Filter text files for processing
            std::vector<std::string> filenames = get_file_list();
            for (const auto& fname : filenames)
            {
                file_content_type ftype;
                if (detect_file_type(fname, ftype))
                {
                    // Process text file
                    process_text_file(fname);
                }
            }
    !*/

    class inference_context
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This class manages a token context for inference with language models.
                It maintains a full history context and provides a sliding window view
                for model input.

                Features:
                - Full context history with configurable capacity
                - Sliding window extraction for model input
                - Left padding when context not full
                - FIFO policy when context reaches capacity
                - Dynamic resizing without data loss

            TYPICAL USAGE
                inference_context ctx(256, 10, 0);  // window=256, capacity=2560, pad=0

                ctx.add_tokens({1, 2, 3, 4, 5});    // Add tokens
                auto input = ctx.get_input_window(); // Get last 256 tokens (padded if needed)

                // Feed to model, get prediction, add to context
                unsigned long next_token = model(input);
                ctx.add_token(next_token);
        !*/
    public:        
        inference_context(
            long window_size = 256,
            long context_multiplier = 10,
            long padding_token = 0
        );
        /*!
            requires
                - window_size > 0
                - context_multiplier > 0
            ensures
                - Constructs an inference context manager
                - context_capacity = window_size * context_multiplier
                - Context is initially empty (will be left-padded)
        !*/
        
        void add_token(unsigned long token);
        /*!
            ensures
                - Adds a single token to the context
                - If context is full, removes oldest token (FIFO)
                - New token is always added at the end
        !*/        

        void add_tokens(const std::vector<unsigned long>& tokens);
        void add_tokens(const std::vector<int>& tokens);
        /*!
            ensures
                - Adds multiple tokens to the context
                - Tokens are added in order
                - FIFO policy applies if capacity exceeded
        !*/

        matrix<int, 0, 1> get_input_window(long custom_window_size = -1) const;
        /*!
            ensures
                - Returns a window of tokens suitable for model input
                - Window size is custom_window_size if specified, otherwise window_size_
                - Window contains the last N tokens from context
                - Left-padded with padding_token if context has fewer than N tokens
                - Returns matrix<int,0,1> of shape (N, 1) compatible with Dlib
        !*/

        void reset();
        /*!
            ensures
                - Clears all tokens from context
                - Resets current_size to 0
                - Context capacity remains unchanged
        !*/

        void resize_context(long new_capacity);
        /*!
            requires
                - new_capacity > 0
            ensures
                - Resizes the context capacity
                - Preserves existing tokens (up to new capacity)
                - If new_capacity < current_size, keeps only the last new_capacity tokens
        !*/

        long size() const;
        /*!
            ensures
                - Returns the current number of tokens in context
        !*/

        long capacity() const;
        /*!
            ensures
                - Returns the maximum capacity of the context
        !*/

        long window_size() const;
        /*!
            ensures
                - Returns the default window size for model input
        !*/

        bool is_full() const;
        /*!
            ensures
                - Returns true if context is at full capacity
        !*/

        const std::vector<int>& get_full_context() const;
        /*!
            ensures
                - Returns a const reference to the full context vector
        !*/

        std::string to_string(bool show_all = false) const;
        /*!
            ensures
                - Returns a string representation of the context for debugging
        !*/

        friend void serialize(const inference_context& item, std::ostream& out);
        /*!
            ensures
                - Serializes the inference_context to an output stream
                - Saves all context data and configuration parameters
        !*/

        friend void deserialize(inference_context& item, std::istream& in);
        /*!
            ensures
                - Deserializes the inference_context from an input stream
                - Restores all context data and configuration parameters
        !*/

    private:
        std::vector<int> context_;      // Full context history
        long context_capacity_;          // Maximum context size
        long window_size_;               // Window size for model input
        long padding_token_;             // Token used for left padding
        long current_size_;              // Current number of tokens
    };

    inline void build_single_token_prediction_dataset(
        const std::vector<std::vector<int>>& token_sequences,
        long window_len,
        long padding_token,
        bool use_left_padding,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<unsigned long>& Y);
    /*!
        ensures
            - Constructs training samples for single next-token prediction using a sliding window approach
            - For each sequence, creates input windows of size window_len paired with the immediately following token
            - If use_left_padding is true:
                * Sequences shorter than window_len are left-padded with padding_token
                * Sequences >= window_len generate initial samples with progressive left padding
            - If use_left_padding is false:
                * Sequences shorter than window_len are skipped
            - Returns samples in X (input windows) and Y (target tokens)
            - X contains matrix<int,0,1> of shape (window_len, 1)
            - Y contains unsigned long values representing the next token
    !*/

    inline void build_multi_token_prediction_dataset(
        const std::vector<std::vector<int>>& source_sequences,
        const std::vector<std::vector<int>>& target_sequences,
        long src_window_len,
        long tgt_window_len,
        long padding_token,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<matrix<unsigned long, 0, 1>>& Y);
    /*!
        requires
            - source_sequences.size() == target_sequences.size()
            - src_window_len > 0
            - tgt_window_len > 0
        ensures
            - Constructs training samples for sequence-to-sequence prediction
            - For each (source, target) pair, creates aligned windows that slide synchronously
            - Source windows are left-padded with padding_token when source length < src_window_len
            - Target windows are right-padded with padding_token when insufficient tokens remain
            - Sliding continues while both windows contain at least one real (non-padding) token
            - Stops when both sequences are fully consumed (all tokens have appeared in windows)
            - Returns samples in X (source windows) and Y (target windows)
            - X contains matrix<int,0,1> of shape (src_window_len, 1)
            - Y contains matrix<unsigned long,0,1> of shape (tgt_window_len, 1)
    !*/

    template <typename sample_type, typename label_type>
    void shuffle_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        unsigned long seed = 0
    );
    /*!
        requires
            - samples.size() == labels.size()
        ensures
            - Randomly shuffles the training dataset in-place
            - Applies the same permutation to both samples and labels to maintain correspondence
            - If seed == 0, uses a random seed based on current time
            - If seed != 0, uses the provided seed for reproducible shuffling
            - After shuffling, samples[i] still corresponds to labels[i]
            - Uses Fisher-Yates shuffle algorithm for uniform random permutation
    !*/

    template <typename sample_type, typename label_type>
    void augment_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        int unk_token,
        int padding_token,
        double augmentation_ratio = 0.2,
        long min_noise_tokens = 1,
        long max_noise_tokens = 3,
        unsigned long seed = 0
    );
    /*!
        requires
            - samples.size() == labels.size()
            - 0.0 <= augmentation_ratio <= 2.0
            - min_noise_tokens >= 0
            - max_noise_tokens >= min_noise_tokens
        ensures
            - Augments the training dataset by adding noisy copies of existing samples
            - Creates floor(samples.size() * augmentation_ratio) new augmented samples
            - For each augmented sample:
                * Randomly selects a source sample from the original dataset
                * Creates a copy of the sample and its corresponding label
                * Randomly replaces between min_noise_tokens and max_noise_tokens
                  non-padding tokens with unk_token
                * Only tokens != padding_token are eligible for noise injection
                * Number of noise tokens is capped at 30% of non-padding tokens
                  to maintain sample quality
            - Corresponding labels are appended to labels vector (unchanged)
            - Original samples and labels are preserved
            - If seed == 0, uses random seed based on current time
            - If seed != 0, uses provided seed for reproducible augmentation
            - Default augmentation_ratio of 0.2 (20%) follows common practices
              in language model training literature
    !*/

} // namespace dlib

#endif // DLIB_LANGUAGE_MODEL_DATA_ABSTRACT_H_