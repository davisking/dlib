#ifndef DLIB_LANGUAGE_MODEL_DATA_H_
#define DLIB_LANGUAGE_MODEL_DATA_H_

#include "language_model_data_abstract.h"

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
        TEXT_PLAIN,      // Plain text file (including CSV, code, etc.)
        TEXT_XML,        // XML or HTML markup
        IMAGE,           // Image formats (PNG, JPEG, GIF, TIFF, BMP, etc.)
        VIDEO,           // Video formats (MP4, AVI, MKV, etc.)
        AUDIO,           // Audio formats (MP3, WAV, FLAC, etc.)
        EXECUTABLE,      // Executable files (EXE, DLL, ELF, Mach-O)
        COMPRESSED,      // Compressed archives (ZIP, GZIP, 7Z, RAR, etc.)
        PDF,             // PDF documents
        OFFICE,          // Office documents (DOCX, XLSX, PPTX, etc.)
        UNKNOWN          // Unknown or undetermined file type
    };

    // ---------------------------------------------------------------------------------

    namespace impl
    {
        // Magic number signature structure
        struct magic_signature
        {
            const unsigned char* bytes;
            size_t length;
            file_content_type type;
            size_t offset;  // Byte offset where signature should appear
        };

        // Common magic number signatures (ordered by frequency/priority)
        static const unsigned char sig_png[] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
        static const unsigned char sig_jpg1[] = { 0xFF, 0xD8, 0xFF, 0xE0 };
        static const unsigned char sig_jpg2[] = { 0xFF, 0xD8, 0xFF, 0xE1 };
        static const unsigned char sig_jpg3[] = { 0xFF, 0xD8, 0xFF, 0xDB };
        static const unsigned char sig_jpg4[] = { 0xFF, 0xD8, 0xFF, 0xEE };
        static const unsigned char sig_gif87[] = { 0x47, 0x49, 0x46, 0x38, 0x37, 0x61 };  // GIF87a
        static const unsigned char sig_gif89[] = { 0x47, 0x49, 0x46, 0x38, 0x39, 0x61 };  // GIF89a
        static const unsigned char sig_tiff_le[] = { 0x49, 0x49, 0x2A, 0x00 };  // Little endian
        static const unsigned char sig_tiff_be[] = { 0x4D, 0x4D, 0x00, 0x2A };  // Big endian
        static const unsigned char sig_bmp[] = { 0x42, 0x4D };
        static const unsigned char sig_webp[] = { 0x52, 0x49, 0x46, 0x46 };  // RIFF (check for WEBP at offset 8)

        static const unsigned char sig_pdf[] = { 0x25, 0x50, 0x44, 0x46 };  // %PDF

        static const unsigned char sig_zip[] = { 0x50, 0x4B, 0x03, 0x04 };
        static const unsigned char sig_gzip[] = { 0x1F, 0x8B };
        static const unsigned char sig_7z[] = { 0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C };
        static const unsigned char sig_rar[] = { 0x52, 0x61, 0x72, 0x21, 0x1A, 0x07 };

        static const unsigned char sig_exe[] = { 0x4D, 0x5A };  // MZ (DOS/Windows executable)
        static const unsigned char sig_elf[] = { 0x7F, 0x45, 0x4C, 0x46 };  // ELF (Unix/Linux executable)
        static const unsigned char sig_macho_32[] = { 0xFE, 0xED, 0xFA, 0xCE };  // Mach-O 32-bit
        static const unsigned char sig_macho_64[] = { 0xFE, 0xED, 0xFA, 0xCF };  // Mach-O 64-bit

        static const unsigned char sig_mp3_id3[] = { 0x49, 0x44, 0x33 };  // ID3
        static const unsigned char sig_mp3_ff[] = { 0xFF, 0xFB };
        static const unsigned char sig_wav[] = { 0x52, 0x49, 0x46, 0x46 };  // RIFF (check for WAVE at offset 8)
        static const unsigned char sig_flac[] = { 0x66, 0x4C, 0x61, 0x43 };  // fLaC
        static const unsigned char sig_ogg[] = { 0x4F, 0x67, 0x67, 0x53 };  // OggS

        static const unsigned char sig_mp4[] = { 0x66, 0x74, 0x79, 0x70 };  // ftyp (at offset 4)
        static const unsigned char sig_avi[] = { 0x52, 0x49, 0x46, 0x46 };  // RIFF (check for AVI at offset 8)
        static const unsigned char sig_mkv[] = { 0x1A, 0x45, 0xDF, 0xA3 };

        static const magic_signature signatures[] = {
            // Images
            {sig_png, sizeof(sig_png), file_content_type::IMAGE, 0},
            {sig_jpg1, sizeof(sig_jpg1), file_content_type::IMAGE, 0},
            {sig_jpg2, sizeof(sig_jpg2), file_content_type::IMAGE, 0},
            {sig_jpg3, sizeof(sig_jpg3), file_content_type::IMAGE, 0},
            {sig_jpg4, sizeof(sig_jpg4), file_content_type::IMAGE, 0},
            {sig_gif87, sizeof(sig_gif87), file_content_type::IMAGE, 0},
            {sig_gif89, sizeof(sig_gif89), file_content_type::IMAGE, 0},
            {sig_tiff_le, sizeof(sig_tiff_le), file_content_type::IMAGE, 0},
            {sig_tiff_be, sizeof(sig_tiff_be), file_content_type::IMAGE, 0},
            {sig_bmp, sizeof(sig_bmp), file_content_type::IMAGE, 0},

            // PDF
            {sig_pdf, sizeof(sig_pdf), file_content_type::PDF, 0},

            // Compressed
            {sig_zip, sizeof(sig_zip), file_content_type::COMPRESSED, 0},
            {sig_gzip, sizeof(sig_gzip), file_content_type::COMPRESSED, 0},
            {sig_7z, sizeof(sig_7z), file_content_type::COMPRESSED, 0},
            {sig_rar, sizeof(sig_rar), file_content_type::COMPRESSED, 0},

            // Executables
            {sig_exe, sizeof(sig_exe), file_content_type::EXECUTABLE, 0},
            {sig_elf, sizeof(sig_elf), file_content_type::EXECUTABLE, 0},
            {sig_macho_32, sizeof(sig_macho_32), file_content_type::EXECUTABLE, 0},
            {sig_macho_64, sizeof(sig_macho_64), file_content_type::EXECUTABLE, 0},

            // Audio
            {sig_mp3_id3, sizeof(sig_mp3_id3), file_content_type::AUDIO, 0},
            {sig_mp3_ff, sizeof(sig_mp3_ff), file_content_type::AUDIO, 0},
            {sig_flac, sizeof(sig_flac), file_content_type::AUDIO, 0},
            {sig_ogg, sizeof(sig_ogg), file_content_type::AUDIO, 0},

            // Video
            {sig_mp4, sizeof(sig_mp4), file_content_type::VIDEO, 4},
            {sig_mkv, sizeof(sig_mkv), file_content_type::VIDEO, 0}
        };

        // Portable case-insensitive string comparison (C++14 compatible)
        inline bool iequals_n(const char* s1, const char* s2, size_t n)
        {
            for (size_t i = 0; i < n; ++i)
            {
                const char c1 = (s1[i] >= 'A' && s1[i] <= 'Z') ? s1[i] + 32 : s1[i];
                const char c2 = (s2[i] >= 'A' && s2[i] <= 'Z') ? s2[i] + 32 : s2[i];
                if (c1 != c2) return false;
            }
            return true;
        }

        // Case-insensitive check for file extension
        inline bool has_extension(const std::string& filename, const char* ext)
        {
            const size_t ext_len = std::strlen(ext);
            if (filename.length() < ext_len) return false;

            const size_t start = filename.length() - ext_len;
            for (size_t i = 0; i < ext_len; ++i)
            {
                const char fc = filename[start + i];
                const char ec = ext[i];
                const char fc_lower = (fc >= 'A' && fc <= 'Z') ? fc + 32 : fc;
                const char ec_lower = (ec >= 'A' && ec <= 'Z') ? ec + 32 : ec;
                if (fc_lower != ec_lower) return false;
            }
            return true;
        }

        // Calculate Shannon entropy for a buffer
        inline double calculate_entropy(const unsigned char* buffer, size_t length)
        {
            if (length == 0) return 0.0;

            // Count byte frequency
            std::array<size_t, 256> counts = {};
            for (size_t i = 0; i < length; ++i)
                counts[buffer[i]]++;

            // Calculate entropy using Shannon's formula: H = -sum(p * log2(p))
            double entropy = 0.0;
            const double length_d = static_cast<double>(length);

            for (size_t i = 0; i < 256; ++i)
            {
                if (counts[i] > 0)
                {
                    const double probability = static_cast<double>(counts[i]) / length_d;
                    entropy -= probability * std::log2(probability);
                }
            }

            return entropy;
        }

        // Check if buffer contains mostly printable ASCII/UTF-8 text
        inline bool is_text_content(const unsigned char* buffer, size_t length)
        {
            if (length == 0) return false;

            size_t printable_count = 0;
            size_t whitespace_count = 0;
            size_t control_count = 0;

            for (size_t i = 0; i < length; ++i)
            {
                const unsigned char ch = buffer[i];

                // Common whitespace characters
                if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
                {
                    whitespace_count++;
                    printable_count++;
                }
                // Printable ASCII range
                else if (ch >= 32 && ch <= 126)
                {
                    printable_count++;
                }
                // UTF-8 continuation bytes (10xxxxxx)
                else if ((ch & 0xC0) == 0x80)
                {
                    printable_count++;
                }
                // UTF-8 multi-byte sequence starts (110xxxxx, 1110xxxx, 11110xxx)
                else if ((ch & 0xE0) == 0xC0 || (ch & 0xF0) == 0xE0 || (ch & 0xF8) == 0xF0)
                {
                    printable_count++;
                }
                // Control characters (excluding common whitespace)
                else if (ch < 32)
                {
                    control_count++;
                }
            }

            // Consider as text if >90% printable and <10% control chars
            const double printable_ratio = static_cast<double>(printable_count) / length;
            const double control_ratio = static_cast<double>(control_count) / length;

            return printable_ratio > 0.90 && control_ratio < 0.10;
        }

        // Check for XML/HTML markers
        inline bool is_xml_content(const unsigned char* buffer, size_t length)
        {
            if (length < 5) return false;

            const char* str = reinterpret_cast<const char*>(buffer);

            // Check for "<?xml" (case-insensitive)
            if (length >= 5 && buffer[0] == '<' && buffer[1] == '?')
            {
                if (iequals_n(str + 2, "xml", 3))
                    return true;
            }

            // Check for HTML doctype (case-insensitive)
            if (length >= 9 && buffer[0] == '<' && buffer[1] == '!')
            {
                if (iequals_n(str + 2, "DOCTYPE", 7))
                    return true;
            }

            // Check for HTML tags (case-insensitive)
            if (length >= 6 && buffer[0] == '<')
            {
                if (iequals_n(str + 1, "html>", 5) || iequals_n(str + 1, "html ", 5))
                    return true;
            }

            return false;
        }

        // Special check for RIFF-based formats (WAV, AVI, WEBP)
        inline file_content_type check_riff_type(const unsigned char* buffer, size_t length)
        {
            if (length < 12) return file_content_type::UNKNOWN;

            // RIFF format: "RIFF" + size (4 bytes) + format type (4 bytes)
            if (std::memcmp(buffer + 8, "WAVE", 4) == 0)
                return file_content_type::AUDIO;
            else if (std::memcmp(buffer + 8, "AVI ", 4) == 0)
                return file_content_type::VIDEO;
            else if (std::memcmp(buffer + 8, "WEBP", 4) == 0)
                return file_content_type::IMAGE;

            return file_content_type::UNKNOWN;
        }

        // Check if ZIP is actually an Office document (DOCX, XLSX, PPTX)
        inline file_content_type check_office_type(const std::string& filename)
        {
            if (has_extension(filename, ".docx") ||
                has_extension(filename, ".xlsx") ||
                has_extension(filename, ".pptx"))
            {
                return file_content_type::OFFICE;
            }

            return file_content_type::COMPRESSED;
        }
    }

    // ---------------------------------------------------------------------------------

    inline bool detect_file_type(
        const std::string& filename,
        file_content_type& detected_type
    )
    {
        detected_type = file_content_type::UNKNOWN;

        // Open file in binary mode
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            return false;

        // Read initial bytes for analysis (8KB should be sufficient)
        constexpr size_t BUFFER_SIZE = 8192;
        std::array<unsigned char, BUFFER_SIZE> buffer;

        file.read(reinterpret_cast<char*>(buffer.data()), BUFFER_SIZE);
        const size_t bytes_read = static_cast<size_t>(file.gcount());
        file.close();

        if (bytes_read == 0)
            return false;

        // Step 1: Check for known magic number signatures
        for (const auto& sig : impl::signatures)
        {
            if (bytes_read >= sig.offset + sig.length)
            {
                if (std::memcmp(buffer.data() + sig.offset, sig.bytes, sig.length) == 0)
                {
                    detected_type = sig.type;

                    // Special handling for RIFF-based formats
                    if (sig.bytes == impl::sig_webp || sig.bytes == impl::sig_wav ||
                        sig.bytes == impl::sig_avi)
                    {
                        const auto riff_type = impl::check_riff_type(buffer.data(), bytes_read);
                        if (riff_type != file_content_type::UNKNOWN)
                            detected_type = riff_type;
                    }

                    // Special handling for ZIP (could be Office document)
                    if (detected_type == file_content_type::COMPRESSED &&
                        sig.bytes == impl::sig_zip)
                    {
                        detected_type = impl::check_office_type(filename);
                    }

                    // Binary types
                    return false;
                }
            }
        }

        // Step 2: Check for XML/HTML content
        if (impl::is_xml_content(buffer.data(), bytes_read))
        {
            detected_type = file_content_type::TEXT_XML;
            return true;
        }

        // Step 3: Calculate entropy to distinguish text from binary
        const double entropy = impl::calculate_entropy(buffer.data(), bytes_read);

        // Step 4: Use heuristics to classify content
        // Entropy thresholds:
        //   < 5.0  : Likely plain text
        //   5.0-6.8: Could be text or structured binary
        //   > 6.8  : Likely compressed/encrypted/random binary

        const bool is_text = impl::is_text_content(buffer.data(), bytes_read);

        if (is_text && entropy < 6.5)
        {
            // High probability of plain text (< 5.5)
            // Or could be text with some binary content (e.g., source code with special chars)
            detected_type = file_content_type::TEXT_PLAIN;
            return true;
        }
       
        // Likely binary content (no recognized format)
        detected_type = file_content_type::UNKNOWN;
        return false;
    }

    // ---------------------------------------------------------------------------------   

    // Compute Levenshtein (edit) distance between two token sequences
    inline size_t edit_distance(const std::vector<int>& tokens1, const std::vector<int>& tokens2)
    {
        const size_t len1 = tokens1.size();
        const size_t len2 = tokens2.size();

        if (len1 == 0) return len2;
        if (len2 == 0) return len1;

        // DP table: dp[i][j] = edit distance between tokens1[0..i-1] and tokens2[0..j-1]
        std::vector<std::vector<size_t>> dp(len1 + 1, std::vector<size_t>(len2 + 1));

        // Initialize base cases
        for (size_t i = 0; i <= len1; ++i)
            dp[i][0] = i;
        for (size_t j = 0; j <= len2; ++j)
            dp[0][j] = j;

        // Fill DP table
        for (size_t i = 1; i <= len1; ++i) {
            for (size_t j = 1; j <= len2; ++j) {
                if (tokens1[i - 1] == tokens2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];  // No edit needed
                }
                else {
                    dp[i][j] = 1 + std::min({ dp[i - 1][j],     // Deletion
                                             dp[i][j - 1],      // Insertion
                                             dp[i - 1][j - 1]   // Substitution
                        });
                }
            }
        }

        return dp[len1][len2];
    }
    
    // Compute normalized edit distance as a similarity score between 0 and 1
    inline double normalized_edit_similarity(const std::vector<int>& tokens1, const std::vector<int>& tokens2)
    {
        if (tokens1.empty() && tokens2.empty())
            return 1.0;

        const size_t max_len = std::max(tokens1.size(), tokens2.size());
        if (max_len == 0)
            return 1.0;

        const size_t dist = edit_distance(tokens1, tokens2);
        return 1.0 - (static_cast<double>(dist) / max_len);
    }

    // Compute token-level precision, recall, and F1-score
    struct token_overlap_metrics
    {
        double precision;  // What fraction of generated tokens appear in reference
        double recall;     // What fraction of reference tokens appear in generated
        double f1_score;   // Harmonic mean of precision and recall

        void print() const
        {
            std::cout << "Token overlap metrics:\n"
                << "  Precision: " << std::fixed << std::setprecision(4) << (precision * 100.0) << "%\n"
                << "  Recall:    " << std::fixed << std::setprecision(4) << (recall * 100.0) << "%\n"
                << "  F1-score:  " << std::fixed << std::setprecision(4) << (f1_score * 100.0) << "%\n";
        }
    };

    inline token_overlap_metrics compute_token_overlap(
        const std::vector<int>& reference,
        const std::vector<int>& generated)
    {
        token_overlap_metrics metrics{ 0.0, 0.0, 0.0 };

        if (reference.empty() || generated.empty())
            return metrics;

        // Count matching tokens
        std::multiset<int> ref_tokens(reference.begin(), reference.end());
        std::multiset<int> gen_tokens(generated.begin(), generated.end());

        size_t matches = 0;
        for (int token : gen_tokens) {
            auto it = ref_tokens.find(token);
            if (it != ref_tokens.end()) {
                ++matches;
                ref_tokens.erase(it);  // Remove to handle duplicates correctly
            }
        }

        // Calculate precision and recall
        metrics.precision = static_cast<double>(matches) / generated.size();
        metrics.recall = static_cast<double>(matches) / reference.size();

        // Calculate F1-score
        if (metrics.precision + metrics.recall > 0.0) {
            metrics.f1_score = 2.0 * (metrics.precision * metrics.recall) /
                (metrics.precision + metrics.recall);
        }

        return metrics;
    }

    // Compute BLEU-like n-gram overlap score
    inline double compute_ngram_overlap(
        const std::vector<int>& reference,
        const std::vector<int>& generated,
        int max_n = 4)
    {
        if (reference.empty() || generated.empty())
            return 0.0;

        double total_score = 0.0;
        int valid_n_count = 0;

        // Compute overlap for n-grams of size 1 to max_n
        for (int n = 1; n <= max_n; ++n) {
            if (static_cast<size_t>(n) > reference.size() ||
                static_cast<size_t>(n) > generated.size())
                break;

            // Extract n-grams from reference
            std::map<std::vector<int>, size_t> ref_ngrams;
            for (size_t i = 0; i <= reference.size() - n; ++i) {
                std::vector<int> ngram(reference.begin() + i, reference.begin() + i + n);
                ref_ngrams[ngram]++;
            }

            // Count matching n-grams in generated
            size_t matches = 0;
            size_t total_gen_ngrams = 0;
            for (size_t i = 0; i <= generated.size() - n; ++i) {
                std::vector<int> ngram(generated.begin() + i, generated.begin() + i + n);
                total_gen_ngrams++;

                auto it = ref_ngrams.find(ngram);
                if (it != ref_ngrams.end() && it->second > 0) {
                    matches++;
                    it->second--;  // Decrement to handle multiple occurrences
                }
            }

            if (total_gen_ngrams > 0) {
                total_score += static_cast<double>(matches) / total_gen_ngrams;
                valid_n_count++;
            }
        }

        // Return average n-gram precision
        return valid_n_count > 0 ? total_score / valid_n_count : 0.0;
    }

    // Text similarity report
    struct text_similarity_report
    {
        double edit_similarity;         // Normalized Levenshtein distance
        token_overlap_metrics overlap;  // Token-level precision/recall/F1
        double ngram_score;             // N-gram overlap (BLEU-like)

        void print() const
        {
            std::cout << "\n=== Text similarity report ===\n";
            std::cout << "Edit similarity (order-sensitive): "
                << std::fixed << std::setprecision(4) << (edit_similarity * 100.0) << "%\n\n";

            overlap.print();

            std::cout << "\nN-gram overlap (BLEU-like): "
                << std::fixed << std::setprecision(4) << (ngram_score * 100.0) << "%\n";
            std::cout << "==============================\n\n";
        }
    };

    inline text_similarity_report compute_text_similarity(
        const std::vector<int>& reference,
        const std::vector<int>& generated)
    {
        text_similarity_report report;

        report.edit_similarity = normalized_edit_similarity(reference, generated);
        report.overlap = compute_token_overlap(reference, generated);
        report.ngram_score = compute_ngram_overlap(reference, generated, 4);

        return report;
    }

    class inference_context
    {
    public:
        inference_context(
            long window_size = 256,
            long context_multiplier = 10,
            long padding_token = 0
        ) : window_size_(window_size),
            context_capacity_(window_size * context_multiplier),
            padding_token_(padding_token),
            current_size_(0)
        {
            DLIB_CASSERT(window_size > 0, "Window size must be positive");
            DLIB_CASSERT(context_multiplier > 0, "Context multiplier must be positive");
            context_.reserve(context_capacity_);
        }

        void add_token(unsigned long token)
        {
            if (current_size_ == context_capacity_)
            {
                // FIFO: remove oldest, add newest
                context_.erase(context_.begin());
                context_.push_back(static_cast<int>(token));
            }
            else
            {
                // Still room in context
                context_.push_back(static_cast<int>(token));
                current_size_++;
            }
        }

        void add_tokens(const std::vector<unsigned long>& tokens)
        {
            for (unsigned long token : tokens) add_token(token);
        }

        void add_tokens(const std::vector<int>& tokens)
        {
            for (int token : tokens) add_token(static_cast<unsigned long>(token));
        }

        matrix<int, 0, 1> get_input_window(long custom_window_size = -1) const
        {
            long win_size = (custom_window_size > 0) ? custom_window_size : window_size_;
            matrix<int, 0, 1> window(win_size, 1);

            if (current_size_ >= win_size)
            {
                // Context has enough tokens - take last win_size tokens
                for (long i = 0; i < win_size; ++i)
                    window(i) = context_[current_size_ - win_size + i];
            }
            else
            {
                // Context has fewer tokens - left pad
                long padding_needed = win_size - current_size_;

                for (long i = 0; i < padding_needed; ++i)
                    window(i) = padding_token_;
                for (long i = 0; i < current_size_; ++i)
                    window(padding_needed + i) = context_[i];
            }

            return window;
        }

        void reset()
        {
            context_.clear();
            current_size_ = 0;
        }

        void resize_context(long new_capacity)
        {
            DLIB_CASSERT(new_capacity > 0, "New capacity must be positive");

            if (new_capacity < current_size_)
            {
                // Keep only the last new_capacity tokens
                context_.erase(context_.begin(), context_.begin() + (current_size_ - new_capacity));
                current_size_ = new_capacity;
            }

            context_capacity_ = new_capacity;
            context_.reserve(context_capacity_);
        }

        long size() const { return current_size_; }
        long capacity() const { return context_capacity_; }
        long window_size() const { return window_size_; }
        bool is_full() const { return current_size_ >= context_capacity_; }
        const std::vector<int>& get_full_context() const { return context_; }

        std::string to_string(bool show_all = false) const
        {
            std::ostringstream ss;
            ss << "InferenceContext[size=" << current_size_
                << "/" << context_capacity_
                << ", window=" << window_size_ << "]\n";

            if (show_all && current_size_ > 0)
            {
                ss << "Tokens: [";
                long display_count = show_all ? current_size_ : std::min(20L, current_size_);
                for (long i = 0; i < display_count; ++i)
                {
                    ss << context_[i];
                    if (i < display_count - 1) ss << ", ";
                }
                if (current_size_ > display_count)
                {
                    ss << " ... +" << (current_size_ - display_count) << " more";
                }
                ss << "]";
            }

            return ss.str();
        }

        friend void serialize(const inference_context& item, std::ostream& out)
        {
            serialize("inference_context", out);
            serialize(item.window_size_, out);
            serialize(item.context_capacity_, out);
            serialize(item.padding_token_, out);
            serialize(item.current_size_, out);
            serialize(item.context_, out);
        }

        friend void deserialize(inference_context& item, std::istream& in)
        {
            std::string name;
            deserialize(name, in);
            if (name != "inference_context")
            {
                throw serialization_error("Error deserializing object of type 'inference_context': "
                    "expected 'inference_context' but got '" + name + "'");
            }

            deserialize(item.window_size_, in);
            deserialize(item.context_capacity_, in);
            deserialize(item.padding_token_, in);
            deserialize(item.current_size_, in);
            deserialize(item.context_, in);
        }

    private:
        std::vector<int> context_;      // Full context history
        long window_size_;               // Window size for model input
        long context_capacity_;          // Maximum context size
        long padding_token_;             // Token used for left padding
        long current_size_;              // Current number of tokens
    };

    inline void build_single_token_prediction_dataset(
        const std::vector<std::vector<int>>& token_sequences,
        long window_len,
        long padding_token,
        bool use_left_padding,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<unsigned long>& Y)
    {
        X.clear();
        Y.clear();

        for (const auto& seq : token_sequences)
        {
            const long len = static_cast<long>(seq.size());
            if (len <= 1) continue;

            long start = 0;
            if (len < window_len)
            {
                if (!use_left_padding) continue;
                start = (len - window_len);
            }

            // Generate initial padded samples for sequences >= window_len
            if (use_left_padding && len >= window_len)
            {
                for (long pos = 1; pos < window_len; ++pos)
                {
                    matrix<int, 0, 1> window(window_len, 1);
                    long pad = window_len - pos;

                    for (long i = 0; i < pad; ++i) window(i) = padding_token;
                    for (long i = 0; i < pos; ++i) window(pad + i) = seq[i];

                    X.push_back(window);
                    Y.push_back(seq[pos]);
                }
            }

            // Slide window through sequence
            for (long pos = start; pos < len - 1; ++pos)
            {
                matrix<int, 0, 1> window(window_len, 1);

                for (long i = 0; i < window_len; ++i)
                {
                    long idx = pos + i;
                    window(i) = (idx >= 0 && idx < len) ? seq[idx] : padding_token;
                }

                long target_idx = pos + window_len;
                if (target_idx >= 0 && target_idx < len)
                {
                    X.push_back(window);
                    Y.push_back(seq[target_idx]);
                }
            }
        }
    }

    inline void build_multi_token_prediction_dataset(
        const std::vector<std::vector<int>>& source_sequences,
        const std::vector<std::vector<int>>& target_sequences,
        long src_window_len,
        long tgt_window_len,
        long padding_token,
        std::vector<matrix<int, 0, 1>>& X,
        std::vector<matrix<unsigned long, 0, 1>>& Y)
    {
        DLIB_CASSERT(source_sequences.size() == target_sequences.size(),
            "Source and target must have same size");

        X.clear();
        Y.clear();

        for (size_t i = 0; i < source_sequences.size(); ++i)
        {
            const auto& src = source_sequences[i];
            const auto& tgt = target_sequences[i];

            const long src_len = static_cast<long>(src.size());
            const long tgt_len = static_cast<long>(tgt.size());

            if (src_len == 0 || tgt_len == 0) continue;

            long src_pos = (src_len < src_window_len) ? (src_len - src_window_len) : 0;
            long tgt_pos = 0;

            while (true)
            {
                // Build source window
                matrix<int, 0, 1> src_window(src_window_len, 1);
                long src_real = 0;

                for (long j = 0; j < src_window_len; ++j)
                {
                    long idx = src_pos + j;
                    if (idx >= 0 && idx < src_len)
                    {
                        src_window(j) = src[idx];
                        src_real++;
                    }
                    else
                    {
                        src_window(j) = padding_token;
                    }
                }

                // Build target window
                matrix<unsigned long, 0, 1> tgt_window(tgt_window_len, 1);
                long tgt_real = 0;

                for (long j = 0; j < tgt_window_len; ++j)
                {
                    long idx = tgt_pos + j;
                    if (idx < tgt_len)
                    {
                        tgt_window(j) = tgt[idx];
                        tgt_real++;
                    }
                    else
                    {
                        tgt_window(j) = padding_token;
                    }
                }

                // Stop if no real tokens in either window
                if (src_real == 0 || tgt_real == 0) break;

                X.push_back(src_window);
                Y.push_back(tgt_window);

                // Stop if both sequences fully consumed
                if (src_pos + src_window_len >= src_len &&
                    tgt_pos + tgt_window_len >= tgt_len) break;

                src_pos++;
                tgt_pos++;
            }
        }
    }

    template <typename sample_type, typename label_type>
    void shuffle_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        unsigned long seed = 0)
    {
        DLIB_CASSERT(samples.size() == labels.size(),
            "samples and labels must have the same size");

        const size_t dataset_size = samples.size();
        if (dataset_size <= 1) return;

        dlib::rand rng;
        if (seed != 0) rng = dlib::rand(seed);

        // Fisher-Yates shuffle algorithm
        for (size_t i = dataset_size - 1; i > 0; --i)
        {
            size_t j = rng.get_random_32bit_number() % (i + 1);

            // Swap samples[i] with samples[j]
            std::swap(samples[i], samples[j]);

            // Swap labels[i] with labels[j]
            std::swap(labels[i], labels[j]);
        }
    }

    template <typename sample_type, typename label_type>
    void augment_training_dataset(
        std::vector<sample_type>& samples,
        std::vector<label_type>& labels,
        int unk_token,
        int padding_token,
        double augmentation_ratio = 0.2,
        long min_noise_tokens = 1,
        long max_noise_tokens = 3,
        unsigned long seed = 0)
    {
        DLIB_CASSERT(samples.size() == labels.size(),
            "samples and labels must have the same size");
        DLIB_CASSERT(augmentation_ratio >= 0.0 && augmentation_ratio <= 2.0,
            "augmentation_ratio must be between 0.0 and 2.0");
        DLIB_CASSERT(min_noise_tokens >= 0 && max_noise_tokens >= min_noise_tokens,
            "Invalid noise token range: min=" << min_noise_tokens << ", max=" << max_noise_tokens);

        const size_t original_size = samples.size();
        if (original_size == 0 || augmentation_ratio == 0.0) return;

        // Calculate number of augmented samples to create
        const size_t num_augmented = static_cast<size_t>(original_size * augmentation_ratio);
        if (num_augmented == 0) return;

        // Reserve space to avoid multiple reallocations
        samples.reserve(original_size + num_augmented);
        labels.reserve(original_size + num_augmented);

        dlib::rand rng;
        if (seed != 0) rng = dlib::rand(seed);

        for (size_t aug_idx = 0; aug_idx < num_augmented; ++aug_idx)
        {
            // Select a random sample to augment
            const size_t source_idx = rng.get_random_32bit_number() % original_size;

            // Create a copy of the sample and its label
            auto augmented_sample = samples[source_idx];
            auto augmented_label = labels[source_idx];

            // Identify non-padding positions in the sample
            std::vector<long> valid_positions;
            const long sample_length = augmented_sample.nr();

            for (long i = 0; i < sample_length; ++i)
            {
                if (augmented_sample(i) != padding_token)
                    valid_positions.push_back(i);
            }

            // Skip if no valid positions to add noise
            if (valid_positions.empty()) continue;

            // Determine number of tokens to replace with noise
            const long num_valid = static_cast<long>(valid_positions.size());
            const long effective_max = std::min(max_noise_tokens, num_valid);
            const long effective_min = std::min(min_noise_tokens, effective_max);

            long num_noise = effective_min;
            if (effective_max > effective_min)
            {
                num_noise = effective_min +
                    (rng.get_random_32bit_number() % (effective_max - effective_min + 1));
            }

            // Ensure noise ratio is reasonable (max 30% of non-padding tokens)
            const long max_reasonable = std::max(1L, static_cast<long>(num_valid * 0.3));
            num_noise = std::min(num_noise, max_reasonable);

            // Randomly select positions to replace with UNK
            std::vector<long> noise_positions = valid_positions;

            // Fisher-Yates shuffle to select random positions
            for (long i = static_cast<long>(noise_positions.size()) - 1; i > 0; --i)
            {
                long j = rng.get_random_32bit_number() % (i + 1);
                std::swap(noise_positions[i], noise_positions[j]);
            }

            // Apply noise to the first num_noise positions
            for (long i = 0; i < num_noise; ++i)
            {
                augmented_sample(noise_positions[i]) = unk_token;
            }

            // Add augmented sample and label to the dataset
            samples.push_back(std::move(augmented_sample));
            labels.push_back(std::move(augmented_label));
        }
    }

} // namespace dlib

#endif // DLIB_LANGUAGE_MODEL_DATA_H_