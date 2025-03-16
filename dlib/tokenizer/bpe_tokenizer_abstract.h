// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_BPE_TOKENIZER_ABSTRACT_
#ifdef DLIB_BPE_TOKENIZER_ABSTRACT_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <mutex>
#include <thread>
#include <future>
#include <queue>

namespace dlib
{

    class bpe_tokenizer
    {
        /*!
            CLASS bpe_tokenizer
                A Byte Pair Encoding (BPE) tokenizer for text processing.

                This class implements a Byte Pair Encoding (BPE) tokenizer, which is a subword
                tokenization algorithm commonly used in natural language processing (NLP). The
                BPE algorithm iteratively merges the most frequent pairs of bytes or characters
                to form a vocabulary of subword units. This approach is particularly useful for
                handling out-of-vocabulary words and reducing the size of the vocabulary while
                maintaining the ability to represent any text.

                The tokenizer supports special tokens, which can be used to mark specific elements
                in the text (e.g., `<text>`, `<url>`, `<image>`, etc.). These special tokens are
                treated as atomic units during tokenization and are not subject to further splitting.

                The class provides methods for training the tokenizer on a given text corpus, encoding
                text into subword tokens, and decoding tokens back into text. The tokenizer can be
                serialized and deserialized to/from a file, allowing for easy storage and reuse.

                INITIAL VALUE
                    - The base vocabulary is initialized with single-byte tokens (0-255).
                    - Special tokens are pre-defined and assigned IDs starting from 256.
                    - The maximum token length is set to 8 bytes.

                WHAT THIS OBJECT REPRESENTS
                    This object represents a BPE tokenizer capable of encoding and decoding text
                    using a learned subword vocabulary. It is designed to handle UTF-8 encoded text
                    and supports multi-threaded processing for efficient tokenization.

                REFERENCES
                    - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of
                      Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of
                      the Association for Computational Linguistics (ACL 2016).
        !*/
    public:
        bpe_tokenizer();
        /*!
            ensures
                - Initializes the tokenizer with a base vocabulary of single-byte tokens (0-255).
                - Pre-defines special tokens and assigns them unique IDs starting from 256.
        !*/

        void train(
            const std::string& text,
            int vocab_size,
            bool verbose = false
        );
        /*!
            requires
                - vocab_size >= 256
            ensures
                - Trains the tokenizer on the provided text corpus.
                - Iteratively merges the most frequent pairs of tokens to form a subword vocabulary
                  of size `vocab_size`.
                - If `verbose` is true, progress information is printed to the standard output.
        !*/

        std::vector<int> encode(
            const std::string& text
        );
        /*!
            ensures
                - Encodes the input text into a sequence of subword tokens.
                - Special tokens are automatically added to mark the beginning and end of paragraphs.
                - Returns a vector of token IDs representing the encoded text.
        !*/

        std::string decode(
            const std::vector<int>& ids,
            bool display_special_tokens = true
        );
        /*!
            ensures
                - Decodes a sequence of token IDs back into a human-readable string.
                - If `display_special_tokens` is true, special tokens are included in the output.
                - Returns the decoded text as a UTF-8 encoded string.
        !*/

        void serialize(
            const bpe_tokenizer& tok,
            std::ostream& out
        );
        /*!
            ensures
                - Serializes the tokenizer's vocabulary and merge operations to the output stream.
                - The serialized data can be used to reconstruct the tokenizer later.
        !*/

        void deserialize(
            bpe_tokenizer& tok,
            std::istream& in
        );
        /*!
            ensures
                - Deserializes the tokenizer's vocabulary and merge operations from the input stream.
                - Restores the tokenizer to the state it was in when serialized.
        !*/

        int get_special_token_id(
            const std::string& token
        ) const;
        /*!
            ensures
                - Returns the ID of the specified special token.
                - Throws an exception if the token is not found in the special tokens map.
        !*/

        size_t get_vocab_size() const;
        /*!
            ensures
                - Returns the total size of the vocabulary, including base tokens and special tokens.
        !*/

    private:
        // Private implementation details
        std::map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> special_token_map;
        std::map<std::pair<int, int>, int> merges;
        std::map<int, std::vector<uint8_t>> vocab;
        int vocab_size;

        static const size_t MAX_TOKEN_LENGTH = 8;
        static const int BASE_VOCAB_SIZE = 256;

        // Helper functions
        std::unordered_map<std::pair<int, int>, int, pair_hash> get_stats(const std::vector<int>& ids);
        std::pair<int, int> get_most_frequent_pair(const std::unordered_map<std::pair<int, int>, int, pair_hash>& stats);
        std::vector<int> merge(std::vector<int>& ids, const std::pair<int, int>& pair, int idx);
        std::string bytes_to_string(const std::vector<uint8_t>& bytes);
        std::vector<uint8_t> string_to_bytes(const std::string& str);
    };

}

#endif // DLIB_BPE_TOKENIZER_ABSTRACT_